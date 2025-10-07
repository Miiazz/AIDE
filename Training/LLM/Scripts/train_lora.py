import torch, datasets, transformers, peft, os, json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import default_data_collator

#=== To improve Efficency (hopefully) - 09/30/25
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# === Config ===
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"  # Dynamic pull from HF
DATASET_PATH = "<Your Dataset>"           # Input your JSONL dataset here
OUTPUT_DIR = "./lora_output"

torch.manual_seed(42)

# === Load tokenizer and model from Hugging Face ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(
    
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)
model.config.return_dict = True
model.config.use_cache = False

# === Load dataset with error handling
# === Load dataset with robust JSONL parsing
def load_jsonl_robust(file_path):
    """Load JSONL with line-by-line error handling for malformed JSON"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_json = ""
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            current_json += line
            try:
                # Try to parse as complete JSON
                sample = json.loads(current_json)
                samples.append(sample)
                current_json = ""
            except json.JSONDecodeError:
                # Continue building the JSON object
                continue
    
    print(f"Loaded {len(samples)} samples from {file_path}")
    return samples

# Load the consolidated dataset
try:
    all_samples = load_jsonl_robust(DATASET_PATH)
except Exception as e:
    print(f"Error loading {DATASET_PATH}: {e}")
    all_samples = []

print(f"Total samples loaded: {len(all_samples)}")

# Convert to dataset
dataset = Dataset.from_list(all_samples)

# === Proper Phi-3 chat template tokenization with precise masking
def apply_template(example):
    messages = example["messages"]
    
    # Apply the full chat template to get proper conversation structure
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        return_tensors=None
    )
    
    # Tokenize the complete conversation
    model_inputs = tokenizer(
        full_text,
        truncation=True,
        max_length=2048,
        padding=False
    )
    
    input_ids = model_inputs["input_ids"]
    labels = [-100] * len(input_ids)  # Start with everything masked
    
    # Now we need to find and unmask only the assistant response content
    # We'll do this by finding the assistant responses in the original text
    # and mapping them to token positions
    
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            # Create conversation up to (but not including) this assistant message
            context_messages = messages[:i+1]  # Include the assistant message
            context_without_assistant = messages[:i]  # Exclude the assistant message
            
            # Get text up to the assistant response
            if context_without_assistant:
                prefix_text = tokenizer.apply_chat_template(
                    context_without_assistant,
                    tokenize=False,
                    return_tensors=None
                )
                # Add the assistant marker
                prefix_text += "<|assistant|>"
            else:
                # First message is assistant (unusual but handle it)
                prefix_text = "<|assistant|>"
            
            # Get the full context including assistant response
            full_context_text = tokenizer.apply_chat_template(
                context_messages,
                tokenize=False,
                return_tensors=None
            )
            
            # Tokenize both to find the boundaries
            prefix_tokens = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            full_context_tokens = tokenizer(full_context_text, add_special_tokens=False)["input_ids"]
            
            # The assistant response is between the context prefix and full context
            # INCLUDE special tokens in training for complete generation capability
            
            # Find the start of the assistant response (after the prefix without assistant)
            if context_without_assistant:
                true_prefix_text = tokenizer.apply_chat_template(
                    context_without_assistant,
                    tokenize=False,
                    return_tensors=None
                )
                true_prefix_tokens = tokenizer(true_prefix_text, add_special_tokens=False)["input_ids"]
                start_idx = len(true_prefix_tokens)
            else:
                start_idx = 0  # Assistant is first message
            
            # End at the full context (includes <|end|> token)
            end_idx = len(full_context_tokens)
            
            # Ensure we don't go beyond the actual input_ids length
            start_idx = min(start_idx, len(input_ids))
            end_idx = min(end_idx, len(input_ids))
            
            # Unmask the complete assistant response INCLUDING special tokens
            # This teaches both Aide's personality AND proper response format
            for j in range(start_idx, end_idx):
                labels[j] = input_ids[j]
    
    return {
        "input_ids": input_ids,
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels
    }

# Create Train / Test split
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Tokenize both splits
train_ds = split_dataset["train"].map(apply_template, remove_columns=split_dataset["train"].column_names)
val_ds   = split_dataset["test"].map(apply_template, remove_columns=split_dataset["test"].column_names)

# === LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,  # Changed from 16 - stronger adaptation for personality training
    target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"], #Verified via find_lora_targets.py
    lora_dropout=0.01,  # Reduced from 0.05 - less interference with personality learning
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()             # redundant but explicit
model.enable_input_require_grads()
model.print_trainable_parameters()
assert any(p.requires_grad for p in model.parameters()), "No trainable parameters!"

# === Training arguments     # Patching to prevent NaNs
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=1e-4,  # More appropriate for LoRA - was 3e-4 (too hot)
    lr_scheduler_type="cosine",  # Better for personality training
    warmup_ratio=0.05,  # Increased warmup for stability with proper LR
    bf16=True,             #<- It does not like when this is false
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    logging_dir="./logs",
    max_grad_norm=1.0,
    gradient_checkpointing=True,  # <--- Changed this line 08/04/25 & it worked
    gradient_checkpointing_kwargs={"use_reentrant": False},   # <--- Added this 9/30/25 to compensate for heftier training on V1 vs lite
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
    weight_decay=0.05, # Helps with generalization and prevents overfitting
)

# === Trainer
trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=default_data_collator  # Dynamic padding - more efficient
)

#===Debug - Test loss with proper training mask
batch = next(iter(train_ds))
input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(model.device)
attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0).to(model.device)
# Use the properly masked labels from our tokenization function
labels = torch.tensor(batch["labels"]).unsqueeze(0).to(model.device)

model.eval()
with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print("Test Loss (assistant-only):", output.loss.item())
    
    # Count how many tokens are actually being trained on
    training_tokens = (labels != -100).sum().item()
    total_tokens = labels.numel()
    print(f"Training on {training_tokens}/{total_tokens} tokens ({training_tokens/total_tokens*100:.1f}%)")

# === Train
model.train()
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
