import torch, datasets, transformers, peft, os, json
from copy import deepcopy
from typing import List, Dict, Any
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
MAX_SEQ_LENGTH = 2048

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

def ensure_eos(text: str) -> str:
    if text.endswith(tokenizer.eos_token):
        return text
    return text + tokenizer.eos_token


def trim_prompt_messages(messages: List[Dict[str, Any]], response: str) -> List[Dict[str, Any]]:
    """Trim older turns until prompt+response fits the context window."""
    trimmed = deepcopy(messages)

    if not trimmed:
        return trimmed

    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    # Account for the EOS token that will be appended via text_target
    response_length = len(response_ids) + 1

    while trimmed:
        prompt_ids = tokenizer.apply_chat_template(
            trimmed,
            tokenize=True,
            add_generation_prompt=True
        )["input_ids"]

        if len(prompt_ids) + response_length <= MAX_SEQ_LENGTH:
            return trimmed

        # Drop the oldest non-system message
        drop_idx = None
        for idx, message in enumerate(trimmed):
            if message["role"] != "system":
                drop_idx = idx
                break

        if drop_idx is None:
            # Only system messages remain; drop the first one
            trimmed = trimmed[1:]
        else:
            trimmed = trimmed[drop_idx + 1 :]

    return trimmed


def conversation_to_examples(messages: List[Dict[str, Any]]):
    context: List[Dict[str, Any]] = []
    examples = []

    for message in messages:
        role = message.get("role")
        if role == "assistant":
            prompt_messages = trim_prompt_messages(context, message.get("content", ""))
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            response_text = ensure_eos(message.get("content", ""))

            tokenized = tokenizer(
                prompt_text,
                text_target=response_text,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                padding=False
            )

            if any(label != -100 for label in tokenized["labels"]):
                examples.append({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": tokenized["labels"]
                })

        context.append(message)

    return examples


processed_examples = []
for sample in dataset:
    processed_examples.extend(conversation_to_examples(sample["messages"]))

processed_dataset = Dataset.from_list(processed_examples)

# Create Train / Test split
split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)

train_ds = split_dataset["train"]
val_ds = split_dataset["test"]

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
if len(train_ds) == 0:
    raise ValueError("No training samples available after preprocessing. Check dataset and filters.")

batch = train_ds[0]
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
