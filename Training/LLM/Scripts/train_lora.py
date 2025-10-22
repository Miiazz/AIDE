import torch, datasets, transformers, peft, os, json
from copy import deepcopy
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from transformers import default_data_collator

#=== To improve Efficency (hopefully) - 09/30/25
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# === Config ===
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"  # Dynamic pull from HF
DATASET_PATH = "<Your JSONL File Path>"           # Input your JSONL dataset here
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

# === Load dataset with error handling & robust JSONL parsing
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
        )

        if len(prompt_ids) + response_length <= MAX_SEQ_LENGTH:
            return trimmed

        # Drop the oldest non-system message
        drop_idx = None
        for idx, message in enumerate(trimmed):
            if message["role"] != "system":
                drop_idx = idx
                break

        if drop_idx is None:
            # Only system messages remain; NEVER drop system messages
            # If we can't fit even with system only, break the loop
            break
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

            # Combine prompt and response for proper tokenization
            full_text = prompt_text + response_text
            
            # Tokenize prompt and response separately to control truncation
            prompt_tokenized = tokenizer(prompt_text, add_special_tokens=False)
            response_tokenized = tokenizer(response_text, add_special_tokens=False)
            
            # Check if combined length exceeds max
            total_length = len(prompt_tokenized["input_ids"]) + len(response_tokenized["input_ids"])
            
            if total_length > MAX_SEQ_LENGTH:
                # Truncate response if needed, but preserve full prompt (especially system message)
                max_response_length = MAX_SEQ_LENGTH - len(prompt_tokenized["input_ids"])
                if max_response_length < 10:  # Need at least 10 tokens for response
                    continue  # Skip this example if prompt is too long
                response_tokenized["input_ids"] = response_tokenized["input_ids"][:max_response_length]
                response_tokenized["attention_mask"] = response_tokenized["attention_mask"][:max_response_length]
            
            # Combine tokens manually 
            input_ids = prompt_tokenized["input_ids"] + response_tokenized["input_ids"]
            attention_mask = prompt_tokenized["attention_mask"] + response_tokenized["attention_mask"]
            
            # Create labels - mask the prompt, train on response
            prompt_length = len(prompt_tokenized["input_ids"])
            labels = [-100] * len(input_ids)
            
            # Only train on the response portion
            if prompt_length < len(input_ids):
                labels[prompt_length:] = input_ids[prompt_length:]
            
            if any(label != -100 for label in labels):
                examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })

        context.append(message)

    return examples


processed_examples = []
total_conversations = len(dataset)
for idx, sample in enumerate(dataset):
    examples = conversation_to_examples(sample["messages"])
    processed_examples.extend(examples)
    
    # Log first few examples for validation
    if idx < 3 and examples:
        print(f"\n=== Conversation {idx + 1} Example Validation ===")
        for ex_idx, example in enumerate(examples[:2]):  # Show max 2 examples per conversation
            # Decode the prompt (everything before labels start)
            input_ids = example["input_ids"]
            labels = example["labels"]
            
            # Find where the response starts (first non-masked token)
            response_start = None
            for i, label in enumerate(labels):
                if label != -100:
                    response_start = i
                    break
            
            if response_start is not None:
                prompt_tokens = input_ids[:response_start]
                response_tokens = [input_ids[i] for i, label in enumerate(labels) if label != -100]
                
                prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
                response_text = tokenizer.decode(response_tokens, skip_special_tokens=False)
                
                # Check if system prompt is preserved - was getting cut off 
                full_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
                has_aide_identity = "You are Aide" in full_prompt
                
                print(f"  Example {ex_idx + 1}:")
                print(f"    System prompt preserved: {'✓' if has_aide_identity else '✗'}")
                print(f"    Prompt length: {len(prompt_tokens)} tokens")
                print(f"    Prompt sample: {repr(full_prompt[:100])}...")  # First 100 chars
                print(f"    Response: {repr(response_text)}")
                print(f"    Training tokens: {sum(1 for l in labels if l != -100)}/{len(labels)}")

print(f"\nDataset Processing Complete:")
print(f"  Original conversations: {total_conversations}")
print(f"  Generated training examples: {len(processed_examples)}")
print(f"  Average examples per conversation: {len(processed_examples)/total_conversations:.1f}")

processed_dataset = Dataset.from_list(processed_examples)

# Create Train / Test split
split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)

train_ds = split_dataset["train"]
val_ds = split_dataset["test"]

print(f"\nTrain/Test Split:")
print(f"  Training examples: {len(train_ds)}")
print(f"  Validation examples: {len(val_ds)}")

# Analyze sequence lengths
train_lengths = [len(example["input_ids"]) for example in train_ds]
if train_lengths:
    print(f"  Sequence length stats:")
    print(f"    Min: {min(train_lengths)}, Max: {max(train_lengths)}")
    print(f"    Average: {sum(train_lengths)/len(train_lengths):.1f}")
    print(f"    Examples at max length ({MAX_SEQ_LENGTH}): {sum(1 for l in train_lengths if l == MAX_SEQ_LENGTH)}")

# === LoRA config - Optimized for Pi5 Q5 performance with intelligence retention
lora_config = LoraConfig(
    r=48,  # Balanced: enough capacity for personality, not too complex for Q5
    lora_alpha=72,  # Strong adaptation but preserves base intelligence
    target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"], #Verified via find_lora_targets.py
    lora_dropout=0.03,  # Lower dropout to preserve intelligence pathways
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()             # redundant but explicit
model.enable_input_require_grads()
model.print_trainable_parameters()
assert any(p.requires_grad for p in model.parameters()), "No trainable parameters!"

# Custom callback for training progress logging
class ValidationCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"\nEpoch {state.epoch} Summary:")
            print(f"  Train Loss: {logs.get('train_loss', 'N/A'):.4f}")
            print(f"  Eval Loss: {logs.get('eval_loss', 'N/A'):.4f}")
            print(f"  Learning Rate: {logs.get('learning_rate', 'N/A'):.2e}")

# === Training arguments - Balanced for personality + intelligence retention on Pi5
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,  # Conservative to preserve base intelligence
    learning_rate=4e-5,  # Moderate LR for personality without overwhelming intelligence
    lr_scheduler_type="cosine",  # Cosine for gentler learning curve
    warmup_ratio=0.15,  # Extended warmup to preserve knowledge
    bf16=True,             #<- It does not like when this is false
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",  # Evaluate each epoch
    save_total_limit=3,     # Keep only last 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    logging_dir="./logs",
    max_grad_norm=0.3,  # Gentler clipping to preserve complex reasoning
    gradient_checkpointing=True,  # <--- Changed this line 08/04/25 & it worked
    gradient_checkpointing_kwargs={"use_reentrant": False},   # <--- Added this 9/30/25 to compensate for heftier training on V1 vs lite
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
    weight_decay=0.05, # Light regularization to maintain intelligence
)

# === Trainer
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8    # Efficient memory alignment (will brick your PC a lot less)
)

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,  # Proper padding for variable lengths
    callbacks=[ValidationCallback()]
)

#===Debug - Test loss with proper training mask
if len(train_ds) == 0:
    raise ValueError("No training samples available after preprocessing. Check dataset and filters.")

# Skip pre-training validation for now due to tensor shape issues
print(f"\n=== Skipping Pre-Training Validation ===")
print("Starting training directly...")

# Analyze training token distribution across dataset
total_training_tokens = 0
total_sequence_tokens = 0
for example in train_ds:
    labels = example["labels"]
    total_training_tokens += sum(1 for l in labels if l != -100)
    total_sequence_tokens += len(labels)

print(f"\nDataset Training Statistics:")
print(f"  Total training tokens: {total_training_tokens:,}")
print(f"  Total sequence tokens: {total_sequence_tokens:,}")
print(f"  Overall training ratio: {total_training_tokens/total_sequence_tokens*100:.1f}%")

# === Train ===
print(f"\n=== Starting Training ===")
model.train()
trainer.train()

# === Post-training validation ===
print(f"\n=== Post-Training Validation ===")
model.eval()

# Generate a sample response to validate the model works
print("\nSample Generation Test:")
test_prompt = "Hello! How can I help you today?"
test_messages = [{"role": "user", "content": test_prompt}]

prompt_text = tokenizer.apply_chat_template(
    test_messages, 
    tokenize=False, 
    add_generation_prompt=True
)

inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(f"  Prompt: {test_prompt}")
print(f"  Response: {response}")
print("\nTraining completed!")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to: {OUTPUT_DIR}")
