import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# === Paths ===
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"    #Dynamic pull from HugginFace - change if using a diffrent model
ADAPTER_PATH = "./lora_output"
MERGED_OUTPUT_PATH = "./finetune_output"

# === Safety check: confirm adapter_model.safetensors exists ===
adapter_model_file = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
if not os.path.exists(adapter_model_file):
    raise FileNotFoundError(f"adapter_model.safetensors not found in {ADAPTER_PATH}")

print("adapter_model.safetensors found. Proceeding...")

# === Load base model ===
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,  # Match training dtype
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"  # Match training config
)

# === Load clean LoRA adapter ===
print("Loading final LoRA adapter only...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    adapter_name="default",
    is_trainable=False,
    use_safetensors=True
)

# === Merge and unload ===
model = model.merge_and_unload()

# === Check for NaNs ===
print("Checking for NaNs...")
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        raise ValueError(f"NaNs detected in: {name}")
print("No NaNs found. Saving model...")

# === Save merged model ===
print("Saving merged model...")
model.save_pretrained(MERGED_OUTPUT_PATH, safe_serialization=True)

# === Load and save tokenizer ===
print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

# === Memory cleanup ===
del model
del base_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"Done! Clean merged model saved to: {MERGED_OUTPUT_PATH}")
print("Memory cleaned up.")
