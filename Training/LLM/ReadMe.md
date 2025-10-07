# LLM Training / Fine Tuning

This folder contains the methodologies and scripts used to train and create a usable model. 

## Requirements

### Python environment
- Python ≥ 3.10  
- `transformers` ≥ 4.43  
- `peft`  
- `datasets`  
- `bitsandbytes` (optional for GPU)  
- `torch` (with CUDA if available)

### CLI tools
- `llama.cpp` (for quantization and testing)
- `git`, `wget`, `cmake`, `make`

---

## llama.cpp Integration
After fine-tuning, models are exported and quantized for runtime efficiency.

```bash
# Convert the trained Hugging Face model to GGUF format
python3 convert-hf-to-gguf.py ./output/AIDE-LoRA --outfile aide-lora-f16.gguf

# Quantize (example: Q5_K_M)
./llama-quantize aide-lora-f16.gguf aide-lora-q5_k_m.gguf Q5_K_M
```
### Tips: 
- Q4_K_M = fastest, smallest [susceptible to quality drop]

- Q5_K_M = balanced (recommended)

- Q6_K = highest quality
