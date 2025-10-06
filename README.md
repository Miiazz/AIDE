# AIDE – Offline Assistant Framework
AIDE is a modular framework for building personalized, fully offline AI assistants.
It’s designed for both embedded and desktop environments (such as the Raspberry Pi 5) and emphasizes modularity, local processing, and personality-driven interaction.

## Status:
The LoRA fine-tuning framework for customizing Phi-3 models has been tested with sucess when used with strong system prompting and elevated model temperatues (starting at: `--temp 0.7`)
Once stable and repeatable results are achieved, backend deployment code for use on the Raspberry Pi 5 will be added to this repository.

## Features (v1 Target)
- Local LLM Support — Originally built around Phi-3

- JSONL-Based Training — LoRA fine-tuning for custom personalities

- Modular Memory Tiers — Short, mid, and long-term context layers stored locally 

- Customizable GUI — Ability to add / modify sprites

- Offline STT/TTS Pipelines — Full voice input and synthesis without internet dependency

- Configurable Persona System — Personality customization via finetuning

- Extendable Architecture — Plugin and mode scripting support for advanced behaviors

## Planned B.O.M
- Raspberry Pi 5 16GB
  
- Raspberry Pi SSD Hat w/ 512GB NVMe SSD
  
- Waveshare 3.5" Touch Screen
  
- Adafruit Mini USB Microphone (PID:3367)
  
- Adafruit I2S 3W Class D Amplifier Breakout - MAX98357A
  
- Uxcell 1.5W 8Ω Mini Speaker
