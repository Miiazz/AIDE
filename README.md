# AIDE – Offline Assistant Framework
AIDE is a modular framework for building personalized, fully offline AI assistants. It integrates local LLMs, memory tiers, TTS/STT pipelines, and a sprite-driven GUI into a single customizable package.
This repo provides the framework and training tools required to create your own assistant. For quicker deployment or testing, you can use the generic Phi-3 model: microsoft/Phi-3-mini-4k-instruct

## Features (v1 Target)
-Local LLM support (Originally built around Phi-3)
-JSONL-based training workflow for LoRA finetunes
-Modular memory tiers (short, mid, long-term) [Stored Locally]
-Sprite-driven GUI (plug-and-play art packs)
-Offline STT/TTS pipelines
-Configurable persona [via training & sprite packs]
-Extendable via plugins and mode scripting
