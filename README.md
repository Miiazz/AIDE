# AIDE – Offline Assistant Framework - W.I.P
![Python](https://img.shields.io/badge/Made%20with-Python-blue)
![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-red)
![Offline AI](https://img.shields.io/badge/Offline_LLM-green)

# AIDE

**AIDE** is a modular, offline-first framework for building personalized AI assistants on edge and desktop hardware.

Rather than centering on a single fixed assistant implementation, AIDE is designed as a flexible foundation for experimenting with and deploying local AI systems that prioritize privacy, personality, and modularity. It is intended for hardware such as the Raspberry Pi 5 as well as desktop Linux environments, and aims to support custom assistant behavior without relying on cloud services.

## Project Direction

AIDE is being developed as a **framework for personal AI on edge hardware** rather than as a single locked-down assistant.

The project explores what is required to build a private, personality-driven assistant using:

- local language models
- offline voice pipelines
- modular memory systems
- customizable interface layers
- hardware-aware deployment strategies

## Current Status

This repository currently focuses on personality fine-tuning workflows and the supporting foundation for a future modular runtime. Full assistant features such as memory tiers, GUI, STT, and TTS are planned but not yet complete.

Early testing suggests that some model families respond better to personality adaptation than others, so AIDE is moving toward a more **model-agnostic architecture** rather than remaining tied to the Phi family alone. As futher testing is done a model recomendation(s) will be provided 

Once model behavior and deployment are stable and repeatable, Raspberry Pi 5 backend integration and runtime support will be added to this repository.

## v1 Targets

### Local LLM Support
Support for running local language models on edge and desktop hardware, with the long-term goal of interchangeable backend model support.

### JSONL-Based Fine-Tuning
Training workflows for adapting assistant tone, behavior, and style using lightweight fine-tuning methods such as LoRA.

### Modular Memory Tiers
Locally stored short-, mid-, and long-term context layers to support personalization and continuity without cloud dependence.

### Customizable Interface Layer
Support for adjustable GUI elements, including sprites, text displays, and other assistant presentation systems.

### Offline STT/TTS Pipelines
Voice input and speech synthesis handled locally for fully offline interaction.

### Persona Framework
Configurable assistant personality through prompting, fine-tuning, and modular behavior logic.

### Extendable Architecture
A structure intended to support plugins, operating modes, and future feature modules without requiring a full rewrite.

## Core Design Goals

- Offline-first operation
- Hardware-conscious deployment
- Modular assistant architecture
- Personalization without cloud reliance
- Experimentation across models and interfaces

## Long-Term Vision

AIDE is ultimately aimed at answering a broader question:

> What does it take to run a useful and personal AI assistant on edge hardware?

This repository is meant to document and support that process through reusable components, testing, and iterative development.
