---
title: NeuralMix
emoji: 🎚
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
hardware: a10g-small
license: apache-2.0
---

# NeuralMix — AI Audio Engineer

Fine-tuned on AMD Instinct MI300X. Domain-specialized for professional audio mixing.

**Side-by-side comparison:** NeuralMix (fine-tuned) vs MiniMax M2.7 (general-purpose backbone).

## What it does

Describe a mixing problem in plain English. NeuralMix returns a specific FX chain:

- **EQ:** exact frequencies (Hz/kHz), dB amounts, Q values
- **Compression:** ratio, threshold (dBFS), attack (ms), release (ms), makeup gain
- **Reverb/Delay:** type, pre-delay, decay time, wet/dry %
- **Gain staging:** target levels, headroom analysis
- **Full stem chains:** complete per-track processing for vocals, drums, bass, guitar

## Comparison panel

Left: NeuralMix (fine-tuned on audio engineering domain knowledge)  
Right: MiniMax M2.7 (general-purpose LLM — the current NeuralMix backbone)

The fine-tuned model produces specific parameter values. The general model produces vague advice.

## Thai language

Select **TH** — explanations in Thai, all parameter values and plugin names in English.

## Model weights

- Fine-tuned adapter: [nawman0209/neuralmix-7b-lora](https://huggingface.co/nawman0209/neuralmix-7b-lora)
- Merged model: [nawman0209/neuralmix-7b](https://huggingface.co/nawman0209/neuralmix-7b)
- Training dataset: [nawman0209/neuralmix-dataset](https://huggingface.co/datasets/nawman0209/neuralmix-dataset)

## Production integration

This model runs at [neuralmix.vaclis.net](https://neuralmix.vaclis.net) replacing the MiniMax M2.7 backbone.
Deploy as API endpoint and update the NeuralMix API call to use this model.

Built for AMD Developer Hackathon — Track 2 (Fine-tuning on AMD GPUs). Apache 2.0.
