"""NeuralMix demo — HF Spaces (hardware: a10g-small).

Side-by-side: NeuralMix fine-tuned model vs MiniMax M2.7 (or base Qwen fallback).
Both responses generated sequentially on the GPU — UX shows NeuralMix first (~12s),
then comparison model (~12s). Total ~24s per query.

Environment variables:
    MINIMAX_API_KEY   — MiniMax API key for comparison panel (optional)
    MINIMAX_MODEL     — MiniMax model name (default: MiniMax-Text-01)
    TUNED_MODEL_ID    — override the fine-tuned model HF repo
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterator

import gradio as gr
import requests
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TUNED_MODEL_ID = os.environ.get("TUNED_MODEL_ID", "nawman0209/neuralmix-7b")
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-Text-01")

SYSTEM_EN = (
    "You are NeuralMix, an expert AI audio engineer. "
    "Given a stem description or mixing problem, provide specific FX chain recommendations "
    "with exact parameter values: EQ frequencies in Hz/kHz, compressor ratios, attack/release in ms, "
    "reverb decay times, gain targets in dBFS. Always explain the reasoning behind each setting."
)

SYSTEM_TH = (
    "You are NeuralMix, an expert AI audio engineer. "
    "Answer in Thai. Use Thai for explanations. "
    "Keep all parameter values, technical terms, plugin names, and API names in English."
)

# Base generation config — pad_token_id added per-call from each model's own tokenizer.
GENERATION_CONFIG: dict = {
    "max_new_tokens": 600,
    "temperature": 0.2,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

EVAL_RESULTS_PATH = Path("data/eval_results.json")

EXAMPLES = [
    ["The lead vocal sounds harsh and sibilant at 3–5kHz. The track has drums and electric guitar competing.",
     "Vocal track: -16dBFS RMS, crest factor ~14dB", "EN"],
    ["How do I compress a kick drum to add punch without killing the transient?", "", "EN"],
    ["My mix sounds muddy in the low-mids. Which frequencies do I cut and on which tracks?",
     "Stems: vocals, kick, snare, bass guitar, electric guitar, synth pad", "EN"],
    ["What reverb settings should I use for vocals in a pop track at 120 BPM?", "", "EN"],
    ["เสียง vocal ฟังดูขุ่นมัวและไม่โดดเด่น จะปรับ EQ และ compressor ยังไงดี?", "", "TH"],
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_models: dict = {}
_load_lock = threading.Lock()


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # fp16 on NVIDIA A10G
    )


def _is_peft_repo(model_id: str) -> bool:
    try:
        from peft import PeftConfig
        PeftConfig.from_pretrained(model_id)
        return True
    except Exception:
        return False


def _load_one(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = _bnb_config()

    if _is_peft_repo(model_id):
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def get_models():
    """Lazy-load tuned model and base model (fallback). Thread-safe."""
    if _models:
        return _models.get("tuned"), _models.get("tuned_tok"), _models.get("base"), _models.get("base_tok")

    with _load_lock:
        if _models:
            return _models.get("tuned"), _models.get("tuned_tok"), _models.get("base"), _models.get("base_tok")

        print(f"Loading fine-tuned model: {TUNED_MODEL_ID}")
        try:
            model, tok = _load_one(TUNED_MODEL_ID)
            _models["tuned"] = model
            _models["tuned_tok"] = tok
        except Exception as e:
            print(f"WARNING: Could not load fine-tuned model: {e}")
            print("Falling back to base model for both panels.")
            _models["tuned"] = None
            _models["tuned_tok"] = None

        # Base model loaded for comparison (or as main model if tuned unavailable).
        print(f"Loading base model: {BASE_MODEL_ID}")
        base_model, base_tok = _load_one(BASE_MODEL_ID)
        _models["base"] = base_model
        _models["base_tok"] = base_tok

        if _models["tuned"] is None:
            _models["tuned"] = base_model
            _models["tuned_tok"] = base_tok

    return _models["tuned"], _models["tuned_tok"], _models["base"], _models["base_tok"]


# ---------------------------------------------------------------------------
# MiniMax API
# ---------------------------------------------------------------------------

def call_minimax(instruction: str, context: str, lang: str) -> str:
    """Call MiniMax API. Returns empty string on failure."""
    if not MINIMAX_API_KEY:
        return ""

    system = SYSTEM_TH if lang == "TH" else SYSTEM_EN
    user_content = instruction.strip()
    if context.strip():
        user_content = f"{user_content}\n\nContext:\n{context.strip()}"

    try:
        resp = requests.post(
            "https://api.minimax.io/v1/text/chatcompletion_v2",
            headers={
                "Authorization": f"Bearer {MINIMAX_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MINIMAX_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 600,
                "temperature": 0.2,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"_MiniMax API error: {e}_"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _make_prompt(instruction: str, context: str, lang: str, tokenizer) -> str:
    system = SYSTEM_TH if lang == "TH" else SYSTEM_EN
    user_content = instruction.strip()
    if context.strip():
        user_content = f"{user_content}\n\nContext:\n{context.strip()}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_streaming(model, tokenizer, prompt: str) -> Iterator[str]:
    # Explicit field extraction — never **encoded (avoids token_type_ids TypeError).
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "streamer": streamer,
            "pad_token_id": tokenizer.eos_token_id,
            **GENERATION_CONFIG,
        },
    )
    thread.daemon = True  # Prevents zombie threads on Gradio cancel
    thread.start()

    try:
        for token in streamer:
            yield token
    finally:
        pass


def respond(
    instruction: str,
    context: str,
    lang: str,
) -> Iterator[tuple[str, str]]:
    """
    Yields (neuralmix_text, comparison_text).
    Phase 1: NeuralMix streams, comparison shows status.
    Phase 2: NeuralMix done, comparison streams (or MiniMax result appears).
    """
    if not instruction.strip():
        yield ("_Please enter a mixing question or stem description._", "")
        return

    tuned_model, tuned_tok, base_model, base_tok = get_models()

    # --- Phase 1: stream NeuralMix fine-tuned ---
    tuned_prompt = _make_prompt(instruction, context, lang, tuned_tok)
    tuned_text = ""
    comparison_label = "MiniMax M2.7" if MINIMAX_API_KEY else "Base Qwen2.5-7B"

    for token in generate_streaming(tuned_model, tuned_tok, tuned_prompt):
        tuned_text += token
        yield (tuned_text, f"_Generating {comparison_label} response..._")

    # --- Phase 2: comparison model ---
    if MINIMAX_API_KEY:
        # MiniMax API call (not streaming — returns full response)
        yield (tuned_text, f"_Calling {comparison_label} API..._")
        comparison_text = call_minimax(instruction, context, lang)
        yield (tuned_text, comparison_text)
    else:
        # Base model streaming
        base_prompt = _make_prompt(instruction, context, lang, base_tok)
        comparison_text = ""
        for token in generate_streaming(base_model, base_tok, base_prompt):
            comparison_text += token
            yield (tuned_text, comparison_text)


# ---------------------------------------------------------------------------
# Benchmark scores
# ---------------------------------------------------------------------------

def _load_scores() -> str:
    if not EVAL_RESULTS_PATH.exists():
        return "_Benchmark scores: pending training run._"
    try:
        data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        tuned = data.get("tuned", {})
        base = data.get("base", {})
        n = data.get("n_eval", "?")

        def fmt(d: dict) -> str:
            rec = d.get("param_recall", "?")
            spec = d.get("specificity", "?")
            rec_s = f"**{rec:.1%}**" if isinstance(rec, float) else rec
            spec_s = f"**{spec:.1%}**" if isinstance(spec, float) else spec
            return f"param_recall {rec_s} · specificity {spec_s}"

        comp_label = "MiniMax M2.7" if MINIMAX_API_KEY else "Base Qwen2.5-7B"
        lines = [
            f"**Benchmark** ({n}-question eval set)",
            f"🟣 NeuralMix (fine-tuned) &nbsp; {fmt(tuned)}",
            f"⬜ {comp_label} &nbsp; {fmt(base)}",
        ]
        return "  \n".join(lines)
    except Exception as e:
        return f"_Benchmark scores: error loading ({e})_"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

SCORES_MD = _load_scores()
COMP_LABEL = "MiniMax M2.7" if MINIMAX_API_KEY else "Base Qwen2.5-7B (no MINIMAX_API_KEY set)"

with gr.Blocks(
    title="NeuralMix — AI Audio Engineer",
    theme=gr.themes.Soft(),
    css=".output-panel textarea { font-family: monospace; font-size: 13px; }",
) as demo:

    gr.Markdown(
        """# 🎚 NeuralMix — AI Audio Engineer
Fine-tuned on AMD Instinct MI300X &nbsp;·&nbsp; Apache 2.0 &nbsp;·&nbsp; [neuralmix.vaclis.net](https://neuralmix.vaclis.net)

Describe a mixing problem in plain English. NeuralMix returns a specific FX chain with exact parameter values.
"""
    )

    gr.Markdown(SCORES_MD)

    with gr.Row():
        instruction = gr.Textbox(
            label="Mixing question or stem description",
            placeholder='e.g. "The lead vocal sounds harsh at 3–5kHz, electric guitar is competing"',
            lines=3,
            scale=5,
        )
        lang = gr.Radio(choices=["EN", "TH"], value="EN", label="Language", scale=1)

    context = gr.Textbox(
        label="Additional context — measurements, stem list, BPM (optional)",
        placeholder='e.g. "Vocal RMS: -16dBFS, stems: vocals/drums/bass/guitar, BPM 120"',
        lines=3,
    )

    submit_btn = gr.Button("Analyze & Generate FX Chain", variant="primary", size="lg")

    with gr.Row(equal_height=True):
        tuned_out = gr.Textbox(
            label="🟣 NeuralMix (fine-tuned — specific parameters)",
            lines=20,
            show_copy_button=True,
            elem_classes=["output-panel"],
        )
        base_out = gr.Textbox(
            label=f"⬜ {COMP_LABEL} (general-purpose)",
            lines=20,
            show_copy_button=True,
            elem_classes=["output-panel"],
        )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[instruction, context, lang],
        label="Example scenarios (click to load)",
    )

    gr.Markdown(
        """---
**How it works:** NeuralMix is trained on audio engineering textbooks, plugin documentation,
and forum Q&A — focused on specific parameter values (EQ frequencies, compressor ratios,
attack/release times). Trained on AMD Instinct MI300X using ROCm PyTorch LoRA bf16.
The right panel shows what a general-purpose model produces for comparison.

**Thai mode:** Explanations in Thai, parameter values and plugin names in English."""
    )

    submit_btn.click(fn=respond, inputs=[instruction, context, lang], outputs=[tuned_out, base_out])
    instruction.submit(fn=respond, inputs=[instruction, context, lang], outputs=[tuned_out, base_out])

if __name__ == "__main__":
    demo.launch()
