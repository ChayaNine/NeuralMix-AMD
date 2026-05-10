"""NeuralMix — AI Audio Engineer powered by Kimi-Audio + Fine-tuned model.

Two-stage pipeline:
Stage 1: Kimi-Audio-7B-Instruct LISTENS to the uploaded stem
Stage 2: Fine-tuned NeuralMix model PRESCRIBES exact FX parameters

Hardware: HF Spaces A10G small (24GB VRAM)
Both models load sequentially to fit within VRAM.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
import librosa
import soundfile as sf
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from peft import PeftModel

# ── Model IDs ─────────────────────────────────────────────────────────────────
KIMI_MODEL_ID  = "moonshotai/Kimi-Audio-7B-Instruct"
TUNED_MODEL_ID = "Chayanine/neuralmix-7b"
BASE_MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"

# ── System prompts ─────────────────────────────────────────────────────────────
KIMI_PROMPT = """You are analyzing an audio stem for a professional music producer.
Listen carefully and describe:
1. Frequency balance: is it muddy, harsh, thin, boomy, or balanced?
2. Dynamic range: is it compressed, punchy, dynamic, or flat?
3. Any problematic frequencies you detect (name them in Hz if possible)
4. Approximate level: quiet, moderate, or loud
5. What are the top 2-3 mixing problems that need fixing?
Be concise, specific, and technical."""

MIXING_SYSTEM = """You are NeuralMix, an expert AI audio engineer trained on professional mixing knowledge.
Given a stem analysis and a producer's request, output ONLY specific FX parameters.
Format your answer as:
EQ: [specific frequencies in Hz and amounts in dB]
Compressor: [ratio, attack ms, release ms, threshold dBFS]  
Reverb/Delay: [type, key values] (if needed)
Rationale: [one sentence explanation]
Never give vague advice. Always give specific numbers."""

EVAL_RESULTS_PATH = Path("data/eval_results.json")

# ── Lazy model cache ───────────────────────────────────────────────────────────
_cache: dict = {}

def load_kimi():
    if "kimi" not in _cache:
        print("Loading Kimi-Audio...")
        _cache["kimi_proc"] = AutoProcessor.from_pretrained(
            KIMI_MODEL_ID, trust_remote_code=True
        )
        _cache["kimi"] = AutoModelForCausalLM.from_pretrained(
            KIMI_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _cache["kimi"].eval()
        print("Kimi-Audio loaded")
    return _cache["kimi"], _cache["kimi_proc"]


def load_tuned():
    if "tuned" not in _cache:
        print("Loading fine-tuned model...")
        _cache["tuned_tok"] = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID, trust_remote_code=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        try:
            _cache["tuned"] = PeftModel.from_pretrained(base, TUNED_MODEL_ID)
        except Exception:
            # Fallback: use base model if fine-tuned not uploaded yet
            print("Fine-tuned model not found, using base model")
            _cache["tuned"] = base
        _cache["tuned"].eval()
        print("NeuralMix model loaded")
    return _cache["tuned"], _cache["tuned_tok"]


# ── Benchmark scores ───────────────────────────────────────────────────────────
def load_scores() -> str:
    if not EVAL_RESULTS_PATH.exists():
        return "_Benchmark: model trained on AMD MI300X ROCm. Scores loading..._"
    try:
        data = json.loads(EVAL_RESULTS_PATH.read_text())
        tuned = data.get("tuned", {})
        base  = data.get("base", {})
        acc_t = tuned.get("api_accuracy", 0)
        acc_b = base.get("api_accuracy", 0)
        rec_t = tuned.get("api_recall", 0)
        delta = round(acc_t - acc_b, 3)
        return (
            f"**NeuralMix fine-tuned on AMD MI300X** &nbsp;|&nbsp; "
            f"Accuracy: **{acc_t:.1%}** vs base {acc_b:.1%} "
            f"({'↑' if delta > 0 else '↓'}{abs(delta):.1%}) &nbsp;|&nbsp; "
            f"Recall: **{rec_t:.1%}**"
        )
    except Exception as e:
        return f"_Scores: {e}_"


# ── Stage 1: Kimi-Audio analysis ──────────────────────────────────────────────
def analyze_audio(audio_path: str) -> str:
    if not audio_path:
        return "No audio uploaded — describing a typical vocal stem."

    try:
        model, processor = load_kimi()
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text",  "text": KIMI_PROMPT},
            ]
        }]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios = []
        audio_data, sr = librosa.load(
            audio_path,
            sr=processor.feature_extractor.sampling_rate
        )
        audios.append(audio_data)
        inputs = processor(
            text=text, audios=audios, return_tensors="pt", padding=True
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        out_ids = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(
            out_ids, skip_special_tokens=True
        )[0].strip()

    except Exception as e:
        return f"Audio analysis unavailable ({e}). Using text-only mode."


# ── Stage 2: FX parameter generation ──────────────────────────────────────────
def generate_fx(audio_analysis: str, user_request: str):
    model, tokenizer = load_tuned()

    user_content = (
        f"Stem analysis:\n{audio_analysis}\n\n"
        f"Producer request: {user_request}"
    )
    messages = [
        {"role": "system", "content": MIXING_SYSTEM},
        {"role": "user",   "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    thread = Thread(
        target=model.generate,
        kwargs={
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "streamer":       streamer,
            "max_new_tokens": 400,
            "do_sample":      False,
            "pad_token_id":   tokenizer.eos_token_id,
        }
    )
    thread.daemon = True
    thread.start()
    for token in streamer:
        yield token


# ── Main respond function ──────────────────────────────────────────────────────
def respond(audio_file, request: str):
    if not request.strip():
        yield "Please describe what you want (e.g. 'make vocals cut through')", "", ""
        return

    # Stage 1 — Kimi-Audio listens
    yield "🎧 Kimi-Audio analyzing your stem...", "", ""
    audio_analysis = analyze_audio(audio_file)
    yield audio_analysis, "⚙️ NeuralMix generating FX parameters...", ""

    # Stage 2 — NeuralMix prescribes
    fx_output = ""
    for token in generate_fx(audio_analysis, request):
        fx_output += token
        yield audio_analysis, fx_output, ""

    # Final state
    yield (
        audio_analysis,
        fx_output,
        "✅ Done — apply these parameters to your track in your DAW"
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────
SCORES = load_scores()

EXAMPLES = [
    [None, "Make the vocals cut through the mix"],
    [None, "The mix sounds muddy in the low-mids, fix it"],
    [None, "Add warmth and presence to this guitar"],
    [None, "The drums sound flat, make them punch"],
    [None, "This bass is clashing with the kick, separate them"],
]

with gr.Blocks(
    title="NeuralMix — AI Audio Engineer",
    theme=gr.themes.Base(),
    css="""
    .output-box textarea { font-family: monospace; font-size: 13px; }
    .header { text-align: center; padding: 20px 0; }
    """
) as demo:

    gr.Markdown("""
# 🎚️ NeuralMix — AI Audio Engineer
**Fine-tuned on AMD Instinct MI300X · Powered by Kimi-Audio**

Upload a stem → describe what you want → get professional FX parameters
""", elem_classes=["header"])

    gr.Markdown(SCORES)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload stem (.wav / .mp3)",
                type="filepath",
            )
            request_input = gr.Textbox(
                label="What do you want? (describe in plain language)",
                placeholder="e.g. 'Make vocals cut through' or 'Fix muddy low-mids'",
                lines=3,
            )
            submit_btn = gr.Button("🎛️ Generate FX Parameters", variant="primary", size="lg")

        with gr.Column(scale=2):
            analysis_out = gr.Textbox(
                label="🎧 Stage 1 — Kimi-Audio stem analysis",
                lines=8,
                elem_classes=["output-box"],
                show_copy_button=True,
            )
            fx_out = gr.Textbox(
                label="⚙️ Stage 2 — NeuralMix FX parameters",
                lines=10,
                elem_classes=["output-box"],
                show_copy_button=True,
            )
            status_out = gr.Textbox(label="Status", lines=1)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[audio_input, request_input],
        label="Try these examples (no audio needed for text-only mode)",
    )

    gr.Markdown("""---
**How it works:**
1. **Kimi-Audio** (Moonshot AI) listens to your stem and analyzes frequency balance, dynamics, and mixing problems
2. **NeuralMix** (fine-tuned Qwen2.5-7B on AMD MI300X) reads the analysis and outputs precise FX parameters
3. Apply the parameters in Ableton, Logic, FL Studio, or any DAW

**Open source** · Apache 2.0 · [GitHub](https://github.com/ChayaNine/NeuralMix-AMD)
""")

    submit_btn.click(
        fn=respond,
        inputs=[audio_input, request_input],
        outputs=[analysis_out, fx_out, status_out],
    )

if __name__ == "__main__":
    demo.launch()