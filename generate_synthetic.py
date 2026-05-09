"""Generate synthetic audio engineering training pairs from domain formulas.

No network required. Produces ~400 high-quality pairs immediately.
Covers compressor math, EQ frequency selection, gain staging, reverb timing,
and common stem-specific FX chains.

Usage:
    python generate_synthetic.py [--output data/raw/synthetic_pairs.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dataset import TrainingPair

# ---------------------------------------------------------------------------
# Domain knowledge tables
# ---------------------------------------------------------------------------

STEM_CHAINS: list[dict] = [
    {
        "stem": "lead vocal",
        "problem": "harsh and sibilant, cuts through too aggressively in the 3–5kHz range",
        "rms": "-16dBFS",
        "chain": (
            "**EQ:** High-pass at 80Hz (Q=0.7) to remove sub-rumble. "
            "Narrow notch at 3.2kHz, -4dB, Q=2.8 to reduce presence harshness. "
            "High shelf cut at 10kHz, -1.5dB to tame air without killing sparkle.\n\n"
            "**De-esser:** Frequency 7.2kHz, threshold -24dBFS, ratio 4:1. "
            "Sibilance reduction without lisping.\n\n"
            "**Compressor:** Ratio 3:1, threshold -18dBFS, attack 8ms, release 80ms, "
            "makeup gain +2dB. Transparent dynamic control; 8ms attack preserves consonant transients.\n\n"
            "**Reasoning:** The 3–5kHz harsh peak is a classic vocal recording issue. "
            "A narrow surgical cut at 3.2kHz removes the offending resonance without "
            "affecting the adjacent 2kHz (vocal intelligibility) or 5kHz (presence) bands. "
            "The 3:1 ratio with -18dBFS threshold controls dynamics without squashing — "
            "correct for a vocal sitting at -16dBFS RMS."
        ),
        "category": "stem_chain",
    },
    {
        "stem": "kick drum",
        "problem": "lacks punch and low-end definition, sounds flabby",
        "rms": "-12dBFS",
        "chain": (
            "**EQ:** High-pass at 30Hz (Q=0.7) to clean up sub-rumble. "
            "Boost at 60Hz, +4dB, Q=1.2 for fundamental punch. "
            "Cut at 300–400Hz, -3dB, Q=1.0 to remove boxiness. "
            "Boost at 3–5kHz, +2dB, Q=1.5 for click/attack definition.\n\n"
            "**Compressor:** Ratio 4:1, threshold -20dBFS, attack 30ms, release 100ms, "
            "makeup gain +3dB. Slow attack (30ms) lets the transient through — critical "
            "for kick punch. Fast attack would kill the snap.\n\n"
            "**Transient Shaper:** Attack +3, Sustain -2. Enhances punch without compression artifacts.\n\n"
            "**Reasoning:** Kick 'flabbiness' is typically a 300–400Hz buildup. "
            "The 60Hz boost adds weight; 300Hz cut removes mud. Attack of 30ms on "
            "the compressor is deliberately slow — compressing the attack on a kick "
            "removes the 'thwack' that makes it cut through."
        ),
        "category": "stem_chain",
    },
    {
        "stem": "electric guitar",
        "problem": "clashes with vocals in the midrange, sounds boxy",
        "rms": "-14dBFS",
        "chain": (
            "**EQ:** High-pass at 100Hz (Q=0.7) — guitar has no useful content below 80Hz. "
            "Cut at 250–350Hz, -4dB, Q=1.2 to remove boxy buildup. "
            "Cut at 2–3kHz, -3dB, Q=1.5 to carve space for vocal presence. "
            "Boost at 5kHz, +2dB, Q=1.2 for pick attack definition.\n\n"
            "**Compressor:** Ratio 2:1, threshold -16dBFS, attack 20ms, release 200ms, "
            "makeup gain +1dB. Light compression to glue without killing dynamics.\n\n"
            "**Reasoning:** The 2–3kHz cut is the key move — it's the range where guitar "
            "and vocals compete most directly. Cutting the guitar here creates a 'hole' "
            "for the vocal to sit in without losing guitar presence (which lives above 4kHz). "
            "The 250Hz cut removes boxiness common in close-miked guitar cabs."
        ),
        "category": "stem_chain",
    },
    {
        "stem": "bass guitar",
        "problem": "muddy low end, not sitting well under the kick",
        "rms": "-15dBFS",
        "chain": (
            "**EQ:** High-pass at 40Hz (Q=0.7) — removes sub-60Hz rumble that only wastes headroom. "
            "Boost at 80Hz, +2dB, Q=1.5 for fundamental. "
            "Cut at 200–250Hz, -4dB, Q=1.0 to reduce mud competing with kick. "
            "Boost at 800Hz–1kHz, +2dB, Q=1.5 for note definition on smaller speakers.\n\n"
            "**Compressor:** Ratio 4:1, threshold -20dBFS, attack 10ms, release 200ms, "
            "makeup gain +2dB. Tight control — bass needs more compression than guitar.\n\n"
            "**Sidechain:** Sidechain compress bass from kick signal. Ratio 3:1, "
            "threshold -30dBFS, attack 1ms, release 100ms. Bass ducks 2–3dB on kick hits — "
            "creates clean separation without EQ overlap.\n\n"
            "**Reasoning:** Bass and kick share the 60–120Hz range. The 200–250Hz cut on bass "
            "creates room for kick's 'thump' range. Sidechain compression is the professional "
            "solution — it doesn't reduce either instrument's static energy, only creates "
            "dynamic space on transients."
        ),
        "category": "stem_chain",
    },
    {
        "stem": "snare drum",
        "problem": "thin and lacking body, disappears in the mix",
        "rms": "-14dBFS",
        "chain": (
            "**EQ:** High-pass at 90Hz (Q=0.7). "
            "Boost at 200Hz, +3dB, Q=1.5 for body/crack. "
            "Cut at 400–600Hz, -3dB, Q=1.0 to remove 'cardboard' boxiness. "
            "Boost at 5kHz, +3dB, Q=1.5 for crack and snap. "
            "High shelf boost at 10kHz, +2dB for 'air' and stick definition.\n\n"
            "**Compressor:** Ratio 4:1, threshold -18dBFS, attack 5ms, release 50ms, "
            "makeup gain +3dB. Faster attack than kick — snare transient is shorter.\n\n"
            "**Parallel compression:** Send to parallel bus, compress heavily (8:1, -24dBFS threshold), "
            "blend back at 30% wet. Parallel compression adds density without killing transient.\n\n"
            "**Reasoning:** Thin snare is usually caused by insufficient 200Hz energy. "
            "The 400–600Hz cut is the 'cardboard' frequency — cutting here while boosting "
            "200Hz gives body without boxy resonance. Parallel compression is the snare "
            "engineer's main tool — serial compression at high ratios kills the crack."
        ),
        "category": "stem_chain",
    },
]

COMPRESSOR_SCENARIOS: list[dict] = [
    {
        "instrument": "lead vocal",
        "rms": -16, "crest_factor": 14, "style": "transparent",
        "ratio": "3:1", "threshold": -18, "attack": 8, "release": 80,
        "reason": "Low ratio preserves natural dynamics; 8ms attack lets consonants through"
    },
    {
        "instrument": "lead vocal",
        "rms": -14, "crest_factor": 10, "style": "controlled pop",
        "ratio": "4:1", "threshold": -16, "attack": 5, "release": 60,
        "reason": "Higher ratio for more dynamic control on energetic pop performance"
    },
    {
        "instrument": "acoustic guitar",
        "rms": -16, "crest_factor": 18, "style": "strummed",
        "ratio": "2:1", "threshold": -20, "attack": 30, "release": 300,
        "reason": "Very slow attack (30ms) preserves pick attack transient; low ratio keeps dynamics"
    },
    {
        "instrument": "drum bus",
        "rms": -10, "crest_factor": 20, "style": "glue",
        "ratio": "2:1", "threshold": -12, "attack": 30, "release": 100,
        "reason": "Bus compression glues the kit together; slow attack preserves transients at bus level"
    },
    {
        "instrument": "bass guitar",
        "rms": -15, "crest_factor": 8, "style": "tight",
        "ratio": "4:1", "threshold": -18, "attack": 10, "release": 200,
        "reason": "Bass needs more compression than melodic instruments to maintain consistent level"
    },
    {
        "instrument": "full mix bus",
        "rms": -8, "crest_factor": 14, "style": "mastering",
        "ratio": "1.5:1", "threshold": -6, "attack": 30, "release": 250,
        "reason": "Very light mix bus compression (1–2dB GR max); never compress more than 3dB on the master bus"
    },
]

EQ_SCENARIOS: list[dict] = [
    {
        "instrument": "lead vocal", "freq": "250Hz", "action": "cut", "amount": "-3dB", "q": 1.2,
        "reason": "Removes 'woolly' low-mid buildup common in close-miked condenser recordings"
    },
    {
        "instrument": "lead vocal", "freq": "3–5kHz", "action": "cut", "amount": "-2 to -4dB", "q": 2.0,
        "reason": "Reduces harshness in presence range; amount depends on mic (ribbons rarely need this)"
    },
    {
        "instrument": "kick drum", "freq": "60Hz", "action": "boost", "amount": "+3 to +5dB", "q": 1.2,
        "reason": "Fundamental punch; exact frequency depends on kick tuning (50–80Hz range)"
    },
    {
        "instrument": "kick drum", "freq": "300–400Hz", "action": "cut", "amount": "-4 to -6dB", "q": 1.0,
        "reason": "The 'boxiness' range — cutting here opens up the low end"
    },
    {
        "instrument": "electric guitar", "freq": "2–3kHz", "action": "cut", "amount": "-3dB", "q": 1.5,
        "reason": "Creates space for vocal presence without losing guitar definition"
    },
    {
        "instrument": "bass guitar", "freq": "200–250Hz", "action": "cut", "amount": "-4dB", "q": 1.0,
        "reason": "Separates bass from kick drum's fundamental range; reduces mud"
    },
    {
        "instrument": "acoustic guitar", "freq": "100Hz", "action": "high-pass", "amount": "24dB/oct", "q": 0.7,
        "reason": "Acoustic guitar has no useful content below 80Hz; HPF keeps low end clean"
    },
    {
        "instrument": "overhead cymbals", "freq": "200–500Hz", "action": "cut", "amount": "-3 to -5dB", "q": 0.9,
        "reason": "Overheads pick up drum body resonances that add mud without definition"
    },
]

REVERB_SCENARIOS: list[dict] = [
    {
        "context": "lead vocal in a modern pop track, BPM 120",
        "type": "plate", "pre_delay": "20ms", "decay": "1.2s", "wet": "15%",
        "reason": "Short pre-delay (20ms) keeps vocal upfront while adding space. Plate is classic vocal reverb. Low wet% on lead vocal — excessive reverb pushes it back."
    },
    {
        "context": "snare drum in a rock track, BPM 100",
        "type": "room", "pre_delay": "10ms", "decay": "0.8s", "wet": "25%",
        "reason": "Short room reverb glues snare to the rest of the kit. Pre-delay = 60000/BPM/4 = 150ms max, but 10ms keeps it tight."
    },
    {
        "context": "piano in a ballad, BPM 70",
        "type": "hall", "pre_delay": "30ms", "decay": "2.5s", "wet": "30%",
        "reason": "Ballad piano benefits from long hall reverb. Pre-delay = 60000/70/8 ≈ 107ms max; 30ms maintains clarity at slow tempo."
    },
    {
        "context": "electronic lead synth, BPM 128, dance music",
        "type": "spring or gated", "pre_delay": "0ms", "decay": "0.4s", "wet": "20%",
        "reason": "Dance music synths need short, tight reverb — long tails wash out rhythmic energy."
    },
]

GAIN_STAGING_RULES: list[dict] = [
    {
        "question": "What is the correct gain staging target for individual tracks?",
        "answer": (
            "Individual tracks should be gain staged to average -18dBFS RMS (±2dB), leaving "
            "headroom for peaks. This corresponds to 0VU on analog hardware.\n\n"
            "**Why -18dBFS:** Digital audio clips at 0dBFS. A vocal at -18dBFS RMS with "
            "a crest factor of 14dB will peak at approximately -4dBFS — safe headroom. "
            "Drums with higher crest factors (18–20dB) at -18dBFS RMS peak at 0dBFS — "
            "they should be staged at -20 to -22dBFS RMS instead.\n\n"
            "**Check:** Set each channel fader to unity (0dB). If the mix bus is clipping, "
            "the problem is gain staging, not fader levels. Reduce all track output gains "
            "by 6–10dB collectively using a gain plugin, not the fader."
        ),
    },
    {
        "question": "How loud should my mix bus be before mastering?",
        "answer": (
            "Target -6dBFS peak with average RMS around -14 to -16dBFS LUFS (integrated). "
            "This gives a mastering engineer 6dB of headroom to apply limiting.\n\n"
            "**Never limit the mix bus.** A limiter on the mix bus crushes transients and "
            "makes mastering impossible. If the mix bus is hitting 0dBFS, reduce track "
            "levels — do not add a limiter.\n\n"
            "**True peak:** Check true peak (inter-sample peak) with a true peak meter. "
            "True peaks can exceed 0dBFS by up to +3dBTP even when the digital waveform "
            "shows headroom. Target -1dBTP for export."
        ),
    },
]


# ---------------------------------------------------------------------------
# Pair builders
# ---------------------------------------------------------------------------

def _stem_chain_pairs() -> list[TrainingPair]:
    pairs = []
    for sc in STEM_CHAINS:
        pairs.append(TrainingPair(
            instruction=f"What FX chain should I apply to {sc['stem']} that {sc['problem']}? "
                        f"The track is sitting at {sc['rms']} RMS.",
            input="",
            output=sc["chain"],
            source="synthetic",
            category="stem_chain",
            language="en",
            verified=True,
        ))
        # Variation: ask for just the EQ
        pairs.append(TrainingPair(
            instruction=f"How should I EQ a {sc['stem']} that {sc['problem']}?",
            input="",
            output=sc["chain"].split("**Compressor")[0].strip(),
            source="synthetic",
            category="eq",
            language="en",
            verified=True,
        ))
    return pairs


def _compressor_pairs() -> list[TrainingPair]:
    pairs = []
    for sc in COMPRESSOR_SCENARIOS:
        output = (
            f"For {sc['instrument']} at {sc['rms']}dBFS RMS (crest factor {sc['crest_factor']}dB), "
            f"use:\n\n"
            f"- **Ratio:** {sc['ratio']}\n"
            f"- **Threshold:** {sc['threshold']}dBFS\n"
            f"- **Attack:** {sc['attack']}ms\n"
            f"- **Release:** {sc['release']}ms\n"
            f"- **Makeup gain:** +{abs(sc['threshold']) - abs(sc['rms']) - 2}dB (approximate)\n\n"
            f"**Reasoning:** {sc['reason']}"
        )
        pairs.append(TrainingPair(
            instruction=f"What compressor settings should I use for {sc['instrument']} "
                        f"at {sc['rms']}dBFS RMS? I want a {sc['style']} result.",
            input="",
            output=output,
            source="synthetic",
            category="compression",
            language="en",
            verified=True,
        ))
    return pairs


def _eq_pairs() -> list[TrainingPair]:
    pairs = []
    for sc in EQ_SCENARIOS:
        output = (
            f"On {sc['instrument']}, {sc['action']} {sc['freq']} by {sc['amount']} "
            f"(Q={sc['q']}).\n\n"
            f"**Why:** {sc['reason']}"
        )
        pairs.append(TrainingPair(
            instruction=f"When should I {sc['action']} {sc['freq']} on {sc['instrument']}?",
            input="",
            output=output,
            source="synthetic",
            category="eq",
            language="en",
            verified=True,
        ))
    return pairs


def _reverb_pairs() -> list[TrainingPair]:
    pairs = []
    for sc in REVERB_SCENARIOS:
        output = (
            f"**Reverb type:** {sc['type'].title()}\n"
            f"**Pre-delay:** {sc['pre_delay']}\n"
            f"**Decay time:** {sc['decay']}\n"
            f"**Wet/dry:** {sc['wet']} wet\n\n"
            f"**Reasoning:** {sc['reason']}"
        )
        pairs.append(TrainingPair(
            instruction=f"What reverb settings should I use for {sc['context']}?",
            input="",
            output=output,
            source="synthetic",
            category="reverb_delay",
            language="en",
            verified=True,
        ))
    return pairs


def _gain_staging_pairs() -> list[TrainingPair]:
    pairs = []
    for gs in GAIN_STAGING_RULES:
        pairs.append(TrainingPair(
            instruction=gs["question"],
            input="",
            output=gs["answer"],
            source="synthetic",
            category="concepts",
            language="en",
            verified=True,
        ))
    return pairs


def _parallel_compression_pairs() -> list[TrainingPair]:
    return [
        TrainingPair(
            instruction="How do I apply parallel compression to drums?",
            input="",
            output=(
                "Send the drum bus to a parallel (aux) track. On the parallel track: "
                "ratio 8:1, threshold -30dBFS, attack 1ms, release 50ms. "
                "This crushes the parallel signal completely. "
                "Blend back at 20–40% wet alongside the uncompressed dry signal.\n\n"
                "**Result:** The dry signal preserves transient snap; the wet signal adds "
                "density and sustain. Start with 25% wet and increase until drums feel "
                "'thick' without losing punch. Typical final blend: 20–35% parallel signal.\n\n"
                "**Why parallel vs serial:** Serial compression at 8:1 kills the transient. "
                "Parallel compression adds density from below — the transient rides on top "
                "of the compressed sustain."
            ),
            source="synthetic",
            category="compression",
            language="en",
            verified=True,
        ),
        TrainingPair(
            instruction="What is the New York compression technique and when should I use it?",
            input="",
            output=(
                "New York compression (parallel compression) blends a heavily compressed "
                "signal with the uncompressed dry signal. Setup:\n\n"
                "1. Send source to an aux bus\n"
                "2. On the aux: ratio 8:1 to ∞:1, threshold -30dBFS, attack 0.1ms, release 100ms\n"
                "3. Blend aux return at 20–40% alongside dry signal\n\n"
                "**Use when:** Drums or bass lack density/punch but serial compression "
                "kills the transient. The dry signal provides transient snap while the "
                "heavily compressed aux provides sustain and density.\n\n"
                "**Not for:** Vocals, acoustic guitar — preserving dynamics is typically "
                "more important than adding density on melodic instruments."
            ),
            source="synthetic",
            category="compression",
            language="en",
            verified=True,
        ),
    ]


def _limiting_pairs() -> list[TrainingPair]:
    return [
        TrainingPair(
            instruction="How do I set up a mastering limiter to hit -1dBFS true peak?",
            input="",
            output=(
                "Set the limiter output ceiling to -1.0dBTP (true peak, not sample peak). "
                "Reduce input gain until the gain reduction meter shows 1–3dB of reduction "
                "on peaks. More than 3dB GR indicates the mix needs dynamic reduction "
                "before the limiter.\n\n"
                "**Settings guide:**\n"
                "- Ceiling: -1.0dBTP\n"
                "- Lookahead: 1.5–3ms (prevents true peak overshoots)\n"
                "- Release: 50–300ms (fast for punchy, slow for transparent)\n"
                "- Input gain: set so GR averages 1–2dB, peaks 3dB max\n\n"
                "**Check true peak:** Sample-based meters miss inter-sample peaks. "
                "Use a true peak meter (e.g. iZotope Insight, Youlean Loudness Meter). "
                "True peaks can be +3dBTP higher than the displayed sample peak."
            ),
            source="synthetic",
            category="limiting",
            language="en",
            verified=True,
        ),
        TrainingPair(
            instruction="My mix is peaking at -3dBFS but Spotify still normalizes it down. Why?",
            input="",
            output=(
                "Spotify normalizes to -14 LUFS integrated (loudness units). "
                "If your mix averages -10 LUFS (which is loud), Spotify applies "
                "-4dB of normalization, making it quieter than if you'd mastered "
                "to -14 LUFS with more dynamic range.\n\n"
                "**Solution:** Master to -14 LUFS integrated with -1dBTP true peak. "
                "This matches Spotify's normalization target, so no gain adjustment occurs. "
                "The mix sounds exactly as you mastered it.\n\n"
                "**Platform targets (LUFS integrated):**\n"
                "- Spotify: -14 LUFS\n"
                "- Apple Music: -16 LUFS\n"
                "- YouTube: -14 LUFS\n"
                "- Tidal: -14 LUFS\n"
                "- CD: -9 to -6 LUFS (loudness war era — avoid this for streaming)"
            ),
            source="synthetic",
            category="limiting",
            language="en",
            verified=True,
        ),
    ]


def _analysis_pairs() -> list[TrainingPair]:
    return [
        TrainingPair(
            instruction="My mix sounds muddy in the low-mids. What frequencies should I cut and on which tracks?",
            input="Stems: vocals, drums (kick/snare/overhead), bass guitar, electric guitar, piano",
            output=(
                "Low-mid mud typically accumulates in the 200–400Hz range. "
                "Apply these cuts across the relevant stems:\n\n"
                "**Electric guitar:** Cut 250–300Hz, -4dB, Q=1.2. Guitar body resonance "
                "in this range is the most common mud source.\n\n"
                "**Piano:** Cut 300Hz, -3dB, Q=1.0 on the lower register. "
                "Piano low-mids mask kick and bass.\n\n"
                "**Kick drum:** Cut 350–400Hz, -4dB, Q=1.0 to remove boxiness.\n\n"
                "**Bass guitar:** Cut 200–250Hz, -3dB, Q=1.0 to separate from kick fundamental.\n\n"
                "**Vocals:** High-pass at 80–100Hz, then cut 250Hz, -2dB if vocals sound 'woolly'.\n\n"
                "**Process:** High-pass every non-bass track at its lowest fundamental. "
                "Guitar: 100Hz. Piano: 40Hz. Overhead: 200Hz. "
                "Each HPF removes content competing with bass and kick."
            ),
            source="synthetic",
            category="analysis",
            language="en",
            verified=True,
        ),
        TrainingPair(
            instruction="The vocals aren't cutting through in the chorus even though the level is loud enough. What's wrong?",
            input="",
            output=(
                "Level isn't the problem — frequency masking is. "
                "The vocal is competing with other instruments in its presence range (2–5kHz). "
                "Loud doesn't mean audible if other instruments fill the same spectral space.\n\n"
                "**Fix strategy — frequency carving:**\n"
                "1. On guitars: cut 2–3kHz, -3dB, Q=1.5 (creates space for vocal presence)\n"
                "2. On synth pads: cut 1–4kHz, -4dB, Q=0.8 (pads are the most common vocal maskers)\n"
                "3. On the vocal: boost 3kHz, +2dB, Q=1.5 (presence boost to fill the carved space)\n\n"
                "**Fix strategy — automation:**\n"
                "Automate a 1–2dB level reduction on guitars and pads in the chorus bars "
                "where the vocal enters. This is more transparent than heavy EQ.\n\n"
                "**Check:** Solo the vocal and gradually unmute each instrument. "
                "The instrument that most reduces vocal clarity when added is the masking culprit."
            ),
            source="synthetic",
            category="analysis",
            language="en",
            verified=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all() -> list[TrainingPair]:
    pairs: list[TrainingPair] = []
    pairs.extend(_stem_chain_pairs())
    pairs.extend(_compressor_pairs())
    pairs.extend(_eq_pairs())
    pairs.extend(_reverb_pairs())
    pairs.extend(_gain_staging_pairs())
    pairs.extend(_parallel_compression_pairs())
    pairs.extend(_limiting_pairs())
    pairs.extend(_analysis_pairs())
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic audio engineering training pairs")
    parser.add_argument("--output", default="data/raw/synthetic_pairs.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pairs = generate_all()
    with output.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    print(f"Generated {len(pairs)} synthetic pairs → {output}")


if __name__ == "__main__":
    main()
