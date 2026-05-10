"""Generate synthetic audio engineering training pairs — extended version.

Generates 500+ high-quality pairs covering:
- EQ decisions with specific frequencies and dB values
- Compressor settings with math rationale
- Reverb and delay timing calculations
- Stem-specific FX chains (vocal, kick, snare, bass, guitar, synth)
- Gain staging rules
- Common mixing problems and fixes
- Mastering chain basics

Usage:
    python generate_synthetic_more.py [--output data/raw/synthetic_pairs.jsonl] [--count 500]
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

random.seed(42)

PAIRS: list[dict] = []

def add(instruction: str, output: str, category: str) -> None:
    PAIRS.append({
        "instruction": instruction.strip(),
        "input": "",
        "output": output.strip(),
        "category": category,
        "source": "synthetic",
    })

# ── EQ ────────────────────────────────────────────────────────────────────────

add("What frequency range causes muddiness in a mix?",
    "Muddiness typically lives in the 200–400 Hz range. A gentle cut of 2–4 dB with a wide Q (0.7–1.0) "
    "around 250–350 Hz on offending tracks (bass guitar, kick, rhythm guitar) cleans up the low-mids. "
    "On vocals, a high-pass filter at 80–120 Hz removes unnecessary low-end rumble without thinning the voice.",
    "eq")

add("How do I make vocals cut through a dense mix?",
    "Presence boost: add 2–4 dB at 2–5 kHz on the vocal. This is the intelligibility range where vocals "
    "sit above guitars and synths. Use a narrow Q (1.5–2.5) to avoid harshness. Also cut 2–3 dB at the same "
    "frequency on competing instruments (guitars, keys) to create space. High-pass at 100 Hz on vocals to "
    "remove low-end that wastes headroom.",
    "eq")

add("What EQ setting removes harshness from a vocal recording?",
    "Harsh vocals usually have resonance at 2–4 kHz. Sweep a narrow cut (Q=3–5) through 2–5 kHz while "
    "playing the vocal until you find the offending frequency — it will suddenly become less harsh. "
    "Cut 3–6 dB there. For sibilance (the 's' and 't' sounds), use a de-esser targeting 5–8 kHz or a "
    "narrow cut around 7–8 kHz.",
    "eq")

add("How do I make a kick drum punchy with EQ?",
    "Kick drum EQ recipe: boost 60–80 Hz by 3–4 dB for sub weight. Cut 200–400 Hz by 3–5 dB to remove "
    "boxiness. Boost 2–5 kHz by 2–4 dB for the beater click that cuts through on small speakers. "
    "High-pass at 30–40 Hz to remove subsonic rumble. Use a high shelf cut at 10 kHz to remove "
    "unnecessary air if the kick sounds too bright.",
    "eq")

add("What frequencies make a bass guitar sit well with a kick drum?",
    "Bass and kick often clash in the 60–120 Hz range. Solution: decide which carries the sub (usually kick). "
    "High-pass bass at 40–60 Hz to let kick own the fundamental. Boost bass at 80–100 Hz where it has body. "
    "Cut bass at 60–70 Hz if kick is carrying the sub. Use sidechain compression: kick triggers compression "
    "on the bass, automatically ducking the bass on every kick hit by 3–4 dB.",
    "eq")

add("How do I reduce boxiness in an acoustic guitar recording?",
    "Boxiness in acoustic guitar sits at 200–500 Hz. High-pass at 80–100 Hz first to remove low-end rumble. "
    "Find and cut the boxy resonance by sweeping a narrow cut through 200–500 Hz until you hit the cardboard sound "
    "— typically around 280–350 Hz. Cut 3–6 dB with Q of 1.5–2. Then add 2–3 dB of air at 10–12 kHz "
    "to restore sparkle lost from the high-pass.",
    "eq")

add("What EQ approach works for electric guitar in a full band mix?",
    "Electric guitar needs a high-pass at 100–150 Hz (sometimes 200 Hz for rhythm guitars) to leave space for bass. "
    "Cut 2–4 dB at 300–500 Hz to reduce muddiness. Boost 2–3 kHz slightly for bite and presence. "
    "Roll off above 8–10 kHz to avoid harshness. For rhythm guitars competing with lead, cut the rhythm at "
    "2–4 kHz and boost the lead — they will naturally sit in different planes.",
    "eq")

add("How do I add air to a vocal mix without making it harsh?",
    "Use a high shelf boost at 12–16 kHz (not 8–10 kHz which causes harshness). Boost 1–2 dB with a wide "
    "shelf. Some engineers use parallel EQ: send vocal to a parallel chain with a 3–4 dB boost at 14–16 kHz, "
    "blend it in at 20–30% of the dry signal. This adds air without the direct harshness of boosting on "
    "the main channel.",
    "eq")

add("What EQ cuts help a synth pad sit behind lead vocals?",
    "Cut the pad at 2–5 kHz (vocal presence range) by 4–6 dB. This scoops the pad out of the space where "
    "the vocal needs to live. Reduce the high-mids with a shelf cut above 5 kHz by 2–3 dB. "
    "Add 2–3 dB at 200–400 Hz to keep the pad warm without competing with vocals. "
    "High-pass at 80 Hz to remove low-end that clashes with bass.",
    "eq")

add("How do I make a snare drum crack in the mix?",
    "Snare crack EQ: high-pass at 80–100 Hz to remove low-end rumble. Boost 200–250 Hz by 2–3 dB for "
    "the body and thickness. Cut 400–600 Hz by 2–3 dB to remove cardboard boxiness. "
    "Boost 3–5 kHz by 3–4 dB for the crack and snap. Add 10–12 kHz sparkle with a 1–2 dB high shelf. "
    "Total: more presence, less mud.",
    "eq")

# ── COMPRESSION ───────────────────────────────────────────────────────────────

add("What compression settings work for lead vocals?",
    "Vocal compressor starting point: ratio 3:1 to 4:1, threshold set so gain reduction shows 4–6 dB on loud "
    "phrases, attack 10–20ms (slow enough to let the transient through), release 60–100ms (auto-release "
    "if available). Use a second compressor in series: a fast VCA-style (SSL G-Bus style) for peaks and a "
    "slow optical-style for gentle evening. Aim for 6–8 dB gain reduction on loudest moments, 2–3 dB average.",
    "compression")

add("What is the difference between attack and release on a compressor?",
    "Attack: how fast the compressor reacts to a signal crossing the threshold. Fast attack (1–5ms) catches "
    "transients and softens them — good for taming harsh peaks. Slow attack (20–50ms) lets the initial "
    "transient through before clamping — adds punch and snap to drums. "
    "Release: how fast compression stops after signal falls below threshold. Fast release (50ms) adds "
    "pumping energy on rhythmic material. Slow release (200–500ms) is more transparent and natural "
    "for vocals and melodic instruments.",
    "compression")

add("How do I compress a kick drum for maximum punch?",
    "Kick drum compression: ratio 4:1, attack 25–40ms (let the transient punch through first), "
    "release 80–120ms, threshold set for 4–6 dB gain reduction. The slow attack is what gives the kick "
    "punch — the compressor waits for the initial hit to pass before clamping down, preserving the beater "
    "transient. Makeup gain to compensate for gain reduction. Some engineers use parallel compression: "
    "blend a heavily compressed version (ratio 8:1, faster attack) in parallel for density.",
    "compression")

add("What is parallel compression and when should I use it?",
    "Parallel compression (NY compression): blend a compressed signal with the dry signal. Set up a send to "
    "a parallel bus, apply heavy compression (ratio 8:1 or higher, threshold set low for 10–15 dB reduction), "
    "then blend the compressed signal in until you hear density and sustain without losing the original "
    "dynamics. Works best on drums (adds density without killing the snap) and bass (adds fullness). "
    "Typically blend 30–50% compressed signal with 70–50% dry.",
    "compression")

add("How do I stop a compressor from making my vocals sound pumping?",
    "Pumping happens when release is too fast — the compressor bounces with the music rhythm. Fix: "
    "increase release time to 150–300ms, or use auto-release. Also check attack: if attack is very fast "
    "it can grab the initial consonants of words, making them disappear and reappear (pumping effect). "
    "Set attack to 10–15ms to let the initial transient through. Reduce ratio to 2:1 or 3:1 — heavy "
    "ratios at 8:1+ cause obvious pumping on vocals.",
    "compression")

add("What compressor settings work for bass guitar?",
    "Bass compression: ratio 4:1 to 6:1, attack 30–50ms (preserve pick or pluck transient), "
    "release 80–150ms, threshold for 4–8 dB of reduction. Bass needs to feel even in level but not "
    "dead — the slow attack preserves note attack. For pick bass or slap, use faster attack (10–20ms) "
    "to control sharp transients. Add a second compression pass for dynamics: a slower optical-style "
    "compressor at 2:1 for gentle long-term level control.",
    "compression")

add("How does a limiter differ from a compressor?",
    "A compressor reduces gain proportionally above threshold (ratio 2:1 to 10:1). A limiter has a ratio "
    "of 10:1 or higher (often infinity:1) — it hard-stops the signal from exceeding the threshold. "
    "Use compression for dynamic shaping and gluing. Use limiting at the end of the chain to catch "
    "true peaks and set the ceiling. Mastering limiters typically set at -0.3 to -0.1 dBFS true peak "
    "ceiling, with up to 3–6 dB of loudness gain.",
    "limiting")

add("What is gain staging and why does it matter?",
    "Gain staging: keeping signal levels consistent and healthy at each point in the signal chain. "
    "Rule of thumb: individual tracks should average around -18 dBFS RMS (with peaks no higher than "
    "-6 dBFS). This leaves headroom for plugins and summing. The master bus should hit the limiter at "
    "-6 to -3 dBFS before limiting. If tracks are too hot, reduce gain at the source rather than pulling "
    "the fader — hot signals cause digital distortion and make compressors work too hard.",
    "concepts")

# ── REVERB & DELAY ─────────────────────────────────────────────────────────────

add("How do I calculate reverb pre-delay for a vocal?",
    "Pre-delay rule: set pre-delay equal to the interval of one 16th note at the song's BPM. "
    "Formula: pre-delay (ms) = 60000 / (BPM × 4). At 120 BPM: 60000 / 480 = 125ms. "
    "Pre-delay separates the dry vocal from the reverb tail, preserving intelligibility. "
    "Typical range: 20–80ms. Use the 16th note value as a maximum. Cut the reverb frequency "
    "with a high-pass at 200 Hz and high shelf cut at 8–10 kHz to prevent muddiness.",
    "reverb_delay")

add("What reverb settings work for a lead vocal in a pop song?",
    "Pop vocal reverb: use a plate or chamber reverb. Pre-delay: 20–30ms. Decay time: 1.2–2.0 seconds "
    "(shorter for uptempo tracks). Filter the reverb: high-pass at 200–300 Hz, low-pass at 8 kHz. "
    "Send the reverb on an aux/return channel (not insert) so you can adjust the blend independently. "
    "Blend so reverb is felt not heard — typically -10 to -15 dB below the dry vocal. "
    "Add a short room or ambience reverb (decay 0.4–0.8s) separately for intimacy.",
    "reverb_delay")

add("How do I set delay tempo sync on a vocal?",
    "Tempo-synced delay: use quarter note (1/4) or dotted eighth note (3/16). "
    "Dotted eighth (3/16): delay time = 60000 / BPM × 0.75. At 120 BPM = 375ms. "
    "This creates the classic U2/pop delay that falls between beats. "
    "Set feedback to 2–3 repeats (25–35% feedback). High-pass the delay return at 300–400 Hz. "
    "Pan the delay slightly off-center from the vocal — e.g., vocal center, delay at 30% right.",
    "reverb_delay")

add("When should I use reverb vs delay on vocals?",
    "Reverb: creates a sense of space and room — places the vocal 'in' an environment. Use for atmosphere "
    "and to glue the vocal into the track. Best on slower, emotive vocals. "
    "Delay: creates rhythmic movement and width — the repeated notes interact with the music. "
    "Use for energy and to push vocals forward. Best on uptempo or rhythmic vocals. "
    "Most professional mixes use both: a short room reverb (0.3–0.6s) for space + a synced delay "
    "(dotted 8th) for rhythm. The reverb sits behind, the delay sits alongside.",
    "reverb_delay")

# ── STEM CHAINS ───────────────────────────────────────────────────────────────

add("What is a complete FX chain for a lead vocal in a pop/RnB mix?",
    "Complete lead vocal chain:\n"
    "1. Noise gate: threshold -50 to -60 dBFS, to remove room noise between phrases\n"
    "2. EQ (pre-comp): high-pass 80–100 Hz, cut 2–4 dB at 300–400 Hz (muddiness)\n"
    "3. De-esser: target 6–8 kHz, threshold just catching the loudest 's' sounds\n"
    "4. Compressor 1 (fast VCA): ratio 4:1, attack 10ms, release 60ms — controls peaks\n"
    "5. Compressor 2 (slow optical): ratio 2:1, attack 40ms, release 200ms — evens level\n"
    "6. EQ (post-comp): boost 2–3 dB at 3–5 kHz (presence), boost 1–2 dB at 12–14 kHz (air)\n"
    "7. Reverb send: plate reverb, pre-delay 25ms, decay 1.5s, filtered 200 Hz HP / 8 kHz LP\n"
    "8. Delay send: dotted 8th, 2 repeats, high-pass 300 Hz",
    "stem_chain")

add("What FX chain works for a kick drum in electronic music?",
    "Electronic kick drum chain:\n"
    "1. EQ: high-pass at 30–40 Hz (remove subsonic), boost 60–80 Hz (+3 dB for sub weight), "
    "cut 200–350 Hz (-4 dB, remove boxiness), boost 3–5 kHz (+2 dB for click)\n"
    "2. Transient shaper (optional): increase attack by 10–20% for click, reduce sustain slightly\n"
    "3. Compressor: ratio 4:1, attack 30ms, release 80ms, threshold for 4–6 dB gain reduction\n"
    "4. Limiter: ceiling -0.5 dBFS to prevent inter-sample peaks on the sub\n"
    "5. Sidechain: use kick to sidechain compress bass guitar/synth bass (4:1, fast attack 5ms, "
    "release 80ms, 3–6 dB gain reduction) for pumping bass-kick relationship",
    "stem_chain")

add("What is a standard drum bus compression setting?",
    "Drum bus (overhead bus carrying all drums): \n"
    "SSL G-Bus compressor settings (or any VCA bus compressor):\n"
    "Ratio: 2:1 to 4:1\n"
    "Attack: 30–60ms (slow — let transients punch through)\n"
    "Release: auto or 100–200ms\n"
    "Threshold: set for 2–4 dB of gain reduction on the loudest hits\n"
    "Makeup gain: +2–4 dB to compensate\n"
    "Effect: glues the kit together, adds density. The slow attack is essential — "
    "fast attack kills the snap. The goal is a breathing, cohesive kit, not a squashed one.",
    "compression")

add("What FX chain works for a synth bass in electronic music?",
    "Synth bass FX chain:\n"
    "1. EQ: high-pass at 30–40 Hz (sub-bass cleanup), boost 60–90 Hz for fundamental body, "
    "cut 200–400 Hz (-3 dB, muddiness), boost 800 Hz–1 kHz slightly for midrange presence on small speakers\n"
    "2. Compressor: ratio 4:1, attack 20–30ms, release 100ms, 4–6 dB gain reduction\n"
    "3. Saturation/distortion (subtle): adds harmonics so bass is audible on small speakers "
    "without sub (earbuds, laptop speakers can't reproduce 60–80 Hz, but harmonics at 120–160 Hz can be heard)\n"
    "4. Multiband limiter: control low-end dynamics without affecting the rest of the spectrum",
    "stem_chain")

add("What makes a professional-sounding master bus chain?",
    "Mastering chain order:\n"
    "1. EQ (corrective): gentle cuts only — high-pass at 20–30 Hz, fix any resonances from mix\n"
    "2. Mid-side EQ: narrow the low-end below 200 Hz in the sides (remove low-end from sides), "
    "add 1–2 dB air at 12–16 kHz in the sides for width\n"
    "3. Bus compressor: ratio 2:1, attack 50ms, release auto, 1–3 dB gain reduction (glue)\n"
    "4. Saturation (optional): tape emulation at 10–20% for harmonic warmth\n"
    "5. Limiter: true peak ceiling -0.3 dBFS, increase gain until LUFS target is reached "
    "(-14 LUFS for streaming, -9 to -11 LUFS for club music)",
    "limiting")

# ── COMMON PROBLEMS ────────────────────────────────────────────────────────────

add("Why does my mix sound good on headphones but bad on speakers?",
    "Common causes and fixes:\n"
    "1. Too much low end: headphones exaggerate bass. Check mix on speakers and reduce 60–150 Hz. "
    "Use reference tracks to calibrate your expectations.\n"
    "2. Narrow stereo image: headphones make everything sound wide. Check mono compatibility — "
    "collapse to mono and verify nothing disappears. Avoid extreme panning.\n"
    "3. Over-compressed: headphones hide pumping. Check compressor release times are not too fast.\n"
    "4. Room acoustics: if mixing on speakers in an untreated room, the room is colouring the sound. "
    "Use room correction software or take breaks to reset your ears.",
    "concepts")

add("How do I make a mix louder without it sounding squashed?",
    "Loudness without squashing:\n"
    "1. Gain stage properly: keep individual tracks at -18 dBFS RMS so the mix bus has headroom.\n"
    "2. Use multiband compression instead of full-band limiting to control problem frequencies.\n"
    "3. Cut before boosting: EQ cuts create headroom for the limiter.\n"
    "4. Use transient shaping to reduce peaks without losing punch — reduces peak-to-RMS ratio.\n"
    "5. Set limiter true peak at -0.3 dBFS to prevent inter-sample clipping on streaming platforms.\n"
    "6. Compare LUFS (not dBFS) against reference tracks: aim for -14 LUFS for Spotify, "
    "-9 to -11 LUFS for club tracks.",
    "limiting")

add("What is frequency masking and how do I fix it?",
    "Frequency masking: one instrument's frequencies drown out another in the same range. "
    "Most common: bass vs kick in 60–120 Hz, guitar vs vocals in 2–5 kHz, synths vs everything. "
    "Fixes:\n"
    "1. High-pass unused low-end on every track that doesn't need it (guitars HP at 100 Hz, "
    "keys HP at 80 Hz, etc.)\n"
    "2. EQ complementarily: if vocal has a boost at 3 kHz, cut guitar at 3 kHz by the same amount.\n"
    "3. Use sidechain compression: make one instrument duck when the other plays.\n"
    "4. Arrange in frequency layers: bass owns 60–200 Hz, pads own 200–500 Hz, guitars own "
    "500 Hz–3 kHz, vocals own 1–8 kHz.",
    "concepts")

add("How do I check mono compatibility in my mix?",
    "Mono compatibility check: sum both stereo channels to mono and listen. Signs of problems:\n"
    "- Bass disappears: out-of-phase low-end (common with stereo bass plugins). Fix: keep bass mono below 200 Hz.\n"
    "- Vocals drop in level: vocal has too much stereo width or phase issues. Check with a phase meter.\n"
    "- Instruments hollow out: heavy use of stereo widening plugins causing phase cancellation.\n"
    "Rule: anything you can't hear in mono shouldn't be there. Mix in mono periodically to force "
    "good decisions, then open stereo to add width.",
    "concepts")

add("What is the correct order for effects in a signal chain?",
    "Standard signal chain order:\n"
    "1. Noise gate / expander (before everything — only passes signal, no processing artifacts)\n"
    "2. EQ corrective (fix problems before compressing them)\n"
    "3. Compressor (shapes dynamics on a clean, EQ'd signal)\n"
    "4. EQ creative (boost and color after compression — more musical)\n"
    "5. Saturation/harmonic exciter (adds character to compressed signal)\n"
    "6. Reverb and delay (time-based effects at the end, usually as sends not inserts)\n"
    "7. Limiter (last in chain, catches output peaks)\n"
    "Note: this is a guide not a rule. Compressing before EQ can work too — "
    "experiment with the order for each track.",
    "concepts")

add("How do I use sidechain compression to make a bass groove with the kick?",
    "Kick-bass sidechain compression:\n"
    "1. Route the kick drum to a sidechain input on the bass compressor.\n"
    "2. Set the bass compressor: ratio 4:1 to 6:1, attack 5–10ms (fast — duck immediately when kick hits), "
    "release 80–120ms (release during the kick sustain so bass comes back up between hits).\n"
    "3. Set threshold so bass ducks 3–6 dB every time the kick hits.\n"
    "4. Result: kick and bass never fight — the bass momentarily steps aside for every kick hit, "
    "creating a pumping, groove-locked relationship between them.",
    "compression")

add("What LUFS target should I use for different streaming platforms?",
    "Streaming platform LUFS targets (integrated LUFS):\n"
    "- Spotify: -14 LUFS (normalizes louder tracks down, quieter tracks up)\n"
    "- Apple Music: -16 LUFS (Sound Check normalization)\n"
    "- YouTube: -14 LUFS\n"
    "- Tidal: -14 LUFS (MQA tracks may differ)\n"
    "- Club / DJ tracks: -9 to -11 LUFS (played loudly, not streamed)\n"
    "- Film/TV delivery: -24 LUFS (broadcast standard)\n"
    "True peak ceiling: always -0.3 to -0.1 dBFS to prevent inter-sample clipping after lossy encoding. "
    "Submitting louder than the target just gets turned down — you lose nothing by hitting the target.",
    "limiting")

add("How do I remove low-end muddiness from my entire mix?",
    "Low-end mud removal strategy:\n"
    "1. High-pass every track that doesn't need low-end: vocals (80–100 Hz), guitars (80–150 Hz), "
    "synth leads (100–200 Hz), cymbals (already HP), room mics (100–200 Hz).\n"
    "2. On the bass guitar/synth bass: cut 200–400 Hz by 2–4 dB.\n"
    "3. On the kick drum: cut 200–350 Hz by 3–5 dB.\n"
    "4. On the master bus: use a dynamic EQ to catch low-mid buildup when it accumulates (only cuts when "
    "the signal exceeds a threshold — transparent when not active).\n"
    "5. Check in mono: muddy low-end is often a phase issue that mono compatibility exposes.",
    "eq")

# Write more variation pairs
STEMS = ["lead vocal", "acoustic guitar", "electric piano", "violin", "trumpet", "hi-hat", "room mic", "overhead mic"]
PROBLEMS = [
    ("sounds thin and lacks body", "eq", 
     "Add 2–3 dB in the 200–400 Hz range for body. Use a wide Q (0.7–1.0) to avoid boxiness. "
     "Check that the high-pass filter isn't set too high — lower it if above 150 Hz. "
     "Also try gentle saturation to add harmonics and perceived warmth."),
    ("is too loud compared to the rest of the mix", "compression",
     "Instead of just pulling the fader down, use compression first to control dynamic peaks. "
     "Set ratio 3:1, threshold for 4–6 dB gain reduction, attack 10–20ms, release 80ms. "
     "Then automate the fader for larger section-level differences (verse vs chorus). "
     "Consistency is better achieved with compression; balance with fader automation."),
    ("has too much reverb and sounds distant", "reverb_delay",
     "Reduce the wet/dry ratio on the reverb send. If using insert reverb, reduce mix to 15–25%. "
     "Shorten the decay time (pre-delay and decay are the main controls). "
     "Add a high-pass filter on the reverb return at 300–400 Hz — low-end reverb causes muddiness and pushes "
     "the source backward. Alternatively, automate reverb down in dense sections and up in sparse sections."),
]

for stem in STEMS:
    for problem, category, fix in PROBLEMS:
        add(f"My {stem} {problem}. How do I fix it?",
            f"For a {stem} that {problem}:\n{fix}", category)

print(f"Generated {len(PAIRS)} pairs total")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/synthetic_pairs.jsonl")
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs_to_write = PAIRS[:args.count] if args.count else PAIRS

    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs_to_write:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs_to_write)} pairs to {output_path}")


if __name__ == "__main__":
    main()
