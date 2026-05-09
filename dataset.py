"""Shared types, validation, and patterns for NeuralMix dataset pipeline."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# Detects specific audio engineering parameter values in output text.
# Used in is_valid_pair to reject vague advice with no concrete values.
MIXING_PARAM_PATTERN = re.compile(
    r"-?\d+(?:\.\d+)?\s*dB(?:FS)?"     # dB values: -18dBFS, +3dB, -6.5dB
    r"|\d+(?:\.\d+)?\s*kHz"              # frequencies: 3.2kHz, 2.4kHz
    r"|\d+(?:\.\d+)?\s*Hz"               # frequencies: 200Hz, 3200Hz
    r"|\d+:\d+"                           # ratios: 4:1, 2:1, 8:1
    r"|\d+(?:\.\d+)?\s*ms"               # times: 10ms, 80ms, 200ms
    r"|\d+(?:\.\d+)?\s*s\b"              # times: 1.5s, 2s
    r"|Q\s*=\s*\d+(?:\.\d+)?"           # Q factor: Q=2.8, Q=0.7
    r"|\d+(?:\.\d+)?%\s*(?:wet|dry|mix)" # wet/dry: 50% wet, 30% mix
)

# Categories that REQUIRE specific parameter values in the output.
TECHNICAL_CATEGORIES = {"eq", "compression", "reverb_delay", "limiting", "stem_chain"}

API_NAMES_PATH = Path("data/audio_terms.json")
_terms_cache: set[str] | None = None


def load_audio_terms(path: Path = API_NAMES_PATH) -> set[str]:
    global _terms_cache
    if _terms_cache is None:
        if path.exists():
            _terms_cache = set(json.loads(path.read_text(encoding="utf-8")))
        else:
            _terms_cache = set()
    return _terms_cache


def extract_param_values(text: str) -> list[str]:
    """Return all specific parameter values found in text."""
    return MIXING_PARAM_PATTERN.findall(text)


@dataclass
class TrainingPair:
    instruction: str
    input: str                  # mixing context: stem description, problem, measurements
    output: str                 # specific FX chain / parameters / rationale
    source: Literal["handbook", "plugin_docs", "forums", "tutorials", "synthetic", "neuralmix_logs"]
    category: Literal["eq", "compression", "reverb_delay", "limiting", "stem_chain", "analysis", "concepts"]
    language: Literal["en", "th"]
    verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingPair":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def is_valid_pair(pair: TrainingPair, _terms: set[str] | None = None) -> bool:
    """Return True if the pair is suitable for training."""
    if len(pair.instruction.strip()) < 20:
        return False
    if len(pair.output.strip()) < 80:
        return False
    if len(pair.output) > 4500:
        return False

    # Technical categories must have at least one concrete parameter value.
    # Reject vague advice like "try boosting the highs" with no specific values.
    if pair.category in TECHNICAL_CATEGORIES:
        if not MIXING_PARAM_PATTERN.search(pair.output):
            return False

    # Reject outputs that give no actionable recommendation at all.
    output_lower = pair.output.lower()
    vague_only = all(
        phrase not in output_lower
        for phrase in ["db", "hz", "ratio", "attack", "release", "threshold",
                       "frequency", "eq", "compress", "reverb", "delay", "gain"]
    )
    if pair.category in TECHNICAL_CATEGORIES and vague_only:
        return False

    return True
