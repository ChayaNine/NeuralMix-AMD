"""Merge, deduplicate, validate, and split all raw NeuralMix training data.

Fixed version:
- Loads raw JSON dicts directly (no dataclass — avoids field mismatch errors)
- Less aggressive validation (short but correct pairs pass through)
- Works with whatever pairs exist in data/raw/

Usage:
    python build_dataset.py [--eval-seed data/eval_seed.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

RAW_DIR    = Path("data/raw")
TRAIN_PATH = Path("data/train.jsonl")
EVAL_PATH  = Path("data/eval.jsonl")
EVAL_SIZE  = 50
RANDOM_SEED = 42

# Very loose thresholds — audio pairs are short but correct
MIN_INSTRUCTION_LEN = 10
MIN_OUTPUT_LEN      = 15
MAX_OUTPUT_LEN      = 8000


def load_raw_pairs(raw_dir: Path) -> list[dict]:
    all_pairs = []
    if not raw_dir.exists():
        print(f"WARNING: {raw_dir} does not exist.")
        return []

    for jsonl_file in sorted(raw_dir.glob("*.jsonl")):
        file_pairs = []
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pair = {
                        "instruction": str(obj.get("instruction", obj.get("question", ""))).strip(),
                        "input":       str(obj.get("input", obj.get("context", ""))).strip(),
                        "output":      str(obj.get("output", obj.get("answer", ""))).strip(),
                        "category":    str(obj.get("category", "general")).strip(),
                        "source":      str(obj.get("source", jsonl_file.name)).strip(),
                    }
                    file_pairs.append(pair)
                except Exception:
                    pass
        print(f"  {jsonl_file.name}: {len(file_pairs)} pairs loaded")
        all_pairs.extend(file_pairs)
    return all_pairs


def is_valid(pair: dict) -> bool:
    instruction = pair.get("instruction", "")
    output      = pair.get("output", "")
    if len(instruction) < MIN_INSTRUCTION_LEN:
        return False
    if len(output) < MIN_OUTPUT_LEN:
        return False
    if len(output) > MAX_OUTPUT_LEN:
        return False
    bad = ["n/a", "none", "todo", "tbd", "placeholder", "unknown", ""]
    if output.lower().strip() in bad:
        return False
    return True


def deduplicate(pairs: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for p in pairs:
        key = p["instruction"].lower().strip()[:200]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def split_eval(pairs: list[dict], eval_size: int, seed_path: Path | None):
    random.seed(RANDOM_SEED)

    if seed_path and seed_path.exists():
        eval_pairs = []
        with open(seed_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        eval_pairs.append(json.loads(line))
                    except Exception:
                        pass
        eval_keys = {p["instruction"].lower().strip() for p in eval_pairs}
        train_pairs = [p for p in pairs if p["instruction"].lower().strip() not in eval_keys]
        print(f"  Eval: {len(eval_pairs)} from seed file")
        print(f"  Train: {len(train_pairs)}")
        return train_pairs, eval_pairs

    # Auto-select: proportional from each category
    categories: dict[str, list] = {}
    for p in pairs:
        cat = p.get("category", "general")
        categories.setdefault(cat, []).append(p)

    eval_pairs = []
    for cat, cat_pairs in sorted(categories.items()):
        n = max(1, round(eval_size * len(cat_pairs) / len(pairs)))
        sampled = random.sample(cat_pairs, min(n, len(cat_pairs)))
        eval_pairs.extend(sampled)

    if len(eval_pairs) > eval_size:
        eval_pairs = eval_pairs[:eval_size]
    elif len(eval_pairs) < eval_size:
        remaining = [p for p in pairs if p not in eval_pairs]
        random.shuffle(remaining)
        eval_pairs.extend(remaining[:eval_size - len(eval_pairs)])

    eval_keys = {p["instruction"].lower().strip() for p in eval_pairs}
    train_pairs = [p for p in pairs if p["instruction"].lower().strip() not in eval_keys]

    print(f"  Eval: {len(eval_pairs)} auto-selected")
    print(f"  Train: {len(train_pairs)}")
    return train_pairs, eval_pairs


def write_jsonl(pairs: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-seed", default=None)
    args = parser.parse_args()
    seed_path = Path(args.eval_seed) if args.eval_seed else None

    print("Loading raw data...")
    raw_pairs = load_raw_pairs(RAW_DIR)
    print(f"Total raw pairs: {len(raw_pairs)}")

    if not raw_pairs:
        print("ERROR: No pairs found. Run scrapers first.")
        return

    print(f"\nDeduplicating {len(raw_pairs)} pairs...")
    deduped = deduplicate(raw_pairs)
    print(f"  After dedup: {len(deduped)} pairs")

    print(f"\nValidating...")
    valid = [p for p in deduped if is_valid(p)]
    print(f"  Valid: {len(valid)}  |  Removed: {len(deduped) - len(valid)}")

    cats: dict[str, int] = {}
    for p in valid:
        cats[p.get("category", "general")] = cats.get(p.get("category", "general"), 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    if len(valid) < 100:
        print(f"\nWARNING: Only {len(valid)} valid pairs — proceeding anyway.")
        print("100+ pairs is enough for a demo fine-tune.")

    print(f"\nSplitting into train/eval...")
    actual_eval = min(EVAL_SIZE, max(10, len(valid) // 4))
    train_pairs, eval_pairs = split_eval(valid, actual_eval, seed_path)

    write_jsonl(train_pairs, TRAIN_PATH)
    write_jsonl(eval_pairs, EVAL_PATH)

    print(f"\nWrote {len(train_pairs)} training pairs → {TRAIN_PATH}")
    print(f"Wrote {len(eval_pairs)} eval pairs → {EVAL_PATH}")

    if len(train_pairs) < 100:
        print(f"\nWARNING: Only {len(train_pairs)} training pairs.")
        print("Run generate_synthetic.py again with --count 500 for more pairs.")
    else:
        print(f"\nOK: {len(train_pairs)} pairs is sufficient for fine-tuning.")

    print("\nNext steps:")
    print("  python train.py --smoke-test --steps 10")
    print(f"  python train.py --train {TRAIN_PATH} --eval {EVAL_PATH} --epochs 5")


if __name__ == "__main__":
    main()
