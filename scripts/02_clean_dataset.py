#!/usr/bin/env python3
"""Stage 1b: Clean, validate, deduplicate, and split the raw dataset.

Loads every ``data/raw/batch_*.jsonl`` file, validates each example
against the transaction schema, deduplicates by normalised input,
splits 90/10 train/eval with seed 42, and writes
``data/clean/train.jsonl`` and ``data/clean/eval.jsonl``.

Each output record additionally carries a ``_source`` field (the source
batch filename) for traceability in the viewer.

Usage:
    python scripts/02_clean_dataset.py
    python scripts/02_clean_dataset.py --eval-frac 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import SCHEMA  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "clean"
LOGS_DIR = REPO_ROOT / "logs"


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def normalize_input(s: str) -> str:
    return " ".join(s.lower().split())


def load_raw(input_dir: Path) -> list[tuple[dict, str]]:
    files = sorted(input_dir.glob("batch_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No batch_*.jsonl files in {input_dir}")
    out: list[tuple[dict, str]] = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append((obj, fp.name))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean, validate, dedup, and split the raw dataset into train/eval."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory containing batch_*.jsonl files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write train.jsonl and eval.jsonl.")
    parser.add_argument("--eval-frac", type=float, default=0.1,
                        help="Fraction of clean examples held out for eval.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling before the split.")
    args = parser.parse_args()

    if not 0.0 < args.eval_frac < 0.5:
        parser.error("--eval-frac must be in (0, 0.5)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"02_clean_dataset_{ts}.log")

    logging.info("Loading raw batches from %s", args.input_dir)
    raw = load_raw(args.input_dir)
    n_sources = len({src for _, src in raw})
    logging.info("Loaded %d raw records from %d batch files", len(raw), n_sources)

    validator = Draft202012Validator(SCHEMA)
    drop_reasons: Counter[str] = Counter()
    seen: set[str] = set()
    kept: list[dict] = []

    for obj, src in tqdm(raw, desc="validating"):
        if not isinstance(obj, dict):
            drop_reasons["not_object"] += 1
            continue
        if "input" not in obj or "output" not in obj:
            drop_reasons["missing_keys"] += 1
            continue
        inp = obj["input"]
        if not isinstance(inp, str) or not inp.strip():
            drop_reasons["bad_input"] += 1
            continue
        out = obj["output"]
        if not isinstance(out, dict):
            drop_reasons["output_not_object"] += 1
            continue
        errors = list(validator.iter_errors(out))
        if errors:
            err = errors[0]
            # Bucket by the field that failed, so the dropped-reason summary is useful.
            path = "/".join(str(p) for p in err.absolute_path) or "<root>"
            drop_reasons[f"schema:{err.validator}@{path}"] += 1
            continue
        key = normalize_input(inp)
        if key in seen:
            drop_reasons["duplicate_input"] += 1
            continue
        seen.add(key)
        kept.append({"input": inp, "output": out, "_source": src})

    logging.info("Kept %d / %d records", len(kept), len(raw))
    for reason, n in drop_reasons.most_common():
        logging.info("  dropped %d: %s", n, reason)

    if len(kept) < 10:
        logging.error("Too few valid records (%d) to produce a usable split", len(kept))
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n_eval = max(1, int(round(len(kept) * args.eval_frac)))
    eval_set = kept[:n_eval]
    train_set = kept[n_eval:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"

    def write(path: Path, items: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(train_path, train_set)
    write(eval_path, eval_set)
    logging.info("Wrote %d train -> %s", len(train_set), train_path)
    logging.info("Wrote %d eval  -> %s", len(eval_set), eval_path)

    cat_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    txn_count_dist: Counter[int] = Counter()
    for r in kept:
        txns = r["output"]["transactions"]
        txn_count_dist[len(txns)] += 1
        for t in txns:
            cat_counts[t["category"]] += 1
            type_counts[t["type"]] += 1
    logging.info("Category distribution: %s", dict(cat_counts.most_common()))
    logging.info("Type distribution: %s", dict(type_counts))
    logging.info("Transactions-per-input distribution: %s",
                 dict(sorted(txn_count_dist.items())))

    return 0


if __name__ == "__main__":
    sys.exit(main())
