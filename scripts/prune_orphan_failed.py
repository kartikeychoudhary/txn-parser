#!/usr/bin/env python3
"""Drop rows from data/distill/failed.jsonl whose `input` is now present
in data/distill/train.jsonl.

Stage 5's retry paths (`--retry-validation-failed`, `--retry-failed`)
APPEND new success rows to train.jsonl but never remove the original
failure rows. Over repeated retry passes, failed.jsonl accumulates
orphan rows for inputs that have since been labeled successfully.

This script reconciles that: it reads train.jsonl into a set of
successful inputs and rewrites failed.jsonl with only the rows whose
input is NOT in that set.

Default is report-only. Pass --apply to rewrite failed.jsonl
(original backed up to failed.jsonl.bak).

Usage:
    python scripts/prune_orphan_failed.py
    python scripts/prune_orphan_failed.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = REPO_ROOT / "data" / "distill" / "train.jsonl"
DEFAULT_FAILED = REPO_ROOT / "data" / "distill" / "failed.jsonl"


def _load_inputs(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("input"), str):
                out.add(row["input"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--failed", type=Path, default=DEFAULT_FAILED)
    p.add_argument("--apply", action="store_true",
                   help="Rewrite failed.jsonl in place (backup at failed.jsonl.bak).")
    args = p.parse_args()

    if not args.failed.exists():
        raise SystemExit(f"Failed file not found: {args.failed}")
    if not args.train.exists():
        raise SystemExit(f"Train file not found: {args.train}")

    train_inputs = _load_inputs(args.train)

    kept_lines: list[str] = []
    pruned = 0
    parse_errors = 0
    total = 0
    pruned_reasons: Counter = Counter()

    with args.failed.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                kept_lines.append(line)
                continue
            if not isinstance(row, dict) or not isinstance(row.get("input"), str):
                kept_lines.append(line)
                continue
            if row["input"] in train_inputs:
                pruned += 1
                pruned_reasons[row.get("reason", "unknown")] += 1
                continue
            kept_lines.append(line)

    print(f"Scanned {total} failed rows against {len(train_inputs)} train inputs")
    print(f"  orphan rows (input now in train.jsonl): {pruned}")
    print(f"  rows kept: {len(kept_lines)}")
    if parse_errors:
        print(f"  unparseable rows kept as-is: {parse_errors}")
    if pruned_reasons:
        print("  orphan breakdown by original reason:")
        for reason, n in pruned_reasons.most_common():
            print(f"    {reason}: {n}")

    if args.apply:
        if pruned == 0:
            print("  --apply: nothing to do (no orphans).")
            return 0
        backup = args.failed.with_suffix(args.failed.suffix + ".bak")
        if backup.exists():
            backup.unlink()
        args.failed.rename(backup)
        with args.failed.open("w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")
        print(f"  failed.jsonl rewritten (backup at {backup.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
