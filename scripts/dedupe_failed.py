#!/usr/bin/env python3
"""Collapse data/distill/failed.jsonl so there is one row per unique input.

Stage 5's retry paths and successive labeling runs APPEND a fresh failed-row
each time an input is re-attempted. The same input can accumulate many rows
across runs, which inflates the file and makes downstream tools (recovery,
revalidation) do redundant work.

This script merges all rows for the same input into a single row:
  - attempts: union of all attempts across rows, deduped by
    (provider, model, raw_output, is_repair_attempt)
  - candidates: taken from the row chosen as 'head' (see below)
  - reason: the most-informative reason across the rows (priority below)

Reason priority (head row picked by lowest rank):
  1. validation_failed       (model produced output that was validated)
  2. repair_exhausted        (validator-driven, has attempts with output)
  3. all_candidates_failed   (validator-driven, has attempts with output)
  4. provider_error          (transient API failure, no model output)
  5. anything else / missing

Default is report-only. Pass --apply to rewrite failed.jsonl in place
(original backed up to failed.jsonl.bak).

Usage:
    python scripts/dedupe_failed.py
    python scripts/dedupe_failed.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAILED = REPO_ROOT / "data" / "distill" / "failed.jsonl"

_REASON_RANK = {
    "validation_failed": 1,
    "repair_exhausted": 2,
    "all_candidates_failed": 3,
    "provider_error": 4,
}


def _reason_rank(reason: str | None) -> int:
    return _REASON_RANK.get(reason or "", 99)


def _attempt_key(a: dict) -> tuple:
    """Identity for attempt-level dedup. Same provider+raw_output+repair flag
    means the same retry; we keep only the first."""
    return (
        a.get("provider"),
        a.get("model"),
        a.get("raw_output"),
        bool(a.get("is_repair_attempt")),
    )


def _merge_rows(rows: list[dict]) -> dict:
    """Collapse multiple failed-rows for one input into a single row."""
    # Pick a 'head' row (most informative reason; tiebreak by total attempt count).
    head = min(
        rows,
        key=lambda r: (_reason_rank(r.get("reason")), -len(r.get("attempts") or [])),
    )

    merged_attempts: list[dict] = []
    seen_keys: set[tuple] = set()
    # Iterate rows in their original order; head's attempts come first so the
    # 'best' attempt (per attempt order within head) stays at the top.
    ordered_rows = [head] + [r for r in rows if r is not head]
    for row in ordered_rows:
        for a in (row.get("attempts") or []):
            k = _attempt_key(a)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            merged_attempts.append(a)

    return {
        "input": head["input"],
        "reason": head.get("reason"),
        "candidates": head.get("candidates") or [],
        "attempts": merged_attempts,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failed", type=Path, default=DEFAULT_FAILED)
    p.add_argument("--apply", action="store_true",
                   help="Rewrite failed.jsonl in place (backup at failed.jsonl.bak).")
    args = p.parse_args()

    if not args.failed.exists():
        raise SystemExit(f"Failed file not found: {args.failed}")

    # Preserve input-order across the file by keeping the first-seen position
    # per input as the order key.
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    parse_errors = 0
    skipped_no_input: list[str] = []
    total = 0

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
                continue
            if not isinstance(row, dict) or not isinstance(row.get("input"), str):
                skipped_no_input.append(line)
                continue
            grouped.setdefault(row["input"], []).append(row)

    out_lines: list[str] = []
    n_inputs_with_dups = 0
    rows_eliminated = 0
    reason_after: Counter = Counter()

    for inp, rows in grouped.items():
        if len(rows) > 1:
            n_inputs_with_dups += 1
            rows_eliminated += len(rows) - 1
            merged = _merge_rows(rows)
        else:
            merged = rows[0]
        reason_after[merged.get("reason") or "unknown"] += 1
        out_lines.append(json.dumps(merged, ensure_ascii=False))

    print(f"Scanned {total} failed rows")
    print(f"  unique inputs            : {len(grouped)}")
    print(f"  inputs with duplicates   : {n_inputs_with_dups}")
    print(f"  duplicate rows to remove : {rows_eliminated}")
    if parse_errors:
        print(f"  unparseable rows dropped : {parse_errors}")
    if skipped_no_input:
        print(f"  rows w/o usable 'input'  : {len(skipped_no_input)} (kept verbatim)")
    print("  reason distribution after dedup:")
    for reason, n in reason_after.most_common():
        print(f"    {reason}: {n}")

    if args.apply:
        if rows_eliminated == 0 and parse_errors == 0:
            print("  --apply: nothing to do (already deduped).")
            return 0
        backup = args.failed.with_suffix(args.failed.suffix + ".bak")
        if backup.exists():
            backup.unlink()
        args.failed.rename(backup)
        with args.failed.open("w", encoding="utf-8") as f:
            for line in out_lines:
                f.write(line + "\n")
            for line in skipped_no_input:
                f.write(line + "\n")
        print(f"  failed.jsonl rewritten (backup at {backup.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
