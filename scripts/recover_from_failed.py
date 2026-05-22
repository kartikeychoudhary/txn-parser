#!/usr/bin/env python3
"""Recover labeled rows from data/distill/failed.jsonl whose parsed_output
now passes the current validator.

Only considers rows whose original failure was VALIDATOR-driven (the rule
set evolved, so old labels may now pass). Rows that failed for transient
reasons — provider errors, JSON parse failures with no validation attempt
— are left alone; re-running stage 5 is the right fix for those.

A row is treated as validation-type if its top-level reason is
'validation_failed' OR any attempt has a validator failure_reason
(schema_invalid, superseded_amount_used, currency_mismatch,
suspicious_duplicate, validation_failed). Mirrors
_row_has_validation_failure_attempt in 05_generate_distillation_data.py.

For each qualifying row:
  - iterate attempts
  - re-run validate_example(input, attempt.parsed_output) for each attempt
    that has a non-null parsed_output
  - if any attempt passes, promote ONE attempt (prefer non-repair, then
    earliest) into train.jsonl with _source='recovered_from_failed'

Default is report-only. Pass --apply to:
  - append recovered rows to train.jsonl
  - rewrite failed.jsonl excluding the recovered rows
    (original is backed up to failed.jsonl.bak)

Usage:
    python scripts/recover_from_failed.py
    python scripts/recover_from_failed.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validator import (  # noqa: E402
    serialize_validation_result,
    validate_example,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = REPO_ROOT / "data" / "distill" / "train.jsonl"
DEFAULT_FAILED = REPO_ROOT / "data" / "distill" / "failed.jsonl"
DEFAULT_REPORT = REPO_ROOT / "data" / "distill" / "recovery_candidates.jsonl"

# Must stay in sync with _row_has_validation_failure_attempt in
# scripts/05_generate_distillation_data.py.
_VALIDATOR_FAILURE_REASONS = frozenset({
    "validation_failed", "schema_invalid", "superseded_amount_used",
    "currency_mismatch", "suspicious_duplicate",
})


def _is_validation_type_row(row: dict) -> bool:
    """True if the row originally failed for a validator reason. Rows
    that failed solely due to provider errors / JSON parse failures with
    no validator attempt are excluded — re-running stage 5 is the fix."""
    if row.get("reason") == "validation_failed":
        return True
    for attempt in row.get("attempts", []):
        if attempt.get("failure_reason") in _VALIDATOR_FAILURE_REASONS:
            return True
    return False


def _pick_best_passing_attempt(input_text: str, attempts: list[dict]):
    """Return (attempt, validation_dict) for the chosen recovery, or
    (None, None) if no attempt passes the current validator.

    Preference order:
      1. is_repair_attempt == False over True
      2. earliest index (stable)
    """
    ranked = sorted(
        enumerate(attempts),
        key=lambda iv: (bool(iv[1].get("is_repair_attempt")), iv[0]),
    )
    for _, attempt in ranked:
        parsed = attempt.get("parsed_output")
        if not isinstance(parsed, dict):
            continue
        result = validate_example(input_text, parsed, mode="strict")
        if result.ok:
            return attempt, serialize_validation_result(result)
    return None, None


def _train_row_for(input_text: str, attempt: dict) -> dict:
    return {
        "input": input_text,
        "output": attempt["parsed_output"],
        "_source": "recovered_from_failed",
        "_provider": attempt.get("provider"),
        "_model": attempt.get("model"),
        "_validation_score": 100,
        "_attempts": 1,
        "_recovered_at": datetime.now(timezone.utc).isoformat(),
    }


def _existing_train_inputs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
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
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--apply", action="store_true",
                   help="Actually move recovered rows back to train.jsonl "
                        "and rewrite failed.jsonl without them. "
                        "Originals backed up to *.bak.")
    args = p.parse_args()

    if not args.failed.exists():
        raise SystemExit(f"Failed file not found: {args.failed}")

    already_in_train = _existing_train_inputs(args.train)

    recovered_train_rows: list[str] = []     # JSONL lines for train.jsonl
    surviving_failed_rows: list[str] = []    # original lines we keep in failed.jsonl
    report_rows: list[str] = []
    provider_counts: Counter = Counter()
    parse_errors = 0
    total = 0
    duplicate_skips = 0
    non_validation_skips = 0

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
                surviving_failed_rows.append(line)
                continue
            if not isinstance(row, dict) or "input" not in row \
                    or not isinstance(row.get("attempts"), list):
                surviving_failed_rows.append(line)
                continue

            if not _is_validation_type_row(row):
                non_validation_skips += 1
                surviving_failed_rows.append(line)
                continue

            input_text = row["input"]
            attempt, validation = _pick_best_passing_attempt(
                input_text, row["attempts"],
            )
            if attempt is None:
                surviving_failed_rows.append(line)
                continue

            if input_text in already_in_train:
                duplicate_skips += 1
                surviving_failed_rows.append(line)
                continue

            train_row = _train_row_for(input_text, attempt)
            recovered_train_rows.append(
                json.dumps(train_row, ensure_ascii=False)
            )
            already_in_train.add(input_text)
            provider_counts[attempt.get("provider", "unknown")] += 1

            report_rows.append(json.dumps({
                "input": input_text,
                "previous_reason": row.get("reason"),
                "recovered_provider": attempt.get("provider"),
                "recovered_model": attempt.get("model"),
                "was_repair_attempt": bool(attempt.get("is_repair_attempt")),
                "output": attempt["parsed_output"],
                "current_validation": validation,
            }, ensure_ascii=False))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        for line in report_rows:
            f.write(line + "\n")

    if args.apply and recovered_train_rows:
        train_backup = args.train.with_suffix(args.train.suffix + ".bak")
        if args.train.exists():
            if train_backup.exists():
                train_backup.unlink()
            # Copy (don't rename) — we're appending, not replacing.
            train_backup.write_bytes(args.train.read_bytes())
        with args.train.open("a", encoding="utf-8") as f:
            for line in recovered_train_rows:
                f.write(line + "\n")

        failed_backup = args.failed.with_suffix(args.failed.suffix + ".bak")
        if failed_backup.exists():
            failed_backup.unlink()
        args.failed.rename(failed_backup)
        with args.failed.open("w", encoding="utf-8") as f:
            for line in surviving_failed_rows:
                f.write(line + "\n")

    print(f"Scanned {total} failed rows")
    print(f"  recoverable      : {len(recovered_train_rows)}")
    print(f"  still failing    : {len(surviving_failed_rows) - parse_errors - non_validation_skips}")
    if non_validation_skips:
        print(f"  skipped (non-validation failure): {non_validation_skips}")
    if duplicate_skips:
        print(f"  skipped (already in train.jsonl): {duplicate_skips}")
    if parse_errors:
        print(f"  unparseable rows kept as-is: {parse_errors}")
    if provider_counts:
        print("  recovered by provider:")
        for provider, n in provider_counts.most_common():
            print(f"    {provider}: {n}")
    print(f"  report           : {args.report}")
    if args.apply:
        if recovered_train_rows:
            print(f"  train.jsonl      : +{len(recovered_train_rows)} rows appended "
                  f"(backup at {args.train.name}.bak)")
            print(f"  failed.jsonl     : rewritten without recovered rows "
                  f"(backup at {args.failed.name}.bak)")
        else:
            print("  --apply: nothing to do (no recoverable rows).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
