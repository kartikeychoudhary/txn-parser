#!/usr/bin/env python3
"""Re-run the current validator over an existing data/distill/train.jsonl.

Useful after validator changes: surfaces rows that were accepted by an
older validator but would be rejected today.

Default is report-only (writes a JSONL of failing rows + a summary).
Pass --prune to additionally rewrite train.jsonl in place keeping only
passing rows; rejected rows are appended to failed.jsonl in the same
shape produced by stage 5 (input/reason/candidates/attempts).

Usage:
    python scripts/revalidate_train.py
    python scripts/revalidate_train.py --train data/distill/train.jsonl --mode strict
    python scripts/revalidate_train.py --prune
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validator import (  # noqa: E402
    serialize_validation_result,
    validate_example,
)
from amount_parser import parse_amounts  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = REPO_ROOT / "data" / "distill" / "train.jsonl"
DEFAULT_FAILED = REPO_ROOT / "data" / "distill" / "failed.jsonl"
DEFAULT_REPORT = REPO_ROOT / "data" / "distill" / "revalidation_failures.jsonl"


def _serialize_candidate(c) -> dict:
    return {
        "value": c.value, "raw": c.raw, "span": list(c.span),
        "status": c.status, "source": c.source, "currency_hint": c.currency_hint,
    }


def _primary_failure_reason(validation: dict) -> str:
    codes = {e["code"] for e in validation["errors"] if e["severity"] == "error"}
    if "SCHEMA_INVALID" in codes:
        return "schema_invalid"
    if "SUPERSEDED_AMOUNT_USED" in codes:
        return "superseded_amount_used"
    if "CURRENCY_MISMATCH" in codes:
        return "currency_mismatch"
    if "SUSPICIOUS_DUPLICATE" in codes:
        return "suspicious_duplicate"
    return "validation_failed"


def _failed_row_for(row: dict, validation: dict, candidates) -> dict:
    """Shape mirrors stage 5's failed.jsonl rows (a synthetic 'attempt'
    representing the original labeled output)."""
    provider = row.get("_provider", "unknown")
    model = row.get("_model")
    attempt = {
        "provider": provider,
        "model": model,
        "raw_output": "",
        "parsed_output": row["output"],
        "validation": validation,
        "score": 0,
        "failure_reason": _primary_failure_reason(validation),
        "is_repair_attempt": False,
        "provider_priority_rank": 0,
        "error": None,
        "_source": "revalidation",
        "_revalidated_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "input": row["input"],
        "reason": "validation_failed",
        "candidates": [_serialize_candidate(c) for c in candidates],
        "attempts": [attempt],
    }


def _process_line(payload: tuple[str, str, bool]) -> dict:
    """Worker: validate one train.jsonl line.

    Returns a dict with keys: status ('parse_error' | 'ok' | 'fail'),
    line (the original JSON line, for passing rows), report_line,
    failed_jsonl_line, reason.

    Defined at module level so it pickles for ProcessPoolExecutor.
    """
    line, mode, want_prune = payload
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return {"status": "parse_error"}
    if not isinstance(row, dict) or "input" not in row or "output" not in row:
        return {"status": "parse_error"}

    result = validate_example(row["input"], row["output"], mode=mode)
    if result.ok:
        return {"status": "ok", "line": line}

    validation = serialize_validation_result(result)
    reason = _primary_failure_reason(validation)
    report_line = json.dumps({
        "input": row["input"],
        "output": row["output"],
        "validation": validation,
        "primary_failure_reason": reason,
        "_provider": row.get("_provider"),
        "_model": row.get("_model"),
    }, ensure_ascii=False)

    failed_jsonl_line = None
    if want_prune:
        # Re-parse candidates only when pruning (cost saved on report-only runs).
        candidates = parse_amounts(row["input"])
        failed_jsonl_line = json.dumps(
            _failed_row_for(row, validation, candidates),
            ensure_ascii=False,
        )
    return {
        "status": "fail",
        "report_line": report_line,
        "failed_jsonl_line": failed_jsonl_line,
        "reason": reason,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN,
                   help=f"Path to train.jsonl (default: {DEFAULT_TRAIN})")
    p.add_argument("--failed", type=Path, default=DEFAULT_FAILED,
                   help="Path to failed.jsonl, used only with --prune")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                   help="Where to write failing rows (always written)")
    p.add_argument("--mode", choices=["strict", "warn"], default="strict",
                   help="Validator mode (default: strict)")
    p.add_argument("--prune", action="store_true",
                   help="Rewrite train.jsonl in place (passing rows only) "
                        "and append failures to failed.jsonl. "
                        "Original train.jsonl is backed up to train.jsonl.bak.")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel worker processes. 0 = use os.cpu_count(). "
                        "Pass 1 to force the serial (single-process) path.")
    p.add_argument("--chunksize", type=int, default=200,
                   help="ProcessPoolExecutor.map chunksize. Larger reduces "
                        "IPC overhead for fast-per-row workloads.")
    args = p.parse_args()

    if not args.train.exists():
        raise SystemExit(f"Train file not found: {args.train}")

    with args.train.open(encoding="utf-8") as f:
        lines = [raw.strip() for raw in f if raw.strip()]
    total = len(lines)

    # Order-preserving payload so passing rows are written back in input order.
    payloads = [(line, args.mode, args.prune) for line in lines]
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    passing: list[str] = []
    failing_report: list[str] = []
    failing_for_failed_jsonl: list[str] = []
    reason_counts: Counter = Counter()
    parse_errors = 0

    if workers == 1 or total < 2 * args.chunksize:
        # Serial path: fewer rows than would amortize process startup.
        results_iter = (_process_line(p) for p in payloads)
    else:
        # executor.map preserves input order; chunksize amortizes IPC cost.
        executor = ProcessPoolExecutor(max_workers=workers)
        results_iter = executor.map(_process_line, payloads, chunksize=args.chunksize)

    try:
        for res in results_iter:
            status = res["status"]
            if status == "parse_error":
                parse_errors += 1
                continue
            if status == "ok":
                passing.append(res["line"])
                continue
            failing_report.append(res["report_line"])
            reason_counts[res["reason"]] += 1
            if args.prune and res["failed_jsonl_line"] is not None:
                failing_for_failed_jsonl.append(res["failed_jsonl_line"])
    finally:
        if workers != 1 and total >= 2 * args.chunksize:
            executor.shutdown(wait=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        for line in failing_report:
            f.write(line + "\n")

    if args.prune:
        backup = args.train.with_suffix(args.train.suffix + ".bak")
        if backup.exists():
            backup.unlink()
        args.train.rename(backup)
        with args.train.open("w", encoding="utf-8") as f:
            for line in passing:
                f.write(line + "\n")
        if failing_for_failed_jsonl:
            args.failed.parent.mkdir(parents=True, exist_ok=True)
            with args.failed.open("a", encoding="utf-8") as f:
                for line in failing_for_failed_jsonl:
                    f.write(line + "\n")

    bad = len(failing_report)
    print(f"Re-validated {total} rows (mode={args.mode})")
    print(f"  passing : {len(passing)}")
    print(f"  failing : {bad}")
    if parse_errors:
        print(f"  unparseable rows skipped: {parse_errors}")
    if reason_counts:
        print("  failure breakdown:")
        for reason, n in reason_counts.most_common():
            print(f"    {reason}: {n}")
    print(f"  report  : {args.report}")
    if args.prune:
        print(f"  pruned  : train.jsonl rewritten, backup at {backup.name}")
        print(f"            {len(failing_for_failed_jsonl)} rows appended to {args.failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
