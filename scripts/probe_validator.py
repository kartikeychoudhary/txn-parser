#!/usr/bin/env python3
"""Standalone validator probe.

Reads a JSONL of {input, output} pairs and reports how many pass the
semantic validator. No GPU, no API, no model load.

Usage:
    python scripts/probe_validator.py --input data/distill/train.jsonl
    python scripts/probe_validator.py --input <jsonl> --output reports/probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

# Flat-import setup — match the repo convention.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import validate_example, serialize_validation_result  # noqa: E402


def iter_jsonl_rows(path: Path) -> Iterator[tuple[int, str, dict | None, str | None]]:
    """Yield (line_no, raw, parsed_or_None, malformed_reason_or_None) for each non-empty line."""
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                yield line_no, raw.rstrip("\n"), json.loads(text), None
            except json.JSONDecodeError as e:
                yield line_no, raw.rstrip("\n"), None, f"json_parse_error: {e}"


def probe(
    rows: Iterator[tuple[int, str, dict | None, str | None]],
    *,
    mode: str = "strict",
    output_file=None,
) -> dict:
    """Walk rows, validate, accumulate counts. Optionally write per-row JSONL.

    Returns summary dict with keys:
      rows_scanned, malformed, ok, failed, failure_counts, warning_counts
    """
    rows_scanned = 0
    malformed = 0
    ok = 0
    failed = 0
    failure_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()

    for line_no, raw, obj, malformed_reason in rows:
        rows_scanned += 1
        if malformed_reason is not None:
            malformed += 1
            if output_file:
                output_file.write(json.dumps({
                    "line": line_no, "raw": raw,
                    "validator_ok": None, "validation": None,
                    "malformed": malformed_reason,
                }, ensure_ascii=False) + "\n")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("input"), str) \
                or not isinstance(obj.get("output"), dict):
            malformed += 1
            if output_file:
                output_file.write(json.dumps({
                    "line": line_no, "raw": raw,
                    "validator_ok": None, "validation": None,
                    "malformed": "missing 'input' string or 'output' dict",
                }, ensure_ascii=False) + "\n")
            continue
        result = validate_example(obj["input"], obj["output"], mode=mode)
        # Severity-based classification — independent of result.ok.
        error_codes = [e.code for e in result.errors if e.severity == "error"]
        warning_codes = [e.code for e in result.errors if e.severity == "warning"]
        row_ok = len(error_codes) == 0
        if row_ok:
            ok += 1
        else:
            failed += 1
            for code in error_codes:
                failure_counts[code] += 1
        for code in warning_codes:
            warning_counts[code] += 1
        if output_file:
            output_file.write(json.dumps({
                "line": line_no,
                "input": obj["input"],
                "output": obj["output"],
                "validator_ok": row_ok,
                "validation": serialize_validation_result(result),
            }, ensure_ascii=False) + "\n")
    return {
        "rows_scanned": rows_scanned,
        "malformed": malformed,
        "ok": ok,
        "failed": failed,
        "failure_counts": failure_counts,
        "warning_counts": warning_counts,
    }


def render_summary(input_path: Path, summary: dict) -> str:
    lines: list[str] = []
    lines.append(f"Validator probe — {input_path}")
    lines.append(f"Rows scanned:      {summary['rows_scanned']}")
    lines.append(f"Malformed:         {summary['malformed']}")
    total_well_formed = summary["rows_scanned"] - summary["malformed"]
    if total_well_formed:
        ok_pct = summary["ok"] / total_well_formed * 100
        fail_pct = summary["failed"] / total_well_formed * 100
        lines.append(f"OK:                {summary['ok']}   ({ok_pct:.1f}%)")
        lines.append(f"Failed:            {summary['failed']}   ({fail_pct:.1f}%)")
    else:
        lines.append(f"OK:                {summary['ok']}")
        lines.append(f"Failed:            {summary['failed']}")
    if summary["failure_counts"]:
        lines.append("")
        lines.append("Failures by code:")
        for code, count in sorted(
            summary["failure_counts"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            lines.append(f"  {code:<32} {count}")
    if summary["warning_counts"]:
        lines.append("")
        lines.append("Warnings by code:")
        for code, count in sorted(
            summary["warning_counts"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            lines.append(f"  {code:<32} {count}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone validator probe.")
    p.add_argument("--input", type=Path, required=True,
                   help="JSONL of {input, output} pairs.")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional per-row report JSONL; parent dirs created automatically.")
    p.add_argument("--limit", type=int, default=0,
                   help="Probe only the first N rows. 0 = all.")
    p.add_argument("--mode", choices=("strict", "warn"), default="strict",
                   help="Validator mode. Summary classification is severity-based "
                        "regardless of mode (so 'warn' does not hide failures).")
    p.add_argument("--fail-on-malformed", action="store_true",
                   help="Exit 3 when any malformed row is encountered.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-row progress (no-op here; summary is always printed).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_file = args.output.open("w", encoding="utf-8")
    else:
        output_file = None

    try:
        rows = iter_jsonl_rows(args.input)
        if args.limit > 0:
            def truncate(it, n):
                for i, x in enumerate(it):
                    if i >= n:
                        break
                    yield x
            rows = truncate(rows, args.limit)
        summary = probe(rows, mode=args.mode, output_file=output_file)
    finally:
        if output_file is not None:
            output_file.close()

    print(render_summary(args.input, summary))

    if args.fail_on_malformed and summary["malformed"] > 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
