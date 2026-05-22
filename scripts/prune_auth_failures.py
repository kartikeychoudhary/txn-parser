#!/usr/bin/env python3
"""Drop rows from data/distill/failed.jsonl whose every attempt failed
with an authentication / API-key provider error.

These rows have no labeling signal in them — the provider rejected the
call before any model output existed. Removing them lets a subsequent
stage 5 run pick the inputs up cleanly (against inputs_raw.jsonl) once
credentials are valid, instead of `--retry-failed` re-erroring on the
same dead rows.

A row is pruned when:
  - top-level `reason` == "provider_error"
  - every attempt has failure_reason == "provider_error" AND
    its `error` string matches one of the auth-error patterns below.

Anything else is kept (mixed-failure rows, JSON parse failures,
validation failures, etc.).

Default is report-only. Pass --apply to rewrite failed.jsonl in place
(original backed up to failed.jsonl.bak).

Usage:
    python scripts/prune_auth_failures.py
    python scripts/prune_auth_failures.py --apply
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAILED = REPO_ROOT / "data" / "distill" / "failed.jsonl"

# Substrings (case-insensitive) that mark an error as an auth/credential
# problem rather than a model/runtime failure.
_AUTH_ERROR_PATTERNS = re.compile(
    r"AccessDeniedException"
    r"|Authentication failed"
    r"|API [Kk]ey"
    r"|UnauthorizedOperation"
    r"|InvalidSignatureException"
    r"|ExpiredTokenException"
    r"|expired token"
    r"|invalid api key"
    r"|invalid_api_key"
    r"|401",
    re.IGNORECASE,
)


def _is_auth_error(err: object) -> bool:
    if not isinstance(err, str) or not err:
        return False
    return bool(_AUTH_ERROR_PATTERNS.search(err))


def _row_is_all_auth_failures(row: dict) -> bool:
    if row.get("reason") != "provider_error":
        return False
    attempts = row.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False
    for att in attempts:
        if not isinstance(att, dict):
            return False
        if att.get("failure_reason") != "provider_error":
            return False
        if not _is_auth_error(att.get("error")):
            return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failed", type=Path, default=DEFAULT_FAILED)
    p.add_argument("--apply", action="store_true",
                   help="Rewrite failed.jsonl in place (backup at failed.jsonl.bak).")
    args = p.parse_args()

    if not args.failed.exists():
        raise SystemExit(f"Failed file not found: {args.failed}")

    kept_lines: list[str] = []
    pruned = 0
    parse_errors = 0
    total = 0
    pruned_providers: Counter = Counter()

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
            if not isinstance(row, dict):
                kept_lines.append(line)
                continue
            if _row_is_all_auth_failures(row):
                pruned += 1
                provider = row["attempts"][0].get("provider", "unknown")
                pruned_providers[provider] += 1
                continue
            kept_lines.append(line)

    print(f"Scanned {total} failed rows")
    print(f"  auth-only failures (would prune): {pruned}")
    print(f"  rows kept                        : {len(kept_lines)}")
    if parse_errors:
        print(f"  unparseable rows kept as-is      : {parse_errors}")
    if pruned_providers:
        print("  pruned by provider:")
        for provider, n in pruned_providers.most_common():
            print(f"    {provider}: {n}")

    if args.apply:
        if pruned == 0:
            print("  --apply: nothing to do.")
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
