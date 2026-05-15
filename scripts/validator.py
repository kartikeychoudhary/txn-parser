"""Semantic validator: (input_text, output_obj) -> ValidationResult.

The validator runs schema preflight (lazy import from _lib) and then
input-vs-output consistency checks. It never calls a model. Errors
returned via ValidationResult.errors carry stable machine-readable codes.

Project scripts use flat imports with scripts/ on sys.path.
"""
from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from typing import Literal

from amount_parser import AmountCandidate, parse_amounts


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationError:
    code: str
    path: str
    message: str
    severity: Severity


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[ValidationError]
    amount_values_match_active_candidates: bool
    txn_count_matches_active_candidates: bool
    duplicate_transactions_found: bool
    superseded_amount_used: bool
    candidates: list[AmountCandidate]


def _result_from(
    errors: list[ValidationError],
    *,
    mode: str,
    candidates: list[AmountCandidate],
    amount_match_active: bool = False,
    txn_count_match_active: bool = False,
    duplicate_found: bool = False,
    superseded_used: bool = False,
) -> ValidationResult:
    has_error = any(e.severity == "error" for e in errors)
    ok = True if mode == "warn" else not has_error
    return ValidationResult(
        ok=ok, errors=errors, candidates=candidates,
        amount_values_match_active_candidates=amount_match_active,
        txn_count_matches_active_candidates=txn_count_match_active,
        duplicate_transactions_found=duplicate_found,
        superseded_amount_used=superseded_used,
    )


_REPETITION_CUES = (
    "twice", "2x", "two times", "do baar",
    "again", "another", "aur ek", "phir se",
)


def _matches_amount(c: AmountCandidate, amount: float) -> bool:
    return math.isclose(c.value, amount, abs_tol=0.001)


def _input_supports_repeat(input_text: str, amount: float, item: str,
                           candidates: list[AmountCandidate],
                           dup_count: int) -> bool:
    """A duplicate group of size dup_count is justified iff:
      1. same amount appears >= dup_count times AND same item phrase
         appears >= dup_count times in the input, OR
      2. a repetition cue appears in the input.

    v1: cue match is global (anywhere in the input). Spec wording allows
    'near the amount/item phrase'; tightening to a span-based check is
    deferred to a follow-up parser slice.
    """
    lower = input_text.lower()
    if any(cue in lower for cue in _REPETITION_CUES):
        return True
    amount_count = sum(1 for c in candidates if _matches_amount(c, amount))
    item_count = len(re.findall(rf"\b{re.escape(item.lower())}\b", lower))
    return amount_count >= dup_count and item_count >= dup_count


def validate_example(
    input_text: str,
    output_obj: dict,
    *,
    mode: Literal["strict", "warn"] = "strict",
) -> ValidationResult:
    from _lib import is_schema_valid, schema_errors

    candidates = parse_amounts(input_text)

    if not is_schema_valid(output_obj):
        errs = [
            ValidationError(
                code="SCHEMA_INVALID", path="/transactions",
                message=err, severity="error",
            )
            for err in schema_errors(output_obj)
        ]
        return _result_from(errs, mode=mode, candidates=candidates)

    errors: list[ValidationError] = []
    active = [c for c in candidates if c.status == "active"]
    txns = output_obj["transactions"]

    # Step 3 — no-amount precedence.
    skip_per_txn_amount = False
    if len(active) == 0 and len(txns) >= 1:
        errors.append(ValidationError(
            code="NO_AMOUNT_IN_INPUT", path="/transactions",
            message="No amount candidates parsed from input but transactions emitted",
            severity="error",
        ))
        skip_per_txn_amount = True

    # Step 4 — per-transaction amount check.
    amount_match_active = True
    superseded_used = False
    if not skip_per_txn_amount:
        for i, txn in enumerate(txns):
            matches = [c for c in candidates if _matches_amount(c, txn["amount"])]
            path = f"/transactions/{i}/amount"
            if not matches:
                errors.append(ValidationError(
                    code="AMOUNT_NOT_IN_INPUT", path=path,
                    message=f"Predicted amount {txn['amount']} not in candidates "
                            f"{[c.value for c in candidates]}",
                    severity="error",
                ))
                amount_match_active = False
            elif all(c.status == "superseded" for c in matches):
                errors.append(ValidationError(
                    code="SUPERSEDED_AMOUNT_USED", path=path,
                    message=f"Predicted amount {txn['amount']} matches only a "
                            f"superseded candidate",
                    severity="error",
                ))
                superseded_used = True
                amount_match_active = False
    else:
        amount_match_active = False

    # Step 5 — per-transaction currency check (per matched active candidate).
    for i, txn in enumerate(txns):
        matched_active = [
            c for c in candidates
            if c.status == "active" and _matches_amount(c, txn["amount"])
        ]
        hints = {c.currency_hint for c in matched_active if c.currency_hint}
        if hints and txn["currency"] not in hints:
            errors.append(ValidationError(
                code="CURRENCY_MISMATCH",
                path=f"/transactions/{i}/currency",
                message=f"Txn currency {txn['currency']!r} does not match "
                        f"matched-candidate hints {sorted(hints)}",
                severity="error",
            ))

    # Step 6 — transaction count check.
    txn_count_match_active = (len(txns) == len(active))
    if len(txns) > len(active):
        errors.append(ValidationError(
            code="TXN_COUNT_EXCEEDS_CANDIDATES", path="/transactions",
            message=f"{len(txns)} transactions but only {len(active)} active candidates",
            severity="error",
        ))
    elif len(txns) < len(active):
        errors.append(ValidationError(
            code="TXN_COUNT_BELOW_CANDIDATES", path="/transactions",
            message=f"{len(txns)} transactions but {len(active)} active candidates",
            severity="warning",
        ))

    # Step 7 — duplicate detection.
    keys: list[tuple] = []
    for t in txns:
        keys.append((
            round(float(t["amount"]) * 1000) / 1000.0,
            t["item"].lower().strip(),
            t["category"], t["type"], t["currency"],
        ))
    dup_groups: dict[tuple, int] = {}
    for k in keys:
        dup_groups[k] = dup_groups.get(k, 0) + 1

    duplicate_found = False
    for i, k in enumerate(keys):
        if dup_groups[k] < 2:
            continue
        first_occurrence = keys.index(k)
        if i == first_occurrence:
            continue
        amount = k[0]; item = k[1]
        if _input_supports_repeat(input_text, amount, item, candidates, dup_groups[k]):
            continue
        duplicate_found = True
        errors.append(ValidationError(
            code="SUSPICIOUS_DUPLICATE", path=f"/transactions/{i}",
            message=f"Transaction is an unjustified duplicate of /transactions/{first_occurrence}",
            severity="error",
        ))

    return _result_from(
        errors, mode=mode, candidates=candidates,
        amount_match_active=amount_match_active,
        txn_count_match_active=txn_count_match_active,
        duplicate_found=duplicate_found,
        superseded_used=superseded_used,
    )


def serialize_validation_result(result: ValidationResult) -> dict:
    """Stable JSON-encodable shape. Excludes `candidates` so callers can
    write them at a different top-level key in failed.jsonl."""
    return {
        "ok": result.ok,
        "amount_values_match_active_candidates": result.amount_values_match_active_candidates,
        "txn_count_matches_active_candidates": result.txn_count_matches_active_candidates,
        "duplicate_transactions_found": result.duplicate_transactions_found,
        "superseded_amount_used": result.superseded_amount_used,
        "errors": [dataclasses.asdict(e) for e in result.errors],
    }
