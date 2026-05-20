"""Pure scoring + selection logic for multi-provider label candidates.

No I/O, no model deps. All inputs are explicit and all outputs are
derived deterministically. Tests are pure unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateOutcome:
    """A single provider's attempt at labeling one input."""
    provider: str
    model: str | None
    raw_output: str
    parsed_output: dict | None
    validation: dict | None
    failure_reason: str | None
    score: int
    is_repair_attempt: bool
    provider_priority_rank: int | None = None
    error: str | None = None


def score_candidate(
    parsed_output: dict | None,
    validation: dict | None,
    failure_reason: str | None,
    provider_priority_rank: int | None = None,
) -> int:
    """Hard-reject any candidate with a failure_reason or any error-severity
    validator failure. Survivors get a base 80 plus bonuses for count match
    (+10), no warnings (+5), and provider priority rank 0 (+5). Max 100.
    """
    if failure_reason is not None or parsed_output is None or validation is None:
        return 0
    error_codes = {
        e["code"] for e in validation.get("errors", []) if e["severity"] == "error"
    }
    if error_codes:
        return 0
    score = 80
    if validation.get("txn_count_matches_active_candidates"):
        score += 10
    warning_codes = {
        e["code"] for e in validation.get("errors", []) if e["severity"] == "warning"
    }
    if not warning_codes:
        score += 5
    if provider_priority_rank == 0:
        score += 5
    return score


def pick_best(outcomes: list[CandidateOutcome]) -> CandidateOutcome | None:
    """Return the highest-scoring non-failure outcome, or None if all failed.

    Tie-break order:
      1. Higher score wins.
      2. Lower provider_priority_rank wins (None treated as +inf).
      3. Alphabetical provider name (stable, deterministic).
    """
    valid = [o for o in outcomes if o.failure_reason is None and o.score > 0]
    if not valid:
        return None
    valid.sort(key=lambda o: (
        -o.score,
        o.provider_priority_rank if o.provider_priority_rank is not None else float("inf"),
        o.provider,
    ))
    return valid[0]


_REPAIR_PROMPT_TEMPLATE = """The previous label attempt for this input failed validation.

INPUT:
{input_text}

PARSED AMOUNT CANDIDATES (from deterministic parser):
{candidates_summary}

FAILURE SUMMARY:
{failure_summary}

Generate a corrected JSON label that:
1. Uses ONLY amounts from the parsed candidates above (do not invent new amounts).
2. Marks the correct currency per the candidate's currency_hint.
3. Has the correct number of transactions matching the active candidates.
4. Does NOT include any superseded amounts (markers like "wait no", "actually").
5. Does NOT duplicate transactions unless the input explicitly repeats them.

Output ONLY the JSON object."""


def build_repair_prompt(
    input_text: str,
    *,
    candidates: list[dict] | None = None,
    failure_summary: str,
) -> str:
    """Build a stricter prompt for a repair attempt."""
    if candidates:
        lines = [
            f"  - {c['raw']!r} -> {c['value']} "
            f"({c.get('currency_hint') or '(no hint, default INR)'}, {c['status']})"
            for c in candidates
        ]
        candidates_summary = "\n".join(lines)
    else:
        candidates_summary = "  (no amount candidates parsed)"
    return _REPAIR_PROMPT_TEMPLATE.format(
        input_text=input_text,
        candidates_summary=candidates_summary,
        failure_summary=failure_summary,
    )
