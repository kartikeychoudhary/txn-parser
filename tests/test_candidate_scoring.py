import pytest

from label_selection import (
    CandidateOutcome,
    build_repair_prompt,
    pick_best,
    score_candidate,
)


def _ok_validation(**overrides) -> dict:
    base = {
        "ok": True,
        "amount_values_match_active_candidates": True,
        "txn_count_matches_active_candidates": True,
        "duplicate_transactions_found": False,
        "superseded_amount_used": False,
        "errors": [],
    }
    base.update(overrides)
    return base


def _validation_with(errors: list[dict], **overrides) -> dict:
    base = _ok_validation()
    base["errors"] = errors
    base["ok"] = not any(e["severity"] == "error" for e in errors)
    base.update(overrides)
    return base


# ---- score_candidate ------------------------------------------------------

def test_score_perfect_candidate():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_ok_validation(),
        failure_reason=None,
        provider_priority_rank=0,
    )
    assert score == 100


def test_score_hard_reject_returns_zero():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_ok_validation(),
        failure_reason="json_parse_failed",
    )
    assert score == 0


def test_score_parsed_none_returns_zero():
    score = score_candidate(
        parsed_output=None,
        validation=_ok_validation(),
        failure_reason=None,
    )
    assert score == 0


def test_score_validation_none_returns_zero():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=None,
        failure_reason=None,
    )
    assert score == 0


def test_score_any_error_severity_returns_zero():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_validation_with([
            {"code": "AMOUNT_NOT_IN_INPUT", "path": "/transactions/0/amount",
             "message": "x", "severity": "error"},
        ]),
        failure_reason=None,
        provider_priority_rank=0,
    )
    assert score == 0


def test_score_clean_pass_no_priority_bonus():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_ok_validation(),
        failure_reason=None,
        provider_priority_rank=1,
    )
    # 80 base + 10 count + 5 no-warnings + 0 priority = 95
    assert score == 95


def test_score_warning_only_with_count_mismatch():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_validation_with(
            [{"code": "TXN_COUNT_BELOW_CANDIDATES", "path": "/transactions",
              "message": "x", "severity": "warning"}],
            txn_count_matches_active_candidates=False,
        ),
        failure_reason=None,
        provider_priority_rank=0,
    )
    # 80 base + 0 count + 0 warnings + 5 priority = 85
    assert score == 85


def test_score_no_priority_rank_at_all():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_ok_validation(),
        failure_reason=None,
        provider_priority_rank=None,
    )
    # 80 + 10 + 5 + 0 = 95
    assert score == 95


# ---- pick_best ------------------------------------------------------------

def _make_outcome(provider="a", score=80, failure_reason=None, rank=None):
    return CandidateOutcome(
        provider=provider, model=None, raw_output="", parsed_output={} if failure_reason is None else None,
        validation=_ok_validation() if failure_reason is None else None,
        failure_reason=failure_reason, score=score, is_repair_attempt=False,
        provider_priority_rank=rank,
    )


def test_pick_best_returns_highest_score():
    a = _make_outcome("a", score=50)
    b = _make_outcome("b", score=95)
    c = _make_outcome("c", score=80)
    assert pick_best([a, b, c]).provider == "b"


def test_pick_best_returns_none_when_all_failed():
    a = _make_outcome("a", score=0, failure_reason="json_parse_failed")
    assert pick_best([a]) is None


def test_pick_best_ignores_zero_score_outcomes():
    a = _make_outcome("a", score=0, failure_reason="provider_error")
    b = _make_outcome("b", score=70)
    assert pick_best([a, b]).provider == "b"


def test_pick_best_tie_break_by_provider_priority_rank():
    low = _make_outcome("alpha", score=95, rank=2)
    high = _make_outcome("zebra", score=95, rank=0)
    # Tied score; rank 0 beats rank 2 regardless of alphabetical order.
    assert pick_best([low, high]).provider == "zebra"


def test_pick_best_tie_break_by_provider_name_when_priority_equal():
    a = _make_outcome("zebra", score=95, rank=1)
    b = _make_outcome("alpha", score=95, rank=1)
    # Tied score AND tied rank; alphabetical name wins.
    assert pick_best([a, b]).provider == "alpha"


def test_pick_best_treats_none_rank_as_lowest_priority():
    high = _make_outcome("alpha", score=95, rank=0)
    none = _make_outcome("zebra", score=95, rank=None)
    # rank None is treated as +infinity (lowest priority).
    assert pick_best([high, none]).provider == "alpha"


# ---- build_repair_prompt --------------------------------------------------

def test_build_repair_prompt_includes_input_and_failures():
    prompt = build_repair_prompt(
        "500 beer",
        candidates=[{"value": 500.0, "raw": "500", "span": [0, 3],
                     "status": "active", "source": "digits", "currency_hint": None}],
        failure_summary="fake_a: AMOUNT_NOT_IN_INPUT",
    )
    assert "500 beer" in prompt
    assert "AMOUNT_NOT_IN_INPUT" in prompt
    assert "500" in prompt
    assert "Output ONLY the JSON object" in prompt


def test_build_repair_prompt_handles_empty_candidates():
    prompt = build_repair_prompt(
        "no amounts here",
        candidates=[],
        failure_summary="fake_a: NO_AMOUNT_IN_INPUT",
    )
    assert "no amount candidates parsed" in prompt
    assert "NO_AMOUNT_IN_INPUT" in prompt


def test_build_repair_prompt_handles_none_candidates():
    prompt = build_repair_prompt(
        "x",
        candidates=None,
        failure_summary="x: y",
    )
    assert "no amount candidates parsed" in prompt
