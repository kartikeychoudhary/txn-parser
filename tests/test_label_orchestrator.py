"""Direct helper-level tests for the multi-provider label orchestrator.

Exercises _try_one_attempt, _process_one_input, and the fixture-failure
mapping without subprocess invocation.
"""
import importlib
import threading
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "fake_labels_with_failures.jsonl"


def _orchestrator():
    """Import the orchestrator module (filename starts with digit)."""
    return importlib.import_module("05_generate_distillation_data")


# ---- Fake provider helpers ------------------------------------------------

class _SuccessProvider:
    name = "succ"
    model = "fake-model-1"
    provider_type = "fake"

    def generate_label(self, text):
        return '{"transactions":[{"amount":500,"currency":"INR","item":"beer","category":"Drinks","type":"expense"}]}'


class _RaisingProvider:
    name = "rais"
    model = None
    provider_type = "fake"

    def generate_label(self, text):
        raise RuntimeError("simulated provider failure")


class _BadJsonProvider:
    name = "badj"
    model = "fake-model-2"
    provider_type = "fake"

    def generate_label(self, text):
        return "{not json"


class _ValidatorFailingProvider:
    """Returns a schema-valid but amount-invented label."""
    name = "valid_fail"
    model = "fake-model-3"
    provider_type = "fake"

    def generate_label(self, text):
        return '{"transactions":[{"amount":999,"currency":"INR","item":"beer","category":"Drinks","type":"expense"}]}'


class _SchemaInvalidProvider:
    name = "sch_inv"
    model = "fake-model-4"
    provider_type = "fake"

    def generate_label(self, text):
        return '{"transactions":[{"amount":100}]}'


class _PcfgStub:
    def __init__(self, name):
        self.name = name


# ---- _try_one_attempt path tests -------------------------------------------

def test_try_one_attempt_success():
    orch = _orchestrator()
    p = _SuccessProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("succ"),
    )
    assert outcome.failure_reason is None
    assert outcome.score > 0
    assert outcome.parsed_output["transactions"][0]["amount"] == 500
    assert outcome.is_repair_attempt is False


def test_try_one_attempt_provider_error():
    orch = _orchestrator()
    p = _RaisingProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("rais"),
    )
    assert outcome.failure_reason == "provider_error"
    assert outcome.score == 0
    assert outcome.error and "simulated provider failure" in outcome.error
    assert outcome.parsed_output is None
    assert outcome.validation is None


def test_try_one_attempt_json_parse_failed():
    orch = _orchestrator()
    p = _BadJsonProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("badj"),
    )
    assert outcome.failure_reason == "json_parse_failed"
    assert outcome.score == 0
    assert outcome.parsed_output is None


def test_try_one_attempt_schema_invalid():
    orch = _orchestrator()
    p = _SchemaInvalidProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("sch_inv"),
    )
    assert outcome.failure_reason == "schema_invalid"
    assert outcome.score == 0


def test_try_one_attempt_validation_failed():
    orch = _orchestrator()
    p = _ValidatorFailingProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("valid_fail"),
    )
    assert outcome.failure_reason == "validation_failed"
    assert outcome.score == 0
    assert outcome.parsed_output is not None
    assert outcome.validation is not None


def test_try_one_attempt_is_repair_attempt_flag():
    orch = _orchestrator()
    p = _SuccessProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("succ"),
        is_repair=True,
        failure_summary="prev failure",
        parser_candidates=[],
    )
    assert outcome.is_repair_attempt is True
    assert outcome.failure_reason is None


# ---- Fixture failure-reason mapping ---------------------------------------

def test_fixture_failure_codes_match_expectation():
    """Each row in fake_labels_with_failures.jsonl maps to a specific
    attempt-level failure reason (or None for clean pass). If validator
    code drifts, this test catches it first."""
    import json
    orch = _orchestrator()
    rows = []
    with FIXTURE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    expected = {
        "500 beer": None,
        "schema fail case": "schema_invalid",
        "json fail case": "json_parse_failed",
        "no amount input": "validation_failed",
        "500 beer wait no 600 beer": "superseded_amount_used",
        "500 beer once": "suspicious_duplicate",
    }

    for row in rows:
        if "raw_output" in row:
            payload = row["raw_output"]
        else:
            payload = json.dumps(row["output"], separators=(",", ":"))

        class FakeProv:
            name = "fp"
            model = None
            def generate_label(self_inner, text):
                return payload

        outcome = orch._try_one_attempt(
            row["input"], provider=FakeProv(), pcfg=_PcfgStub("fp"),
        )
        assert outcome.failure_reason == expected[row["input"]], (
            f"row {row['input']!r}: got {outcome.failure_reason!r}, "
            f"expected {expected[row['input']]!r}"
        )


# ---- _process_one_input integration ---------------------------------------

class _StubScheduler:
    """Deterministic scheduler that yields names from a pre-set list."""
    def __init__(self, names: list[str]):
        self._names = names
        self._idx = 0

    def next_provider(self) -> str:
        name = self._names[self._idx % len(self._names)]
        self._idx += 1
        return name


class _CfgStub:
    """Minimal cfg duck-type used by _process_one_input."""
    class _Og:
        label_attempts_per_input = 2
        providers = []
        provider_priority = ()
    class _V:
        retry_invalid_with_stricter_prompt = False
        max_repair_attempts = 0
    output_generation = _Og()
    validation = _V()


def test_process_one_input_picks_best_of_two():
    """Two providers; only one returns a valid label; best is the valid one."""
    orch = _orchestrator()
    providers = {
        "succ": _SuccessProvider(),
        "rais": _RaisingProvider(),
    }
    pcfgs = {n: _PcfgStub(n) for n in providers}
    scheduler = _StubScheduler(["succ", "rais"])
    cfg = _CfgStub()

    inp, best, all_outcomes = orch._process_one_input(
        "500 beer",
        providers_by_name=providers,
        pcfgs_by_name=pcfgs,
        scheduler=scheduler,
        priority_map={"succ": 0, "rais": 1},
        provider_limits={"succ": threading.Semaphore(1), "rais": threading.Semaphore(1)},
        cfg=cfg,
        parser_candidates=[],
    )
    assert best is not None
    assert best.provider == "succ"
    assert len(all_outcomes) == 2


def test_process_one_input_returns_none_when_all_fail():
    """All providers fail; best is None; all outcomes have failure_reason."""
    orch = _orchestrator()
    providers = {
        "bad1": _BadJsonProvider(),
        "bad2": _RaisingProvider(),
    }
    pcfgs = {n: _PcfgStub(n) for n in providers}
    scheduler = _StubScheduler(["bad1", "bad2"])
    cfg = _CfgStub()

    inp, best, all_outcomes = orch._process_one_input(
        "500 beer",
        providers_by_name=providers,
        pcfgs_by_name=pcfgs,
        scheduler=scheduler,
        priority_map={"bad1": 0, "bad2": 1},
        provider_limits={"bad1": threading.Semaphore(1), "bad2": threading.Semaphore(1)},
        cfg=cfg,
        parser_candidates=[],
    )
    assert best is None
    assert len(all_outcomes) == 2
    assert all(o.failure_reason is not None for o in all_outcomes)


def test_process_one_input_repair_succeeds():
    """First attempt fails; repair succeeds."""
    orch = _orchestrator()

    class FlakyProvider:
        """Fails on first two calls (initial attempts), succeeds on repair."""
        name = "flaky"
        model = "fake-model-flaky"
        def __init__(self):
            self.calls = 0
        def generate_label(self, text):
            self.calls += 1
            if self.calls <= 2:
                return "{not json"
            return '{"transactions":[{"amount":500,"currency":"INR","item":"beer","category":"Drinks","type":"expense"}]}'

    provider = FlakyProvider()
    providers = {"flaky": provider}
    pcfgs = {"flaky": _PcfgStub("flaky")}
    scheduler = _StubScheduler(["flaky", "flaky"])

    class _CfgRepair:
        class _Og:
            label_attempts_per_input = 2
            providers = []
            provider_priority = ()
        class _V:
            retry_invalid_with_stricter_prompt = True
            max_repair_attempts = 1
        output_generation = _Og()
        validation = _V()

    cfg = _CfgRepair()
    cfg.output_generation.providers = [_PcfgStub("flaky")]

    inp, best, all_outcomes = orch._process_one_input(
        "500 beer",
        providers_by_name=providers,
        pcfgs_by_name=pcfgs,
        scheduler=scheduler,
        priority_map={"flaky": 0},
        provider_limits={"flaky": threading.Semaphore(1)},
        cfg=cfg,
        parser_candidates=[],
    )
    assert best is not None
    assert best.is_repair_attempt is True
    assert len(all_outcomes) == 3   # 2 initial + 1 repair


def test_provider_error_records_call_failure_only():
    """The provider-error path records record_call(success=False) and does
    NOT call record_label_candidate. Keystone of the no-double-count invariant.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

    from datetime import datetime, timezone
    from metrics import MetricsRecorder
    from llm_providers import ProviderError

    # Spying recorder.
    class _SpyRecorder(MetricsRecorder):
        def __init__(self):
            super().__init__(
                prices={"default": {"input_per_million": 0.0, "output_per_million": 0.0}},
                started_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
                prices_source=None,
            )
            self.call_log = []
            self.candidate_log = []
        def record_call(self, **kw):
            self.call_log.append(kw)
            super().record_call(**kw)
        def record_label_candidate(self, **kw):
            self.candidate_log.append(kw)
            super().record_label_candidate(**kw)

    class _RaisingProvider:
        name = "boom"
        model = "m"
        def generate_label(self, _):
            raise ProviderError("simulated failure")

    class _Pcfg:
        name = "boom"
        provider_type = "fake"

    orch = _orchestrator()
    rec = _SpyRecorder()
    outcome = orch._try_one_attempt(
        "input text",
        provider=_RaisingProvider(),
        pcfg=_Pcfg(),
        recorder=rec,
        provider_type="fake",
    )
    assert outcome.failure_reason == "provider_error"
    assert len(rec.call_log) == 1
    assert rec.call_log[0]["success"] is False
    assert rec.call_log[0]["failure_reason"] == "provider_error"
    assert len(rec.candidate_log) == 0
