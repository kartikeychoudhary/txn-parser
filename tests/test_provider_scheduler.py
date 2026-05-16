import threading
from collections import Counter

import pytest

from generation_config import ProviderConfig
from generation_orchestrator import RoundRobinScheduler, allocate_quota, render_dry_run


def _provs(*pairs: tuple[str, int]) -> list[ProviderConfig]:
    return [ProviderConfig(name=n, provider_type="fake", weight=w) for n, w in pairs]


def test_allocate_quota_70_30_split():
    p = _provs(("deepseek", 70), ("gemini", 30))
    result = allocate_quota(100000, p)
    assert result == {"deepseek": 70000, "gemini": 30000}


def test_allocate_quota_60_40_small_total():
    p = _provs(("a", 60), ("b", 40))
    result = allocate_quota(10, p)
    assert result == {"a": 6, "b": 4}


def test_allocate_quota_tie_break_by_declaration_order():
    p = _provs(("a", 1), ("b", 1), ("c", 1))
    result = allocate_quota(10, p)
    # 10/3 = 3.33 each; floor gives 3 each (sum 9); leftover 1 by order -> a gets +1
    assert result == {"a": 4, "b": 3, "c": 3}
    assert sum(result.values()) == 10


def test_allocate_quota_zero_total():
    p = _provs(("a", 1), ("b", 1))
    assert allocate_quota(0, p) == {"a": 0, "b": 0}


def test_allocate_quota_single_provider_gets_all():
    p = _provs(("only", 1))
    assert allocate_quota(42, p) == {"only": 42}


def test_allocate_quota_empty_raises():
    with pytest.raises(ValueError):
        allocate_quota(10, [])


def test_allocate_quota_sums_to_total_random_weights():
    p = _provs(("a", 17), ("b", 41), ("c", 23), ("d", 5))
    for total in (1, 7, 100, 1234, 99999):
        result = allocate_quota(total, p)
        assert sum(result.values()) == total
        for v in result.values():
            assert v >= 0


def test_round_robin_cycle_length_normalized_by_gcd():
    p = _provs(("a", 60), ("b", 40))
    sched = RoundRobinScheduler(p)
    # gcd(60, 40) = 20; normalized to 3:2; cycle length 5.
    assert sched.cycle_length == 5


def test_round_robin_cycle_length_huge_weights_normalized():
    p = _provs(("a", 60000), ("b", 40000))
    sched = RoundRobinScheduler(p)
    assert sched.cycle_length == 5  # NOT 100000


def test_round_robin_distribution_over_full_cycles():
    p = _provs(("a", 3), ("b", 2))  # cycle length 5
    sched = RoundRobinScheduler(p)
    # 100 cycles = 500 calls; expect exactly 300 a's, 200 b's.
    counts = Counter(sched.next_provider() for _ in range(500))
    assert counts == {"a": 300, "b": 200}


def test_round_robin_distribution_over_full_cycles_50_30_20():
    p = _provs(("a", 50), ("b", 30), ("c", 20))  # gcd 10 -> 5:3:2 -> cycle 10
    sched = RoundRobinScheduler(p)
    assert sched.cycle_length == 10
    # 100 cycles = 1000 calls -> exactly 500/300/200
    counts = Counter(sched.next_provider() for _ in range(1000))
    assert counts == {"a": 500, "b": 300, "c": 200}


def test_round_robin_no_long_runs():
    """Interleaving should avoid blocky runs. Max run length should be small."""
    p = _provs(("a", 3), ("b", 2))
    sched = RoundRobinScheduler(p)
    seq = [sched.next_provider() for _ in range(20)]
    # Find longest consecutive run of same provider.
    max_run = 1
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    assert max_run <= 2, f"too-blocky sequence: {seq}"


def test_round_robin_reset():
    p = _provs(("a", 1), ("b", 1))
    sched = RoundRobinScheduler(p)
    first = [sched.next_provider() for _ in range(4)]
    sched.reset()
    second = [sched.next_provider() for _ in range(4)]
    assert first == second


def test_round_robin_thread_safe_no_lost_calls():
    """8 threads × 250 calls each = 2000 calls. Cycle 5 (3:2). Expect 1200 a, 800 b."""
    p = _provs(("a", 3), ("b", 2))
    sched = RoundRobinScheduler(p)
    counts = Counter()
    counts_lock = threading.Lock()

    def worker():
        local = []
        for _ in range(250):
            local.append(sched.next_provider())
        with counts_lock:
            counts.update(local)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counts == {"a": 1200, "b": 800}


def _make_minimal_cfg(tmp_path):
    """Build a minimal GenerationConfig for dry-run tests."""
    from generation_config import (
        GenerationConfig, InputGenerationConfig, OutputGenerationConfig,
        ValidationGateConfig, RateLimitsConfig,
    )
    p_input = ProviderConfig(
        name="fake_a", provider_type="fake", weight=60, threads=1,
        fixture_inputs="/tmp/fake.jsonl",
    )
    p_input2 = ProviderConfig(
        name="fake_b", provider_type="fake", weight=40, threads=1,
        fixture_inputs="/tmp/fake.jsonl",
    )
    p_output = ProviderConfig(
        name="fake_a", provider_type="fake", weight=50, threads=1,
        fixture_labels="/tmp/fake.jsonl",
    )
    p_output2 = ProviderConfig(
        name="fake_b", provider_type="fake", weight=50, threads=1,
        fixture_labels="/tmp/fake.jsonl",
    )
    return GenerationConfig(
        version=1,
        input_generation=InputGenerationConfig(
            enabled=True, target_inputs=10, batch_size=5, dedupe=True,
            providers=(p_input, p_input2),
        ),
        output_generation=OutputGenerationConfig(
            enabled=True, label_attempts_per_input=1,
            selection_policy="first_valid_then_score",
            providers=(p_output, p_output2),
        ),
        validation=ValidationGateConfig(),
        rate_limits=RateLimitsConfig(global_max_workers=4, write_flush_every=10),
        source_path=str(tmp_path / "providers.json"),
    )


def test_render_dry_run_contains_key_sections(tmp_path):
    cfg = _make_minimal_cfg(tmp_path)
    output = render_dry_run(cfg)
    assert "Multi-provider dry run" in output
    assert "Config:" in output
    assert "(version 1)" in output
    assert "Input generation: enabled" in output
    assert "target_inputs: 10" in output
    assert "Output generation: enabled" in output
    assert "selection_policy: first_valid_then_score" in output
    assert "scheduler preview (first 10):" in output
    assert "Validation gate:" in output
    assert "Execution limits:" in output
    assert "Execution: not started" in output


def test_render_dry_run_shows_quota_for_each_input_provider(tmp_path):
    cfg = _make_minimal_cfg(tmp_path)
    output = render_dry_run(cfg)
    # target=10, weights 60/40 -> 6/4
    assert "quota=6" in output
    assert "quota=4" in output


def test_render_dry_run_disabled_input_phase(tmp_path):
    from generation_config import InputGenerationConfig
    cfg = _make_minimal_cfg(tmp_path)
    cfg = type(cfg)(
        version=cfg.version,
        input_generation=InputGenerationConfig(enabled=False),
        output_generation=cfg.output_generation,
        validation=cfg.validation,
        rate_limits=cfg.rate_limits,
        source_path=cfg.source_path,
    )
    output = render_dry_run(cfg)
    assert "Input generation: disabled" in output
    assert "quota=" not in output  # no quota block at all
    assert "Output generation: enabled" in output


def test_render_dry_run_scheduler_preview_has_10_names(tmp_path):
    cfg = _make_minimal_cfg(tmp_path)
    output = render_dry_run(cfg)
    # Find the "scheduler preview (first 10):" line and verify 10 names follow.
    for line in output.splitlines():
        if "scheduler preview (first 10):" in line:
            after = line.split(":", 1)[1].strip()
            names = after.split()
            assert len(names) == 10
            assert all(n in {"fake_a", "fake_b"} for n in names)
            return
    raise AssertionError("scheduler preview line not found")


def test_render_dry_run_disabled_output_phase(tmp_path):
    from generation_config import OutputGenerationConfig
    cfg = _make_minimal_cfg(tmp_path)
    cfg = type(cfg)(
        version=cfg.version,
        input_generation=cfg.input_generation,
        output_generation=OutputGenerationConfig(enabled=False),
        validation=cfg.validation,
        rate_limits=cfg.rate_limits,
        source_path=cfg.source_path,
    )
    output = render_dry_run(cfg)
    assert "Output generation: disabled" in output
    assert "scheduler preview" not in output  # no scheduler when disabled
    assert "Input generation: enabled" in output
