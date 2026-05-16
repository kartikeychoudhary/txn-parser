"""Multi-provider orchestration: quota allocation + weighted scheduler + dry-run rendering.

Pure stdlib. No I/O, no provider calls, no threading execution.
"""
from __future__ import annotations

import math
import threading
from typing import Sequence

from generation_config import GenerationConfig, ProviderConfig


def allocate_quota(
    total: int,
    providers: Sequence[ProviderConfig],
) -> dict[str, int]:
    """Allocate `total` across providers by weight (largest-remainder method).

    Guarantees:
      - sum(result.values()) == total
      - all values >= 0
      - deterministic given fixed provider order and weights

    Raises ValueError if providers is empty.
    """
    if not providers:
        raise ValueError("allocate_quota: providers must be non-empty")

    total_weight = sum(p.weight for p in providers)
    if total_weight <= 0:
        raise ValueError("allocate_quota: sum of weights must be positive")

    # Largest-remainder method.
    raw = [(p.name, total * p.weight / total_weight) for p in providers]
    floors = {name: int(value) for name, value in raw}
    leftover = total - sum(floors.values())
    # Remainders, with original index to break ties by declaration order.
    remainders = sorted(
        ((value - int(value), idx, name) for idx, (name, value) in enumerate(raw)),
        key=lambda x: (-x[0], x[1]),  # largest fractional remainder first; then by idx
    )
    for i in range(leftover):
        _, _, name = remainders[i]
        floors[name] += 1
    return floors


class RoundRobinScheduler:
    """Deterministic weighted round-robin. Thread-safe.

    Weights are normalized by GCD before building the cycle, so [60000, 40000]
    behaves like [3, 2]. Bresenham-style interleaving keeps no long runs.

    Guarantees:
      - Over every full cycle, counts exactly match normalized weights.
      - Over partial cycles, counts deviate by at most one slot per provider.
      - next_provider() is protected by threading.Lock — concurrent callers
        receive serialized consecutive scheduler positions; cursor is never corrupted.
      - State is in-memory only; resets between process runs.
    """

    def __init__(self, providers: Sequence[ProviderConfig]) -> None:
        if not providers:
            raise ValueError("RoundRobinScheduler: providers must be non-empty")
        # Normalize by GCD.
        weights = [p.weight for p in providers]
        g = weights[0]
        for w in weights[1:]:
            g = math.gcd(g, w)
        normalized = [(p.name, w // g) for p, w in zip(providers, weights)]
        self._cycle: list[str] = _bresenham_interleave(normalized)
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def cycle_length(self) -> int:
        return len(self._cycle)

    def next_provider(self) -> str:
        with self._lock:
            name = self._cycle[self._cursor % len(self._cycle)]
            self._cursor += 1
            return name

    def reset(self) -> None:
        with self._lock:
            self._cursor = 0


def _bresenham_interleave(weighted: list[tuple[str, int]]) -> list[str]:
    """Build a length-sum(weights) sequence that interleaves names by weight.

    Deterministic weighted interleaver. At each slot, pick the provider
    whose (cumulative-credit + 1) / weight ratio is smallest, breaking
    ties by declaration order. Produces a sequence with no long single-
    provider runs at high weight ratios. Name historical: it is informed
    by Bresenham's line drawing but not an exact port; tests pin behavior
    by distribution and max-run-length, not by the exact algorithm.
    """
    total = sum(w for _, w in weighted)
    credits = {name: 0 for name, _ in weighted}
    weights = dict(weighted)
    order = [name for name, _ in weighted]
    out: list[str] = []
    for _ in range(total):
        # Pick the provider with the most "deserved" next slot.
        # Score = (credits + 1) / weight; smaller score = more deserving.
        best_name = order[0]
        best_score = (credits[best_name] + 1) / weights[best_name]
        for name in order[1:]:
            score = (credits[name] + 1) / weights[name]
            if score < best_score:
                best_score = score
                best_name = name
        out.append(best_name)
        credits[best_name] += 1
    return out


def render_dry_run(cfg: GenerationConfig) -> str:
    """Human-readable summary of what the config would do.
    Pinned format — see spec §5.3. No execution, no side effects."""
    lines: list[str] = []
    lines.append("Multi-provider dry run")
    src = cfg.source_path or "<unknown>"
    lines.append(f"Config: {src}  (version {cfg.version})")
    lines.append("")

    # Input generation.
    if not cfg.input_generation.enabled:
        lines.append("Input generation: disabled")
    else:
        ig = cfg.input_generation
        lines.append("Input generation: enabled")
        lines.append(f"  target_inputs: {ig.target_inputs}")
        lines.append(f"  batch_size: {ig.batch_size}")
        lines.append(f"  dedupe: {ig.dedupe}")
        lines.append("  providers:")
        quotas = allocate_quota(ig.target_inputs, ig.providers) if ig.providers else {}
        for p in ig.providers:
            lines.append(
                f"    {p.name}  type={p.provider_type}  weight={p.weight}  "
                f"threads={p.threads}  quota={quotas.get(p.name, 0)}"
            )
    lines.append("")

    # Output generation.
    if not cfg.output_generation.enabled:
        lines.append("Output generation: disabled")
    else:
        og = cfg.output_generation
        lines.append("Output generation: enabled")
        lines.append(f"  label_attempts_per_input: {og.label_attempts_per_input}")
        lines.append(f"  selection_policy: {og.selection_policy}")
        priority = " ".join(og.provider_priority) if og.provider_priority else "(default declaration order)"
        lines.append(f"  provider_priority: {priority}")
        lines.append("  providers:")
        for p in og.providers:
            lines.append(
                f"    {p.name}  type={p.provider_type}  weight={p.weight}  threads={p.threads}"
            )
        if og.providers:
            sched = RoundRobinScheduler(og.providers)
            preview = " ".join(sched.next_provider() for _ in range(10))
            lines.append(f"  scheduler preview (first 10): {preview}")
    lines.append("")

    # Validation gate.
    v = cfg.validation
    lines.append("Validation gate:")
    lines.append(f"  schema: {v.schema}")
    lines.append(f"  semantic_validator: {v.semantic_validator}")
    lines.append(f"  reject_invalid: {v.reject_invalid}")
    lines.append(f"  retry_invalid_with_stricter_prompt: {v.retry_invalid_with_stricter_prompt}")
    lines.append(f"  max_repair_attempts: {v.max_repair_attempts}")
    lines.append("")

    # Execution limits.
    rl = cfg.rate_limits
    lines.append("Execution limits:")
    lines.append(f"  global_max_workers: {rl.global_max_workers}")
    lines.append(f"  write_flush_every: {rl.write_flush_every}")
    lines.append("")
    lines.append("Execution: not started. This was a dry run.")
    return "\n".join(lines)
