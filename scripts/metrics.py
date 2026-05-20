"""Per-run metrics for multi-provider Stage 5 runs.

Pure stdlib. Importing this module triggers no SDK imports, no I/O.

Public surface:
    - MetricsConfigError
    - ProviderCallMetric, CandidateMetric (dataclasses)
    - load_prices(path) -> dict
    - lookup_price(prices, provider_type, model) -> (input_per_million, output_per_million)
    - estimate_tokens(text) -> int
    - percentiles(values) -> {"p50": ..., "p95": ...}
    - MetricsRecorder (added in Task 2)

Spec: docs/superpowers/specs/2026-05-17-multi-provider-generation-slice4-design.md
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


logger = logging.getLogger("metrics")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MetricsConfigError(ValueError):
    """Raised when prices.json fails schema or numeric checks."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProviderCallMetric:
    provider: str
    provider_type: str
    model: str | None
    phase: Literal["inputs", "label"]
    attempt_type: Literal["input_batch", "initial", "repair"]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_tokens: bool
    success: bool
    failure_reason: str | None


@dataclass
class CandidateMetric:
    provider: str
    model: str | None
    phase: Literal["label"]
    score: int
    accepted: bool
    failure_reason: str | None
    is_repair_attempt: bool


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


_REQUIRED_PRICE_FIELDS = {"input_per_million", "output_per_million"}


def load_prices(path: Path) -> dict:
    """Load configs/prices.json. See spec §6 for full behavior.

    Returns a dict mapping lookup-key -> {"input_per_million": float,
                                          "output_per_million": float}.
    """
    if not path.exists():
        logger.warning(
            "Prices file not found at %s; cost reports will read $0.00", path
        )
        return {"default": {"input_per_million": 0.0, "output_per_million": 0.0}}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MetricsConfigError(f"prices.json is not valid JSON: {e}") from e

    if not isinstance(raw, dict):
        raise MetricsConfigError("prices.json top level must be an object")

    out: dict = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            raise MetricsConfigError(
                f"prices.json[{key!r}] must be an object, got {type(entry).__name__}"
            )
        # Validate required numeric fields.
        cleaned = {}
        for field in _REQUIRED_PRICE_FIELDS:
            if field not in entry:
                raise MetricsConfigError(
                    f"prices.json[{key!r}] missing required field {field!r}"
                )
            val = entry[field]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise MetricsConfigError(
                    f"prices.json[{key!r}].{field} must be a number, got {val!r}"
                )
            if val < 0:
                raise MetricsConfigError(
                    f"prices.json[{key!r}].{field} must be >= 0, got {val}"
                )
            cleaned[field] = float(val)
        # Unknown extra fields are dropped with a warning.
        unknown = set(entry) - _REQUIRED_PRICE_FIELDS
        for u in sorted(unknown):
            logger.warning(
                "prices.json[%r] has unknown field %r; ignoring.", key, u
            )
        out[key] = cleaned
    return out


def lookup_price(
    prices: dict, provider_type: str, model: str | None
) -> tuple[float, float]:
    """Lookup order: 'provider_type:model' -> model -> provider_type -> 'default'.

    Returns (input_per_million, output_per_million); (0.0, 0.0) if nothing matches.
    """
    keys = []
    if model:
        keys.append(f"{provider_type}:{model}")
        keys.append(model)
    keys.append(provider_type)
    keys.append("default")
    for k in keys:
        if k in prices:
            entry = prices[k]
            return (
                float(entry["input_per_million"]),
                float(entry["output_per_million"]),
            )
    return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Token estimation (fallback when SDK usage is unavailable)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str | None) -> int:
    """Cheap char/4 estimate. Returns 0 for None or empty string.

    Used only when a provider does not report SDK usage metadata.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


# ---------------------------------------------------------------------------
# Percentile math
# ---------------------------------------------------------------------------


def percentiles(values: list[float]) -> dict[str, float]:
    """Returns {"p50": ..., "p95": ...}.

    Empty list   -> {"p50": 0.0, "p95": 0.0}.
    n < 20       -> p95 = max(values); p50 = median.
    n >= 20      -> statistics.quantiles(n=20); take index 9 (p50) and 18 (p95).
    """
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    if len(values) == 1:
        return {"p50": float(values[0]), "p95": float(values[0])}
    if len(values) < 20:
        return {
            "p50": float(statistics.median(values)),
            "p95": float(max(values)),
        }
    qs = statistics.quantiles(values, n=20)
    # qs has 19 cut points: index 0 = 5th pct, index 9 = 50th, index 18 = 95th.
    return {"p50": float(qs[9]), "p95": float(qs[18])}


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

import subprocess
import threading
from collections import Counter
from datetime import datetime, timezone


def _git_short_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:    # noqa: BLE001 — git unavailable; non-fatal
        pass
    return None


def _round4(x: float) -> float:
    return round(float(x), 4)


class MetricsRecorder:
    """Per-run accumulator. Methods are thread-safe (single internal Lock).

    finalize() must run single-threaded after all workers have joined.
    """

    def __init__(
        self,
        *,
        prices: dict,
        started_at: datetime,
        prices_source: str | None,
    ) -> None:
        self._calls: list[ProviderCallMetric] = []
        self._candidates: list[CandidateMetric] = []
        self._input_rows = 0
        self._train_rows = 0
        self._failed_rows = 0
        self._repair_exhausted = 0
        self._lock = threading.Lock()
        self._prices = prices
        self._prices_source = prices_source
        self._started_at = started_at

    # -- recording --

    def record_call(
        self,
        *,
        provider: str,
        provider_type: str,
        model: str | None,
        phase: str,
        attempt_type: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_tokens: bool,
        success: bool,
        failure_reason: str | None,
    ) -> None:
        m = ProviderCallMetric(
            provider=provider,
            provider_type=provider_type,
            model=model,
            phase=phase,
            attempt_type=attempt_type,
            latency_ms=float(latency_ms),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens) + int(completion_tokens),
            estimated_tokens=bool(estimated_tokens),
            success=bool(success),
            failure_reason=failure_reason,
        )
        with self._lock:
            self._calls.append(m)

    def record_label_candidate(
        self,
        *,
        provider: str,
        model: str | None,
        score: int,
        accepted: bool,
        failure_reason: str | None,
        is_repair_attempt: bool,
    ) -> None:
        m = CandidateMetric(
            provider=provider,
            model=model,
            phase="label",
            score=int(score),
            accepted=bool(accepted),
            failure_reason=failure_reason,
            is_repair_attempt=bool(is_repair_attempt),
        )
        with self._lock:
            self._candidates.append(m)

    def record_output_row(self, kind: Literal["input", "train", "failed"]) -> None:
        with self._lock:
            if kind == "input":
                self._input_rows += 1
            elif kind == "train":
                self._train_rows += 1
            elif kind == "failed":
                self._failed_rows += 1
            else:
                raise ValueError(f"unknown output row kind: {kind!r}")

    def record_repair_exhausted(self) -> None:
        """Called once per input whose repair loop finished without accepting
        any candidate. Distinct from record_output_row('failed') because a
        failed row may or may not have entered repair."""
        with self._lock:
            self._repair_exhausted += 1

    # -- finalize --

    def finalize(
        self,
        *,
        phase: str,
        inputs_processed: int,
        finished_at: datetime | None = None,
    ) -> dict:
        finished_at = finished_at or datetime.now(timezone.utc)
        duration_ms = (finished_at - self._started_at).total_seconds() * 1000.0
        duration_ms = max(0.0, duration_ms)

        # Per-provider aggregation.
        provider_keys: list[str] = []
        seen_provider = set()
        for c in self._calls:
            if c.provider not in seen_provider:
                seen_provider.add(c.provider)
                provider_keys.append(c.provider)

        providers_out: dict = {}
        for pname in provider_keys:
            pcalls = [c for c in self._calls if c.provider == pname]
            # provider_type/model from the first call (all calls for one provider share these)
            ptype = pcalls[0].provider_type
            pmodel = pcalls[0].model
            success_calls = sum(1 for c in pcalls if c.success)
            failed_calls = sum(1 for c in pcalls if not c.success)
            prompt_tokens = sum(c.prompt_tokens for c in pcalls)
            completion_tokens = sum(c.completion_tokens for c in pcalls)
            has_est = any(c.estimated_tokens for c in pcalls)
            input_pm, output_pm = lookup_price(self._prices, ptype, pmodel)
            cost = (prompt_tokens / 1_000_000.0) * input_pm + \
                   (completion_tokens / 1_000_000.0) * output_pm
            lat = percentiles([c.latency_ms for c in pcalls])
            call_failure_rate = (failed_calls / len(pcalls)) if pcalls else 0.0

            block: dict = {
                "provider_type": ptype,
                "model": pmodel,
                "calls": len(pcalls),
                "success_calls": success_calls,
                "failed_calls": failed_calls,
                "call_failure_rate": _round4(call_failure_rate),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "has_estimated_tokens": has_est,
                "estimated_cost_usd": _round4(cost),
                "latency_ms": {
                    "p50": _round4(lat["p50"]),
                    "p95": _round4(lat["p95"]),
                },
            }

            if phase == "label":
                pcands = [c for c in self._candidates if c.provider == pname]
                accepted = sum(1 for c in pcands if c.accepted)
                rejected = sum(1 for c in pcands if not c.accepted)
                total_cands = accepted + rejected
                block["candidates_accepted"] = accepted
                block["candidates_rejected"] = rejected
                block["acceptance_rate"] = _round4(accepted / total_cands) \
                    if total_cands else 0.0

            providers_out[pname] = block

        # Totals.
        total_calls = len(self._calls)
        calls_succeeded = sum(1 for c in self._calls if c.success)
        calls_failed = total_calls - calls_succeeded
        total_cost = sum(p["estimated_cost_usd"] for p in providers_out.values())
        totals: dict = {
            "calls": total_calls,
            "calls_succeeded": calls_succeeded,
            "calls_failed": calls_failed,
            "candidates_accepted": sum(1 for c in self._candidates if c.accepted),
            "candidates_rejected": sum(1 for c in self._candidates if not c.accepted),
            "estimated_cost_usd": _round4(total_cost),
            "input_rows_written": self._input_rows,
            "train_rows_written": self._train_rows,
            "failed_rows_written": self._failed_rows,
            "has_estimated_tokens": any(c.estimated_tokens for c in self._calls),
        }

        # Failures: terminal failure_reason from calls + candidates; only non-null.
        failures: Counter = Counter()
        for c in self._calls:
            if c.failure_reason:
                failures[c.failure_reason] += 1
        for c in self._candidates:
            if c.failure_reason:
                failures[c.failure_reason] += 1

        # Throughput.
        duration_s = duration_ms / 1000.0
        throughput = {
            "inputs_per_sec": _round4(inputs_processed / duration_s) if duration_s > 0 else 0.0,
            "calls_per_sec": _round4(total_calls / duration_s) if duration_s > 0 else 0.0,
        }

        out: dict = {
            "run": {
                "started_at": self._started_at.isoformat().replace("+00:00", "Z"),
                "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
                "duration_ms": int(duration_ms),
                "phase": phase,
                "inputs_processed": int(inputs_processed),
                "git_sha": _git_short_sha(),
                "prices_source": self._prices_source,
                "throughput": throughput,
            },
            "totals": totals,
            "providers": providers_out,
            "failures": dict(failures),
        }

        # Repair block only for label phase.
        if phase == "label":
            repair_attempts = sum(
                1 for c in self._calls if c.attempt_type == "repair"
            )
            repair_accepted = sum(
                1 for c in self._candidates
                if c.is_repair_attempt and c.accepted
            )
            out["repair"] = {
                "attempts": repair_attempts,
                "accepted": repair_accepted,
                "exhausted": self._repair_exhausted,
            }

        return out

    # -- stdout summary --

    def render_stdout(self, summary: dict) -> str:
        run = summary["run"]
        totals = summary["totals"]
        providers = summary["providers"]
        phase = run["phase"]
        duration_ms = run["duration_ms"]
        mins, secs = divmod(duration_ms // 1000, 60)
        dur = f"{mins}m{secs:02d}s" if mins else f"{secs}s"

        lines = [
            f"=== Stage 5 metrics (phase={phase}, "
            f"{run['inputs_processed']} inputs, {dur}) ===",
            f"Calls: {totals['calls']} ({totals['calls_succeeded']} ok, "
            f"{totals['calls_failed']} failed)"
            + (f"  Accepted: {totals['candidates_accepted']}  "
               f"Failed rows: {totals['failed_rows_written']}"
               if phase == "label" else
               f"  Input rows: {totals['input_rows_written']}"),
        ]
        prices_source = run.get("prices_source")
        cost_label = (
            f"rates from {prices_source}" if prices_source
            else "rates from default zero pricing"
        )
        lines.append(
            f"Estimated cost: ${totals['estimated_cost_usd']:.4f} ({cost_label})"
        )
        lines.append("")

        # Per-provider table.
        if phase == "label":
            header = ["Provider", "Calls", "Ok", "Fail%", "Acc", "Cost", "p50", "p95", "Top failure"]
        else:
            header = ["Provider", "Calls", "Ok", "Fail%", "Cost", "p50", "p95"]

        rows = [header]
        for pname, pblock in providers.items():
            if phase == "label":
                top_fail = self._top_failure_for(pname)
                rows.append([
                    pname,
                    str(pblock["calls"]),
                    str(pblock["success_calls"]),
                    f"{pblock['call_failure_rate']*100:.2f}%",
                    str(pblock.get("candidates_accepted", 0)),
                    f"${pblock['estimated_cost_usd']:.4f}",
                    f"{int(pblock['latency_ms']['p50'])}",
                    f"{int(pblock['latency_ms']['p95'])}",
                    top_fail,
                ])
            else:
                rows.append([
                    pname,
                    str(pblock["calls"]),
                    str(pblock["success_calls"]),
                    f"{pblock['call_failure_rate']*100:.2f}%",
                    f"${pblock['estimated_cost_usd']:.4f}",
                    f"{int(pblock['latency_ms']['p50'])}",
                    f"{int(pblock['latency_ms']['p95'])}",
                ])

        widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
        for row in rows:
            lines.append("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))

        if phase == "label" and "repair" in summary:
            rep = summary["repair"]
            lines.append("")
            lines.append(
                f"Repair: {rep['attempts']} attempted, "
                f"{rep['accepted']} accepted, {rep['exhausted']} exhausted"
            )

        return "\n".join(lines)

    def _top_failure_for(self, provider: str) -> str:
        c: Counter = Counter()
        for call in self._calls:
            if call.provider == provider and call.failure_reason:
                c[call.failure_reason] += 1
        for cand in self._candidates:
            if cand.provider == provider and cand.failure_reason:
                c[cand.failure_reason] += 1
        if not c:
            return ""
        reason, n = c.most_common(1)[0]
        return f"{reason} ({n})"
