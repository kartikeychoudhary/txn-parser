"""Multi-provider generation config — dataclasses + JSON loader.

Stdlib only. Loader raises ConfigError with field-path messages.
Validation rules (§4.3) land in Task 5; this file establishes the
parsing surface and strict-type helpers.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_VALID_PROVIDER_TYPES = {"fake", "deepseek", "gemini", "local_teacher", "bedrock"}
_VALID_SELECTION_POLICIES = {"first_valid_then_score"}
_KNOWN_PROVIDER_FIELDS = {
    "name", "type", "weight", "threads", "model", "temperature", "max_tokens",
    "max_retries", "structured_output", "seed", "fixture_inputs", "fixture_labels",
    "region", "thinking_budget_tokens", "cache_system_prompt",
}
_KNOWN_TOP_LEVEL = {
    "version", "input_generation", "output_generation", "validation", "rate_limits",
}


class ConfigError(ValueError):
    """Raised with a field path in the message (e.g. 'input_generation.providers[1].weight')."""


# ---- Strict type helpers ---------------------------------------------------
# These reject wrong types instead of coercing. bool("false") == True and
# int("10") == 10 are footguns we deliberately avoid.

def _require_bool(value, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path}: must be a boolean (got {type(value).__name__})")
    return value


def _require_int(value, path: str, *, min_value: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{path}: must be an integer (got {type(value).__name__})")
    if min_value is not None and value < min_value:
        raise ConfigError(f"{path}: must be >= {min_value}")
    return value


def _require_str(value, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path}: must be a string (got {type(value).__name__})")
    return value


def _optional_int(value, path: str, *, min_value: int | None = None) -> int | None:
    if value is None:
        return None
    return _require_int(value, path, min_value=min_value)


def _optional_float(value, path: str, *, min_value: float | None = None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{path}: must be a number or null (got {type(value).__name__})")
    result = float(value)
    if min_value is not None and result < min_value:
        raise ConfigError(f"{path}: must be >= {min_value}")
    return result


def _optional_str(value, path: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, path)


# ---- Dataclasses -----------------------------------------------------------

@dataclass(frozen=True)
class ProviderConfig:
    name: str
    provider_type: str
    weight: int = 1
    threads: int = 1
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int = 3
    structured_output: bool = False
    seed: int | None = None
    fixture_inputs: str | None = None
    fixture_labels: str | None = None
    region: str | None = None
    thinking_budget_tokens: int | None = None
    cache_system_prompt: bool = False


@dataclass(frozen=True)
class InputGenerationConfig:
    enabled: bool = True
    target_inputs: int = 0
    batch_size: int = 100
    dedupe: bool = True
    providers: tuple[ProviderConfig, ...] = ()


@dataclass(frozen=True)
class OutputGenerationConfig:
    enabled: bool = True
    label_attempts_per_input: int = 1
    selection_policy: str = "first_valid_then_score"
    providers: tuple[ProviderConfig, ...] = ()
    provider_priority: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationGateConfig:
    schema: bool = True
    semantic_validator: bool = True
    reject_invalid: bool = True
    retry_invalid_with_stricter_prompt: bool = False
    max_repair_attempts: int = 0


@dataclass(frozen=True)
class RateLimitsConfig:
    global_max_workers: int = 8
    write_flush_every: int = 50


@dataclass(frozen=True)
class GenerationConfig:
    version: int
    input_generation: InputGenerationConfig
    output_generation: OutputGenerationConfig
    validation: ValidationGateConfig
    rate_limits: RateLimitsConfig
    source_path: str | None = None


# ---- Loader ----------------------------------------------------------------

def load_generation_config(path: Path | str) -> GenerationConfig:
    """Parse JSON config into typed dataclasses. See spec §4."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: invalid JSON: {e}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level must be an object")

    version = raw.get("version", 1)
    if not isinstance(version, int) or version != 1:
        raise ConfigError(
            f"version: unsupported config version {version!r}, expected 1"
        )

    for key in raw.keys():
        if key not in _KNOWN_TOP_LEVEL:
            logger.warning("Unknown top-level key in %s: %r", path, key)

    config_dir = path.parent

    input_gen = _parse_input_generation(raw.get("input_generation", {}), config_dir)
    output_gen = _parse_output_generation(raw.get("output_generation", {}), config_dir)
    validation = _parse_validation_gate(raw.get("validation", {}))
    rate_limits = _parse_rate_limits(raw.get("rate_limits", {}))

    cfg = GenerationConfig(
        version=version,
        input_generation=input_gen,
        output_generation=output_gen,
        validation=validation,
        rate_limits=rate_limits,
        source_path=str(path),
    )
    _validate(cfg)
    return cfg


def _parse_input_generation(d: Any, config_dir: Path) -> InputGenerationConfig:
    if not isinstance(d, dict):
        raise ConfigError("input_generation: must be an object")
    enabled = _require_bool(d.get("enabled", True), "input_generation.enabled")
    target_inputs = _require_int(d.get("target_inputs", 0), "input_generation.target_inputs", min_value=0)
    batch_size = _require_int(d.get("batch_size", 100), "input_generation.batch_size", min_value=1)
    dedupe = _require_bool(d.get("dedupe", True), "input_generation.dedupe")
    providers_raw = d.get("providers", [])
    providers = _parse_providers(
        providers_raw, "input_generation", config_dir, used_in_phase="input",
    )
    return InputGenerationConfig(
        enabled=enabled,
        target_inputs=target_inputs,
        batch_size=batch_size,
        dedupe=dedupe,
        providers=tuple(providers),
    )


def _parse_output_generation(d: Any, config_dir: Path) -> OutputGenerationConfig:
    if not isinstance(d, dict):
        raise ConfigError("output_generation: must be an object")
    enabled = _require_bool(d.get("enabled", True), "output_generation.enabled")
    label_attempts = _require_int(
        d.get("label_attempts_per_input", 1),
        "output_generation.label_attempts_per_input",
        min_value=1,
    )
    selection_policy = _require_str(
        d.get("selection_policy", "first_valid_then_score"),
        "output_generation.selection_policy",
    )
    providers_raw = d.get("providers", [])
    providers = _parse_providers(
        providers_raw, "output_generation", config_dir, used_in_phase="output",
    )
    priority_raw = d.get("provider_priority", [])
    if not isinstance(priority_raw, list) or not all(isinstance(x, str) for x in priority_raw):
        raise ConfigError(
            "output_generation.provider_priority: must be a list of strings"
        )
    return OutputGenerationConfig(
        enabled=enabled,
        label_attempts_per_input=label_attempts,
        selection_policy=selection_policy,
        providers=tuple(providers),
        provider_priority=tuple(priority_raw),
    )


def _parse_validation_gate(d: Any) -> ValidationGateConfig:
    if not isinstance(d, dict):
        raise ConfigError("validation: must be an object")
    return ValidationGateConfig(
        schema=_require_bool(d.get("schema", True), "validation.schema"),
        semantic_validator=_require_bool(
            d.get("semantic_validator", True), "validation.semantic_validator",
        ),
        reject_invalid=_require_bool(d.get("reject_invalid", True), "validation.reject_invalid"),
        retry_invalid_with_stricter_prompt=_require_bool(
            d.get("retry_invalid_with_stricter_prompt", False),
            "validation.retry_invalid_with_stricter_prompt",
        ),
        max_repair_attempts=_require_int(
            d.get("max_repair_attempts", 0),
            "validation.max_repair_attempts",
            min_value=0,
        ),
    )


def _parse_rate_limits(d: Any) -> RateLimitsConfig:
    if not isinstance(d, dict):
        raise ConfigError("rate_limits: must be an object")
    return RateLimitsConfig(
        global_max_workers=_require_int(
            d.get("global_max_workers", 8),
            "rate_limits.global_max_workers",
            min_value=1,
        ),
        write_flush_every=_require_int(
            d.get("write_flush_every", 50),
            "rate_limits.write_flush_every",
            min_value=1,
        ),
    )


def _parse_providers(
    raw: Any, phase_path: str, config_dir: Path, *, used_in_phase: str,
) -> list[ProviderConfig]:
    if not isinstance(raw, list):
        raise ConfigError(f"{phase_path}.providers: must be a list")
    out: list[ProviderConfig] = []
    for i, item in enumerate(raw):
        prefix = f"{phase_path}.providers[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{prefix}: must be an object")
        for key in item.keys():
            if key not in _KNOWN_PROVIDER_FIELDS:
                logger.warning("Unknown provider field at %s: %r", prefix, key)
        # JSON key 'type' -> Python attribute 'provider_type'
        ptype = item.get("type")
        if not isinstance(ptype, str):
            raise ConfigError(f"{prefix}.type: missing or not a string")
        fixture_inputs = item.get("fixture_inputs")
        if fixture_inputs is not None:
            fixture_inputs = _resolve_fixture(
                fixture_inputs, config_dir, f"{prefix}.fixture_inputs",
            )
        fixture_labels = item.get("fixture_labels")
        if fixture_labels is not None:
            fixture_labels = _resolve_fixture(
                fixture_labels, config_dir, f"{prefix}.fixture_labels",
            )
        out.append(ProviderConfig(
            name=_require_str(item.get("name", ""), f"{prefix}.name"),
            provider_type=ptype,
            weight=_require_int(item.get("weight", 1), f"{prefix}.weight", min_value=1),
            threads=_require_int(item.get("threads", 1), f"{prefix}.threads", min_value=1),
            model=_optional_str(item.get("model"), f"{prefix}.model"),
            temperature=_optional_float(item.get("temperature"), f"{prefix}.temperature", min_value=0),
            max_tokens=_optional_int(item.get("max_tokens"), f"{prefix}.max_tokens", min_value=1),
            max_retries=_require_int(item.get("max_retries", 3), f"{prefix}.max_retries", min_value=0),
            structured_output=_require_bool(
                item.get("structured_output", False), f"{prefix}.structured_output",
            ),
            seed=_optional_int(item.get("seed"), f"{prefix}.seed"),
            fixture_inputs=fixture_inputs,
            fixture_labels=fixture_labels,
            region=_optional_str(item.get("region"), f"{prefix}.region"),
            thinking_budget_tokens=_optional_int(
                item.get("thinking_budget_tokens"),
                f"{prefix}.thinking_budget_tokens",
                min_value=1,
            ),
            cache_system_prompt=_require_bool(
                item.get("cache_system_prompt", False),
                f"{prefix}.cache_system_prompt",
            ),
        ))
    return out


def _resolve_fixture(value: Any, config_dir: Path, field_path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field_path}: must be a string path")
    p = Path(value)
    if not p.is_absolute():
        p = (config_dir / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise ConfigError(f"{field_path}: fixture not found at {p}")
    return str(p)


def _validate(cfg: GenerationConfig) -> None:
    """Apply cross-field rules. Min-value and type checks are already enforced
    by the strict helpers during parsing; this function covers the rules that
    span multiple fields (enabled+providers, selection policy, priority refs)."""
    ig = cfg.input_generation
    if ig.enabled and ig.target_inputs >= 1 and not ig.providers:
        raise ConfigError(
            "input_generation.providers: must be non-empty when enabled=true and target_inputs>=1"
        )
    _validate_providers(ig.providers, "input_generation", used_in_phase="input")

    og = cfg.output_generation
    if og.selection_policy not in _VALID_SELECTION_POLICIES:
        raise ConfigError(
            f"output_generation.selection_policy: {og.selection_policy!r} "
            f"not in {sorted(_VALID_SELECTION_POLICIES)}"
        )
    if og.enabled and not og.providers:
        raise ConfigError(
            "output_generation.providers: must be non-empty when enabled=true"
        )
    _validate_providers(og.providers, "output_generation", used_in_phase="output")
    declared = {p.name for p in og.providers}
    for name in og.provider_priority:
        if name not in declared:
            raise ConfigError(
                f"output_generation.provider_priority: {name!r} not in declared providers"
            )


def _validate_providers(
    providers: tuple[ProviderConfig, ...], phase_path: str, *, used_in_phase: str,
) -> None:
    """Cross-provider rules that the parser couldn't check per-row (uniqueness,
    type-conditional requirements). Min-value/type already enforced upstream."""
    seen_names: set[str] = set()
    for i, p in enumerate(providers):
        prefix = f"{phase_path}.providers[{i}]"
        if not p.name:
            raise ConfigError(f"{prefix}.name: must be non-empty")
        if p.name in seen_names:
            raise ConfigError(
                f"{prefix}.name: duplicate provider name {p.name!r} within {phase_path}"
            )
        seen_names.add(p.name)
        if p.provider_type not in _VALID_PROVIDER_TYPES:
            raise ConfigError(
                f"{prefix}.type: {p.provider_type!r} not in {sorted(_VALID_PROVIDER_TYPES)}"
            )
        if p.provider_type == "fake":
            if used_in_phase == "input" and not p.fixture_inputs:
                raise ConfigError(
                    f"{prefix}: fake provider in input phase requires fixture_inputs"
                )
            if used_in_phase == "output" and not p.fixture_labels:
                raise ConfigError(
                    f"{prefix}: fake provider in output phase requires fixture_labels"
                )
        if p.provider_type in {"deepseek", "gemini", "bedrock"} and not p.model:
            raise ConfigError(f"{prefix}: {p.provider_type} provider requires model")
        if p.structured_output and p.provider_type not in {"gemini", "bedrock"}:
            logger.warning(
                "%s: structured_output=true on provider type %r is a no-op",
                prefix, p.provider_type,
            )
