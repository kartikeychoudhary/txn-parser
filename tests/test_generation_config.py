import json
from pathlib import Path

import pytest

from generation_config import (
    ConfigError,
    GenerationConfig,
    InputGenerationConfig,
    OutputGenerationConfig,
    ProviderConfig,
    RateLimitsConfig,
    ValidationGateConfig,
    load_generation_config,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _minimal_dict() -> dict:
    return {
        "version": 1,
        "input_generation": {
            "enabled": True,
            "target_inputs": 10,
            "batch_size": 5,
            "providers": [
                {"name": "fake_a", "type": "fake", "weight": 60, "threads": 1,
                 "fixture_inputs": str(FIXTURES / "fake_inputs.jsonl")},
            ],
        },
        "output_generation": {
            "enabled": True,
            "label_attempts_per_input": 1,
            "providers": [
                {"name": "fake_a", "type": "fake", "weight": 50, "threads": 1,
                 "fixture_labels": str(FIXTURES / "fake_labels.jsonl")},
            ],
        },
        "validation": {"schema": True, "semantic_validator": True,
                       "reject_invalid": True,
                       "retry_invalid_with_stricter_prompt": False,
                       "max_repair_attempts": 0},
        "rate_limits": {"global_max_workers": 4, "write_flush_every": 10},
    }


def test_providerconfig_is_frozen_dataclass():
    cfg = ProviderConfig(name="x", provider_type="fake")
    assert cfg.name == "x"
    assert cfg.provider_type == "fake"
    assert cfg.weight == 1  # default
    with pytest.raises(Exception):
        cfg.name = "changed"  # frozen


def test_load_minimal_valid_config(tmp_path):
    path = _write_config(tmp_path, _minimal_dict())
    cfg = load_generation_config(path)
    assert isinstance(cfg, GenerationConfig)
    assert cfg.version == 1
    assert cfg.input_generation.target_inputs == 10
    assert cfg.input_generation.providers[0].name == "fake_a"
    assert cfg.input_generation.providers[0].provider_type == "fake"
    assert cfg.source_path == str(path)


def test_load_default_version_is_1(tmp_path):
    data = _minimal_dict()
    del data["version"]
    path = _write_config(tmp_path, data)
    cfg = load_generation_config(path)
    assert cfg.version == 1


def test_config_error_is_value_error():
    assert issubclass(ConfigError, ValueError)


def test_unknown_type_raises_config_error(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"][0]["type"] = "not_a_real_type"
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "not_a_real_type" in str(exc.value)


def test_duplicate_provider_name_in_phase_raises(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"].append(
        {"name": "fake_a", "type": "fake", "weight": 10, "threads": 1,
         "fixture_inputs": str(FIXTURES / "fake_inputs.jsonl")},
    )
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "fake_a" in str(exc.value)


def test_weight_zero_raises(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"][0]["weight"] = 0
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "weight" in str(exc.value)


def test_negative_max_tokens_raises(tmp_path):
    data = _minimal_dict()
    data["output_generation"]["providers"][0]["max_tokens"] = 0
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        load_generation_config(path)


def test_negative_temperature_raises(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"][0]["temperature"] = -0.5
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        load_generation_config(path)


def test_fake_provider_in_input_requires_fixture_inputs(tmp_path):
    data = _minimal_dict()
    del data["input_generation"]["providers"][0]["fixture_inputs"]
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "fixture_inputs" in str(exc.value)


def test_deepseek_provider_requires_model(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"][0] = {
        "name": "ds", "type": "deepseek", "weight": 10, "threads": 1,
    }
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "model" in str(exc.value)


def test_enabled_false_skips_required_provider_list(tmp_path):
    data = _minimal_dict()
    data["input_generation"] = {"enabled": False, "target_inputs": 0,
                                 "batch_size": 1, "providers": []}
    path = _write_config(tmp_path, data)
    cfg = load_generation_config(path)
    assert cfg.input_generation.enabled is False
    assert cfg.input_generation.providers == ()


def test_missing_input_providers_when_enabled_raises(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"] = []
    data["input_generation"]["target_inputs"] = 10
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "providers" in str(exc.value)


def test_missing_fixture_file_raises(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"][0]["fixture_inputs"] = "nonexistent.jsonl"
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "not found" in str(exc.value)


def test_unknown_top_level_key_warns(tmp_path, caplog):
    data = _minimal_dict()
    data["unknown_top_key"] = "x"
    path = _write_config(tmp_path, data)
    with caplog.at_level("WARNING"):
        load_generation_config(path)
    assert any("unknown_top_key" in r.message.lower() or "unknown_top_key" in r.message
               for r in caplog.records)


def test_provider_priority_must_reference_declared_provider(tmp_path):
    data = _minimal_dict()
    data["output_generation"]["provider_priority"] = ["fake_a", "ghost"]
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "ghost" in str(exc.value)


def test_version_2_raises(tmp_path):
    data = _minimal_dict()
    data["version"] = 2
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "version" in str(exc.value)


def test_unknown_selection_policy_raises(tmp_path):
    data = _minimal_dict()
    data["output_generation"]["selection_policy"] = "magic"
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "selection_policy" in str(exc.value)


@pytest.mark.parametrize("path,value,expect_substring", [
    ("input_generation.enabled", "false", "enabled"),
    ("input_generation.target_inputs", "10", "target_inputs"),
    ("input_generation.batch_size", "5", "batch_size"),
    ("output_generation.label_attempts_per_input", "2", "label_attempts_per_input"),
    ("output_generation.selection_policy", 123, "selection_policy"),
    ("output_generation.provider_priority", "fake_a", "provider_priority"),
    ("validation.max_repair_attempts", "0", "max_repair_attempts"),
    ("rate_limits.global_max_workers", "4", "global_max_workers"),
])
def test_wrong_type_fields_raise_config_error(tmp_path, path, value, expect_substring):
    """Strict loader: no silent coercion of strings to int/bool."""
    data = _minimal_dict()
    # Walk dotted path and set the value.
    parts = path.split(".")
    target = data
    for key in parts[:-1]:
        target = target[key]
    target[parts[-1]] = value
    config_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(config_path)
    assert expect_substring in str(exc.value)


@pytest.mark.parametrize("field,bad_value", [
    ("weight", "1"),
    ("threads", "1"),
    ("temperature", "0.5"),
    ("max_tokens", "512"),
    ("max_retries", "3"),
    ("structured_output", "true"),
    ("seed", "42"),
])
def test_wrong_type_provider_fields_raise(tmp_path, field, bad_value):
    data = _minimal_dict()
    data["input_generation"]["providers"][0][field] = bad_value
    config_path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(config_path)
    assert field in str(exc.value)


def test_fixture_path_resolved_relative_to_config_dir(tmp_path):
    """Place config in tmp_path/configs/, fixture path '../inputs.jsonl'
    must resolve to tmp_path/inputs.jsonl."""
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text('{"input":"x"}\n', encoding="utf-8")
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    labels = FIXTURES / "fake_labels.jsonl"
    data = _minimal_dict()
    data["input_generation"]["providers"][0]["fixture_inputs"] = "../inputs.jsonl"
    data["output_generation"]["providers"][0]["fixture_labels"] = str(labels)
    config_path = configs_dir / "providers.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    cfg = load_generation_config(config_path)
    resolved = cfg.input_generation.providers[0].fixture_inputs
    assert Path(resolved) == inputs.resolve()


def test_test_providers_config_loads_and_dry_runs():
    from generation_orchestrator import render_dry_run
    config_path = REPO_ROOT / "configs" / "test_providers.json"
    cfg = load_generation_config(config_path)
    output = render_dry_run(cfg)
    assert "Multi-provider dry run" in output
    assert "fake_a" in output
    assert "fake_b" in output


def test_example_config_loads_and_dry_runs_without_sdk_imports():
    """The example config has deepseek/gemini providers but must load
    cleanly without network I/O.

    Since Task 4, deepseek returns a real DeepSeekProvider (requires an API
    key and imports openai). Since Task 5, gemini returns a real GeminiProvider
    (requires an API key and imports google-genai). We skip create_provider for
    deepseek and gemini here — their construction is covered by
    test_real_providers.py. local_teacher remains as UnimplementedProvider.
    """
    import sys
    from generation_orchestrator import render_dry_run
    from llm_providers import UnimplementedProvider, create_provider

    config_path = REPO_ROOT / "configs" / "example_providers.json"
    cfg = load_generation_config(config_path)
    output = render_dry_run(cfg)
    # local_teacher still uses UnimplementedProvider (real impl lands later).
    for p in cfg.input_generation.providers:
        if p.provider_type == "local_teacher":
            provider = create_provider(p)
            assert isinstance(provider, UnimplementedProvider)
    assert "deepseek_v4_pro" in output
    assert "gemini_flash" in output
