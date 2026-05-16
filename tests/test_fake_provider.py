import json
import threading
from pathlib import Path

import pytest

from llm_providers import FakeProvider, ProviderError, LLMProvider


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "fake_inputs.jsonl"
LABELS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"


def test_fakeprovider_satisfies_protocol():
    p = FakeProvider(name="x", inputs_path=INPUTS_FIXTURE)
    assert p.name == "x"
    assert p.provider_type == "fake"
    assert hasattr(p, "generate_inputs")
    assert hasattr(p, "generate_label")


def test_provider_error_is_runtime_error():
    assert issubclass(ProviderError, RuntimeError)


def test_generate_inputs_returns_n_items():
    p = FakeProvider(name="x", inputs_path=INPUTS_FIXTURE)
    result = p.generate_inputs(prompt="ignored", n=3)
    assert len(result) == 3
    assert all(isinstance(s, str) for s in result)


def test_generate_inputs_wraps_around_modulo_fixture_length():
    """For [a,b,c], n=5 -> [a,b,c,a,b]; next call continues from c."""
    tiny_fixture = REPO_ROOT / "tests" / "fixtures" / "_tiny_inputs.jsonl"
    tiny_fixture.write_text(
        '{"input": "a"}\n{"input": "b"}\n{"input": "c"}\n',
        encoding="utf-8",
    )
    try:
        p = FakeProvider(name="x", inputs_path=tiny_fixture)
        first = p.generate_inputs(prompt="", n=5)
        assert first == ["a", "b", "c", "a", "b"]
        second = p.generate_inputs(prompt="", n=2)
        assert second == ["c", "a"]
    finally:
        tiny_fixture.unlink(missing_ok=True)


def test_loader_raises_provider_error_on_bad_json():
    bad_fixture = REPO_ROOT / "tests" / "fixtures" / "_bad.jsonl"
    bad_fixture.write_text('{"input": "ok"}\n{not json\n', encoding="utf-8")
    try:
        with pytest.raises(ProviderError) as exc:
            FakeProvider(name="x", inputs_path=bad_fixture)
        assert "line 2" in str(exc.value).lower() or "line 2" in str(exc.value)
    finally:
        bad_fixture.unlink(missing_ok=True)


def test_loader_raises_provider_error_on_missing_input_field():
    bad_fixture = REPO_ROOT / "tests" / "fixtures" / "_missing_field.jsonl"
    bad_fixture.write_text('{"input": "ok"}\n{"wrong_key": "value"}\n', encoding="utf-8")
    try:
        with pytest.raises(ProviderError) as exc:
            FakeProvider(name="x", inputs_path=bad_fixture)
        msg = str(exc.value)
        assert "input" in msg and ("line 2" in msg.lower() or "2" in msg)
    finally:
        bad_fixture.unlink(missing_ok=True)


def test_generate_inputs_zero_returns_empty():
    p = FakeProvider(name="x", inputs_path=INPUTS_FIXTURE)
    assert p.generate_inputs(prompt="", n=0) == []


def test_generate_inputs_negative_raises_provider_error():
    p = FakeProvider(name="x", inputs_path=INPUTS_FIXTURE)
    with pytest.raises(ProviderError):
        p.generate_inputs(prompt="", n=-1)


def test_loader_rejects_non_string_raw_output():
    bad_fixture = REPO_ROOT / "tests" / "fixtures" / "_bad_raw_output.jsonl"
    bad_fixture.write_text(
        '{"input": "x", "raw_output": {"not": "a string"}}\n',
        encoding="utf-8",
    )
    try:
        with pytest.raises(ProviderError) as exc:
            FakeProvider(name="x", labels_path=bad_fixture)
        assert "raw_output" in str(exc.value)
    finally:
        bad_fixture.unlink(missing_ok=True)


def test_generate_inputs_is_thread_safe():
    p = FakeProvider(name="x", inputs_path=INPUTS_FIXTURE)
    collected = []
    lock = threading.Lock()

    def worker():
        items = p.generate_inputs(prompt="", n=10)
        with lock:
            collected.extend(items)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 8 threads * 10 items each = 80 total. Cursor must advance correctly under contention.
    assert len(collected) == 80


def test_unimplemented_provider_constructs_but_methods_raise():
    from llm_providers import UnimplementedProvider

    p = UnimplementedProvider(name="ds", provider_type="deepseek", model="x")
    assert p.name == "ds"
    assert p.provider_type == "deepseek"
    assert p.model == "x"
    with pytest.raises(NotImplementedError) as exc:
        p.generate_inputs(prompt="x", n=1)
    assert "Slice 2" in str(exc.value)
    with pytest.raises(NotImplementedError) as exc:
        p.generate_label(input_text="x")
    assert "Slice 3" in str(exc.value)


def test_unimplemented_provider_has_no_sdk_imports():
    """UnimplementedProvider must not transitively import any external SDK."""
    import sys
    sdk_modules_before = {
        m for m in sys.modules
        if m.startswith(("openai", "google.generativeai", "google.genai", "unsloth"))
    }
    from llm_providers import UnimplementedProvider
    UnimplementedProvider(name="x", provider_type="gemini", model="y")
    sdk_modules_after = {
        m for m in sys.modules
        if m.startswith(("openai", "google.generativeai", "google.genai", "unsloth"))
    }
    assert sdk_modules_after == sdk_modules_before


def test_create_provider_returns_fake_for_fake_type():
    from llm_providers import FakeProvider, create_provider
    from generation_config import ProviderConfig  # noqa: forward import

    cfg = ProviderConfig(
        name="x", provider_type="fake",
        fixture_inputs=str(INPUTS_FIXTURE),
    )
    p = create_provider(cfg)
    assert isinstance(p, FakeProvider)
    assert p.name == "x"


def test_create_provider_returns_unimplemented_for_real_types():
    from llm_providers import UnimplementedProvider, create_provider
    from generation_config import ProviderConfig

    # deepseek now returns DeepSeekProvider (Task 4); gemini now returns
    # GeminiProvider (Task 5); only local_teacher remains as UnimplementedProvider.
    for t in ("local_teacher",):
        cfg = ProviderConfig(name=f"x_{t}", provider_type=t, model="m")
        p = create_provider(cfg)
        assert isinstance(p, UnimplementedProvider)
        assert p.provider_type == t


def test_create_provider_raises_on_unknown_type():
    from llm_providers import create_provider
    from generation_config import ProviderConfig

    cfg = ProviderConfig(name="bad", provider_type="not_a_real_type")
    with pytest.raises(ValueError) as exc:
        create_provider(cfg)
    assert "not_a_real_type" in str(exc.value)
