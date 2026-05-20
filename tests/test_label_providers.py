"""Tests for real provider generate_label implementations.

Uses the shared `patch_sdk_clients` fixture from tests/conftest.py so
provider __init__ never makes network calls.
"""
import pytest

from llm_providers import DeepSeekProvider, ProviderError


# ---- DeepSeek generate_label ---------------------------------------------

def test_deepseek_generate_label_returns_raw(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    monkeypatch.setattr(
        p, "_call_api_messages",
        lambda msgs: '{"transactions":[{"amount":500,"currency":"INR","item":"beer","category":"Drinks","type":"expense"}]}',
    )
    raw = p.generate_label("500 beer")
    assert '"transactions"' in raw
    assert '"beer"' in raw


def test_deepseek_label_uses_build_messages(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    captured = {}
    def capture(msgs):
        captured["msgs"] = msgs
        return "{}"
    monkeypatch.setattr(p, "_call_api_messages", capture)
    p.generate_label("500 beer")
    msgs = captured["msgs"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "500 beer"


# ---- DeepSeek omits None kwargs in label path -----------------------------

class _FakeMessage:
    content = "ok line"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResp:
    choices = [_FakeChoice()]


def test_deepseek_label_omits_none_kwargs(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    captured = {}
    p = DeepSeekProvider(name="ds", model="deepseek-chat",
                         temperature=None, max_tokens=None)
    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResp()
    monkeypatch.setattr(p._client.chat.completions, "create", fake_create)
    p.generate_label("500 beer")
    assert "temperature" not in captured
    assert "max_tokens" not in captured
    assert captured["model"] == "deepseek-chat"
    assert captured["messages"][1]["content"] == "500 beer"


# ---- DeepSeek retry on transient (using fake exception class) -------------

class _FakeTransientError(Exception):
    pass


def test_deepseek_label_retries_on_transient(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat", max_retries=2)
    p._retryable_excs = (_FakeTransientError,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def flaky(_msgs):
        state["calls"] += 1
        if state["calls"] < 2:
            raise _FakeTransientError("transient")
        return "ok"
    monkeypatch.setattr(p, "_call_api_messages", flaky)

    result = p.generate_label("500 beer")
    assert state["calls"] == 2
    assert result == "ok"


# ---- Gemini structured output --------------------------------------------

def test_label_schema_for_gemini_uses_lib_enums():
    from llm_providers import _label_schema_for_gemini
    from _lib import CATEGORIES, TYPES, CURRENCIES
    schema = _label_schema_for_gemini()
    txn_props = schema["properties"]["transactions"]["items"]["properties"]
    assert set(txn_props["category"]["enum"]) == set(CATEGORIES)
    assert set(txn_props["type"]["enum"]) == set(TYPES)
    assert set(txn_props["currency"]["enum"]) == set(CURRENCIES)


def test_gemini_label_with_structured_output_passes_schema(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", structured_output=True)
    captured = {}

    class FakeResp:
        text = '{"transactions":[]}'

    def fake_generate(model, contents, config):
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return FakeResp()
    monkeypatch.setattr(p._client.models, "generate_content", fake_generate)
    p.generate_label("500 beer")
    assert captured["model"] == "gemini-2.5-flash"
    assert captured["contents"] == "500 beer"
    assert captured["config"] is not None


def test_gemini_label_without_structured_output_still_runs(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", structured_output=False)
    captured = {}

    class FakeResp:
        text = '{"transactions":[]}'

    def fake_generate(model, contents, config):
        captured["config"] = config
        return FakeResp()
    monkeypatch.setattr(p._client.models, "generate_content", fake_generate)
    p.generate_label("500 beer")
    assert captured["config"] is not None


def test_gemini_generate_inputs_still_ignores_structured_output(monkeypatch, patch_sdk_clients):
    """structured_output=True must NOT affect generate_inputs (regression check)."""
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", structured_output=True)
    monkeypatch.setattr(p, "_call_api", lambda prompt: "500 beer\n200 chai")
    result = p.generate_inputs("ignored", n=2)
    assert result == ["500 beer", "200 chai"]


# ---- Gemini label retry behavior ------------------------------------------

class _FakeGeminiRetryable(Exception):
    def __init__(self, status):
        super().__init__()
        self.status_code = status


def test_gemini_label_retries_on_429(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", max_retries=2)
    p._retryable_excs = (_FakeGeminiRetryable,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def flaky(input_text):
        state["calls"] += 1
        if state["calls"] < 2:
            raise _FakeGeminiRetryable(429)
        return "ok"
    monkeypatch.setattr(p, "_call_api_label", flaky)
    result = p.generate_label("500 beer")
    assert state["calls"] == 2
    assert result == "ok"


# ---- LocalTeacherProvider -------------------------------------------------

def test_local_teacher_factory_requires_cfg_model(monkeypatch):
    """Factory raises ConfigError when local_teacher has no model (adapter dir)."""
    from llm_providers import create_provider
    from generation_config import ProviderConfig, ConfigError

    cfg = ProviderConfig(name="lt", provider_type="local_teacher", model=None)
    with pytest.raises(ConfigError) as exc:
        create_provider(cfg)
    assert "local_teacher" in str(exc.value)
    assert "model" in str(exc.value).lower() or "adapter" in str(exc.value).lower()


def test_local_teacher_init_does_no_io(tmp_path):
    from llm_providers import LocalTeacherProvider
    p = LocalTeacherProvider(name="lt", adapter_dir=tmp_path / "fake_adapter")
    assert p._backend is None
    assert p.model is None


def test_local_teacher_generate_inputs_raises():
    from llm_providers import LocalTeacherProvider
    p = LocalTeacherProvider(name="lt", adapter_dir=None)
    with pytest.raises(NotImplementedError) as exc:
        p.generate_inputs("prompt", n=5)
    assert "label-only" in str(exc.value)


def test_local_teacher_missing_adapter_dir_raises(monkeypatch, tmp_path):
    """If adapter_dir doesn't exist, first generate_label raises ProviderError."""
    from llm_providers import LocalTeacherProvider, ProviderError
    p = LocalTeacherProvider(name="lt", adapter_dir=tmp_path / "does_not_exist")
    with pytest.raises(ProviderError) as exc:
        p.generate_label("500 beer")
    assert "adapter dir not found" in str(exc.value).lower()


def test_local_teacher_lazy_loads_on_first_call(monkeypatch, tmp_path):
    """First generate_label triggers backend load; second reuses."""
    from llm_providers import LocalTeacherProvider
    fake_adapter = tmp_path / "fake_adapter"
    fake_adapter.mkdir()
    p = LocalTeacherProvider(name="lt", adapter_dir=fake_adapter)

    state = {"build_calls": 0, "generate_calls": 0}

    class FakeBackend:
        def generate_label(self, msgs, *, max_new_tokens):
            state["generate_calls"] += 1
            return '{"transactions":[]}'

    def fake_build(*, adapter_dir, max_seq_length):
        state["build_calls"] += 1
        return FakeBackend()

    import _lib
    monkeypatch.setattr(_lib, "build_teacher_fp16_backend", fake_build)

    p.generate_label("500 beer")
    assert state["build_calls"] == 1
    assert state["generate_calls"] == 1
    assert p._backend is not None
    assert p.model == "fake_adapter"

    p.generate_label("200 chai")
    assert state["build_calls"] == 1   # reused
    assert state["generate_calls"] == 2


def test_local_teacher_serializes_concurrent_calls(monkeypatch, tmp_path):
    """Lock ensures multiple threads don't call generate_label concurrently."""
    import threading
    import time
    from llm_providers import LocalTeacherProvider
    fake_adapter = tmp_path / "fake_adapter"
    fake_adapter.mkdir()
    p = LocalTeacherProvider(name="lt", adapter_dir=fake_adapter)

    in_flight = [0]
    max_concurrent = [0]
    counter_lock = threading.Lock()

    class FakeBackend:
        def generate_label(self, msgs, *, max_new_tokens):
            with counter_lock:
                in_flight[0] += 1
                max_concurrent[0] = max(max_concurrent[0], in_flight[0])
            time.sleep(0.02)
            with counter_lock:
                in_flight[0] -= 1
            return '{"transactions":[]}'

    import _lib
    monkeypatch.setattr(_lib, "build_teacher_fp16_backend",
                        lambda *, adapter_dir, max_seq_length: FakeBackend())

    threads = [threading.Thread(target=lambda: p.generate_label("x")) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert max_concurrent[0] == 1


def test_local_teacher_via_factory_smoke(monkeypatch, tmp_path):
    """create_provider builds a LocalTeacherProvider when cfg.model is set."""
    from llm_providers import create_provider, LocalTeacherProvider
    from generation_config import ProviderConfig

    fake_adapter = tmp_path / "adapter"
    fake_adapter.mkdir()
    cfg = ProviderConfig(name="lt", provider_type="local_teacher", model=str(fake_adapter))
    p = create_provider(cfg)
    assert isinstance(p, LocalTeacherProvider)
    assert p.adapter_dir == fake_adapter
    assert p.max_new_tokens == 384   # default
