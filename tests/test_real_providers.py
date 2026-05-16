"""Tests for real provider implementations (DeepSeek + Gemini).

Strategy: SDK imports happen inside provider __init__, so the imports
are exercised by construction. SDK Client constructors are patched
(see _stub_openai_client / _stub_gemini_client fixtures) so no real
network I/O happens during construction. Behavior tests then patch
provider._call_api directly to control responses.
"""
import pytest

from llm_providers import (
    DeepSeekProvider,
    ProviderError,
    _parse_input_lines,
)


# ---- DeepSeek construction ------------------------------------------------

def test_deepseek_constructs_with_env_key(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    assert p.name == "ds"
    assert p.provider_type == "deepseek"
    assert p.model == "deepseek-chat"


def test_deepseek_constructs_with_explicit_api_key(monkeypatch, patch_sdk_clients):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    p = DeepSeekProvider(name="ds", model="deepseek-chat", api_key="explicit-key")
    assert p.name == "ds"


def test_deepseek_missing_key_raises_provider_error(monkeypatch, patch_sdk_clients):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ProviderError) as exc:
        DeepSeekProvider(name="ds", model="deepseek-chat")
    assert "DEEPSEEK_API_KEY" in str(exc.value)


def test_deepseek_does_not_fall_back_to_openai_key(monkeypatch, patch_sdk_clients):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-be-used")
    with pytest.raises(ProviderError):
        DeepSeekProvider(name="ds", model="deepseek-chat")


# ---- DeepSeek generate_inputs ---------------------------------------------

def test_deepseek_generate_inputs_returns_parsed_lines(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    monkeypatch.setattr(p, "_call_api", lambda prompt: "500 beer\n200 chai\n100 samosa")
    result = p.generate_inputs("ignored prompt", n=3)
    assert result == ["500 beer", "200 chai", "100 samosa"]


def test_deepseek_generate_inputs_truncates_to_n(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    monkeypatch.setattr(p, "_call_api", lambda prompt: "a x\nb y\nc z\nd q\ne r")
    assert len(p.generate_inputs("ignored", n=3)) == 3


def test_deepseek_generate_inputs_zero_returns_empty(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    # _call_api should not even be invoked.
    monkeypatch.setattr(p, "_call_api",
                        lambda _: (_ for _ in ()).throw(AssertionError("should not call API")))
    assert p.generate_inputs("ignored", n=0) == []


def test_deepseek_generate_inputs_negative_n_raises(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    with pytest.raises(ProviderError):
        p.generate_inputs("ignored", n=-1)


# ---- DeepSeek retry behavior (using fake exception, NOT real SDK error) ---

class _FakeTransientError(Exception):
    pass


def test_deepseek_retries_on_transient(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat", max_retries=2)
    # Reassign retryable to our fake class so we don't depend on SDK exception constructor.
    p._retryable_excs = (_FakeTransientError,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def flaky(_prompt):
        state["calls"] += 1
        if state["calls"] < 2:
            raise _FakeTransientError("transient")
        return "ok line one"
    monkeypatch.setattr(p, "_call_api", flaky)

    result = p.generate_inputs("prompt", n=1)
    assert state["calls"] == 2
    assert result == ["ok line one"]


def test_deepseek_max_retries_exhausted_raises(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat", max_retries=1)
    p._retryable_excs = (_FakeTransientError,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def always_fail(_prompt):
        state["calls"] += 1
        raise _FakeTransientError("persistent")
    monkeypatch.setattr(p, "_call_api", always_fail)

    with pytest.raises(_FakeTransientError):
        p.generate_inputs("prompt", n=1)
    assert state["calls"] == 2   # 1 try + 1 retry


def test_deepseek_retryable_excs_is_populated_with_real_sdk_types(monkeypatch, patch_sdk_clients):
    """Verify __init__ wires the real SDK exception classes (just shape, no call)."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    from openai import RateLimitError, APITimeoutError, APIConnectionError
    assert RateLimitError in p._retryable_excs
    assert APITimeoutError in p._retryable_excs
    assert APIConnectionError in p._retryable_excs


# ---- DeepSeek omits None kwargs --------------------------------------------

class _FakeMessage:
    content = "ok line"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResp:
    choices = [_FakeChoice()]


def test_deepseek_omits_temperature_and_max_tokens_when_none(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    captured = {}
    p = DeepSeekProvider(name="ds", model="deepseek-chat",
                         temperature=None, max_tokens=None)
    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResp()
    monkeypatch.setattr(p._client.chat.completions, "create", fake_create)
    p.generate_inputs("prompt", n=1)
    assert "temperature" not in captured
    assert "max_tokens" not in captured
    assert captured["model"] == "deepseek-chat"


def test_deepseek_passes_temperature_and_max_tokens_when_set(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    captured = {}
    p = DeepSeekProvider(name="ds", model="deepseek-chat",
                         temperature=0.7, max_tokens=500)
    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResp()
    monkeypatch.setattr(p._client.chat.completions, "create", fake_create)
    p.generate_inputs("prompt", n=1)
    assert captured["temperature"] == 0.7
    assert captured["max_tokens"] == 500


# ---- _parse_input_lines (direct unit) -------------------------------------

def test_parse_input_lines_cleans_dedupes_and_truncates():
    text = "1. 500 beer\n500 beer\n- 200 chai\n```"
    assert _parse_input_lines(text, expected=2) == ["500 beer", "200 chai"]


def test_parse_input_lines_handles_empty_input():
    assert _parse_input_lines("", expected=5) == []


def test_parse_input_lines_local_dedupe_is_case_insensitive():
    text = "500 Beer\n500 BEER\n200 chai"
    # First "500 Beer" wins; "500 BEER" is rejected as duplicate.
    result = _parse_input_lines(text, expected=10)
    assert result == ["500 Beer", "200 chai"]


# ---- Gemini construction --------------------------------------------------

def test_gemini_constructs_with_google_api_key(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash")
    assert p.name == "g"
    assert p.provider_type == "gemini"


def test_gemini_falls_back_to_gemini_api_key_env(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash")
    assert p.name == "g"


def test_gemini_missing_key_raises(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError) as exc:
        GeminiProvider(name="g", model="gemini-2.5-flash")
    assert "GOOGLE_API_KEY" in str(exc.value) or "GEMINI_API_KEY" in str(exc.value)


def test_gemini_generate_inputs_returns_parsed_lines(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash")
    monkeypatch.setattr(p, "_call_api", lambda prompt: "500 beer\n200 chai")
    result = p.generate_inputs("ignored", n=2)
    assert result == ["500 beer", "200 chai"]


# ---- _is_retryable_gemini_error predicate ---------------------------------

def test_is_retryable_gemini_error_retries_429_and_5xx(monkeypatch):
    from llm_providers import _is_retryable_gemini_error

    class E(Exception):
        def __init__(self, status):
            super().__init__()
            self.status_code = status

    for status in (429, 500, 502, 503, 504):
        assert _is_retryable_gemini_error(E(status)) is True


def test_is_retryable_gemini_error_does_not_retry_4xx(monkeypatch):
    from llm_providers import _is_retryable_gemini_error

    class E(Exception):
        def __init__(self, status):
            super().__init__()
            self.status_code = status

    for status in (400, 401, 403, 404):
        assert _is_retryable_gemini_error(E(status)) is False


def test_is_retryable_gemini_error_handles_string_status(monkeypatch):
    """SDK variation: status may be exposed as a string in some versions."""
    from llm_providers import _is_retryable_gemini_error

    class E(Exception):
        def __init__(self, status):
            super().__init__()
            self.status_code = status

    assert _is_retryable_gemini_error(E("429")) is True
    assert _is_retryable_gemini_error(E("400")) is False


def test_is_retryable_gemini_error_handles_no_status_attr():
    from llm_providers import _is_retryable_gemini_error
    assert _is_retryable_gemini_error(Exception("no status here")) is False


# ---- Gemini retry behavior (using fake exception class) -------------------

def test_gemini_retries_on_429_then_succeeds(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", max_retries=2)

    class FakeRetryable(Exception):
        def __init__(self, status):
            super().__init__()
            self.status_code = status

    p._retryable_excs = (FakeRetryable,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def flaky(_prompt):
        state["calls"] += 1
        if state["calls"] < 2:
            raise FakeRetryable(429)
        return "ok line"
    monkeypatch.setattr(p, "_call_api", flaky)
    result = p.generate_inputs("prompt", n=1)
    assert state["calls"] == 2
    assert result == ["ok line"]


def test_gemini_does_not_retry_400(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", max_retries=5)

    class FakeRetryable(Exception):
        def __init__(self, status):
            super().__init__()
            self.status_code = status

    p._retryable_excs = (FakeRetryable,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def fails_with_400(_prompt):
        state["calls"] += 1
        raise FakeRetryable(400)
    monkeypatch.setattr(p, "_call_api", fails_with_400)

    with pytest.raises(FakeRetryable):
        p.generate_inputs("prompt", n=1)
    assert state["calls"] == 1   # NOT retried — predicate filters 400 out


def test_gemini_retryable_excs_is_populated_with_real_sdk_types(monkeypatch, patch_sdk_clients):
    """Verify __init__ wires the real SDK exception classes (just shape, no call)."""
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash")
    from google.genai import errors as genai_errors
    assert genai_errors.APIError in p._retryable_excs
    assert genai_errors.ClientError in p._retryable_excs
