"""Pytest config — put scripts/ on sys.path so tests can use the same
flat-import style as the scripts themselves (`from amount_parser import ...`)."""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Shared SDK-stub fixture used by tests that construct real DeepSeek/Gemini
# providers and need their SDK Clients stubbed. Opt-in via parameter name —
# NOT autouse globally, so tests that want to verify lazy SDK imports can
# skip it.
# ---------------------------------------------------------------------------


class _StubChatCompletions:
    def create(self, **kwargs):
        raise AssertionError("real OpenAI client should not be called in tests")


class _StubChat:
    def __init__(self):
        self.completions = _StubChatCompletions()


class _StubOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = _StubChat()


class _StubGeminiModels:
    def generate_content(self, **kwargs):
        raise AssertionError("real Gemini client should not be called in tests")


class _StubGeminiClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = _StubGeminiModels()


class _StubBedrockRuntimeClient:
    """Stand-in for boto3.client('bedrock-runtime', ...). Captures the last
    converse(**kwargs) call and returns a configurable response. Tests
    set `.next_response` to control what converse() returns and inspect
    `.last_kwargs` to verify the request shape."""

    def __init__(self, **client_kwargs):
        self.client_kwargs = client_kwargs
        self.last_kwargs: dict | None = None
        self.next_response: dict = {
            "output": {"message": {"role": "assistant", "content": [{"text": ""}]}},
            "usage": {"inputTokens": 0, "outputTokens": 0},
            "stopReason": "end_turn",
        }
        self.raise_on_call: BaseException | None = None

    def converse(self, **kwargs):
        self.last_kwargs = kwargs
        if self.raise_on_call is not None:
            exc = self.raise_on_call
            self.raise_on_call = None  # one-shot
            raise exc
        return self.next_response


import pytest as _pytest    # alias avoids confusion with test-level pytest imports


@_pytest.fixture
def patch_sdk_clients(monkeypatch):
    """Patch openai.OpenAI, google.genai.Client, and boto3.client so
    provider __init__ never makes network calls. Opt-in: tests reference
    `patch_sdk_clients` by parameter name when they want SDK construction
    stubbed.
    """
    import openai
    monkeypatch.setattr(openai, "OpenAI", _StubOpenAIClient)
    from google import genai
    monkeypatch.setattr(genai, "Client", _StubGeminiClient)

    # boto3 may not be installed in every test environment; only patch if
    # importable. Tests that need it should also be marked or import-guarded.
    try:
        import boto3
    except ImportError:
        return

    def _fake_client(service_name, **kwargs):
        # Only Bedrock runtime is stubbed; raise loudly for any other
        # service so tests catch surprise S3/STS calls.
        if service_name != "bedrock-runtime":
            raise AssertionError(
                f"Unexpected boto3.client({service_name!r}) in test"
            )
        return _StubBedrockRuntimeClient(**kwargs)

    monkeypatch.setattr(boto3, "client", _fake_client)
