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


import pytest as _pytest    # alias avoids confusion with test-level pytest imports


@_pytest.fixture
def patch_sdk_clients(monkeypatch):
    """Patch openai.OpenAI and google.genai.Client so provider __init__
    never makes network calls. Opt-in: tests reference `patch_sdk_clients`
    by parameter name when they want SDK construction stubbed.
    """
    import openai
    monkeypatch.setattr(openai, "OpenAI", _StubOpenAIClient)
    from google import genai
    monkeypatch.setattr(genai, "Client", _StubGeminiClient)
