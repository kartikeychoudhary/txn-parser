"""GgufBackend wiring tests: verify the grammar plumbing without real
llama_cpp or any real GGUF file. We inject a fake `llama_cpp` module
into sys.modules so tests run anywhere.
"""
import sys
import types
from importlib import import_module
from pathlib import Path

import pytest


class _FakeLlama:
    def __init__(self, *args, **kwargs):
        # Tolerant of any signature shift in real Llama; we only care
        # that GgufBackend can construct one.
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {"content": '{"transactions":[]}'}}],
            "usage": {"completion_tokens": 3},
        }


class _FakeGrammar:
    """Stand-in for llama_cpp.LlamaGrammar. Provides a from_string
    classmethod so any code path that compiles grammar (instead of
    monkeypatching grammar.load_label_grammar) still works."""

    @classmethod
    def from_string(cls, _grammar: str):
        return cls()


def _inject_fake_llama_cpp(monkeypatch):
    fake = types.SimpleNamespace(Llama=_FakeLlama, LlamaGrammar=_FakeGrammar)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)


def _eval_module():
    """Import the stage-5-eval module by its file-path name.

    The script is named `04_eval.py` (leading digit), so a normal
    `import 04_eval` is illegal. importlib.import_module works because
    we add scripts/ to sys.path via the test scaffolding.
    """
    SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return import_module("04_eval")


def test_gguf_backend_with_grammar_true_loads_grammar(monkeypatch):
    """use_grammar=True calls load_label_grammar once during init."""
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    sentinel = object()
    monkeypatch.setattr(grammar, "load_label_grammar", lambda: sentinel)
    eval_mod = _eval_module()
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=True)
    assert backend._grammar is sentinel


def test_gguf_backend_with_grammar_false_skips_load(monkeypatch):
    """use_grammar=False does NOT call load_label_grammar."""
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    called = []
    monkeypatch.setattr(grammar, "load_label_grammar",
                        lambda: called.append(1) or _FakeGrammar())
    eval_mod = _eval_module()
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=False)
    assert called == []
    assert backend._grammar is None


def test_gguf_backend_grammar_load_failure_is_loud(monkeypatch):
    """Default use_grammar=True must raise if grammar load fails."""
    _inject_fake_llama_cpp(monkeypatch)
    import grammar

    def _raises():
        raise RuntimeError("forced grammar load failure")

    monkeypatch.setattr(grammar, "load_label_grammar", _raises)
    eval_mod = _eval_module()
    with pytest.raises(RuntimeError, match="grammar"):
        eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                              n_gpu_layers=0, use_grammar=True)


def test_gguf_backend_batch_infer_passes_grammar(monkeypatch):
    """When grammar is enabled, create_chat_completion receives grammar=<obj>."""
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    sentinel = object()
    monkeypatch.setattr(grammar, "load_label_grammar", lambda: sentinel)
    eval_mod = _eval_module()
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=True)
    backend.batch_infer(["500 beer"], max_tokens=128)
    assert backend.llm.calls[0]["grammar"] is sentinel


def test_gguf_backend_batch_infer_omits_grammar_when_disabled(monkeypatch):
    """When grammar is disabled, the grammar kwarg must NOT be in the call."""
    _inject_fake_llama_cpp(monkeypatch)
    eval_mod = _eval_module()
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=False)
    backend.batch_infer(["500 beer"], max_tokens=128)
    assert "grammar" not in backend.llm.calls[0]
