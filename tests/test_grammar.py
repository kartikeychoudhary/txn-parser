"""Unit tests for scripts/grammar.py.

Covers:
- _gbnf_literal escape semantics
- build_label_grammar presence + uniqueness checks (pure stdlib, no llama_cpp)
- load_label_grammar caching + loud-failure behavior (requires llama_cpp;
  uses pytest.importorskip)
- module load-time validation (raises if _lib enums are empty)
"""
import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _json_key_terminal(key: str) -> str:
    """Return the GBNF terminal string that emits the JSON key plus colon,
    e.g. _json_key_terminal('amount') == '"\\"amount\\":"'."""
    return f'"\\"{key}\\":"'


# ---------------------------------------------------------------------------
# _gbnf_literal
# ---------------------------------------------------------------------------


def test_gbnf_literal_wraps_json_string_quotes():
    """_gbnf_literal('INR') must produce the GBNF terminal "\"INR\"".

    Python repr of that terminal is '"\\"INR\\""'.
    """
    from grammar import _gbnf_literal
    assert _gbnf_literal("INR") == '"\\"INR\\""'


def test_gbnf_literal_escapes_backslash():
    """_gbnf_literal('a\\b') must produce GBNF terminal "\"a\\\\b\"".

    That is: in the GBNF text we want a literal backslash twice (so the
    GBNF parser reads one backslash). In Python repr, that's '"\\"a\\\\\\\\b\\""'.
    """
    from grammar import _gbnf_literal
    assert _gbnf_literal("a\\b") == '"\\"a\\\\\\\\b\\""'


def test_gbnf_literal_escapes_embedded_quote():
    """_gbnf_literal('he said \"hi\"') must produce GBNF terminal
    "\"he said \\\"hi\\\"\"" (each embedded quote becomes \\\")."""
    from grammar import _gbnf_literal
    assert _gbnf_literal('he said "hi"') == '"\\"he said \\\\\\"hi\\\\\\"\\""'


# ---------------------------------------------------------------------------
# build_label_grammar — presence
# ---------------------------------------------------------------------------


def test_build_label_grammar_contains_every_category():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CATEGORIES
    g = build_label_grammar()
    for cat in CATEGORIES:
        assert _gbnf_literal(cat) in g, f"missing category literal for {cat!r}"


def test_build_label_grammar_contains_every_type():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import TYPES
    g = build_label_grammar()
    for t in TYPES:
        assert _gbnf_literal(t) in g, f"missing type literal for {t!r}"


def test_build_label_grammar_contains_every_currency():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CURRENCIES
    g = build_label_grammar()
    for c in CURRENCIES:
        assert _gbnf_literal(c) in g, f"missing currency literal for {c!r}"


def test_build_label_grammar_has_required_top_level_shape():
    from grammar import build_label_grammar
    g = build_label_grammar()
    assert 'root ::= "{"' in g
    assert _json_key_terminal("transactions") in g


def test_build_label_grammar_has_required_transaction_keys():
    from grammar import build_label_grammar
    g = build_label_grammar()
    for key in ("amount", "currency", "item", "category", "type"):
        assert _json_key_terminal(key) in g, f"missing key terminal for {key!r}"


def test_build_label_grammar_uses_crlf_safe_whitespace():
    from grammar import build_label_grammar
    g = build_label_grammar()
    # `ws` rule must include all four whitespace chars including \r.
    assert "[ \\t\\n\\r]*" in g


# ---------------------------------------------------------------------------
# build_label_grammar — enum uniqueness
# ---------------------------------------------------------------------------


def _rule_body(grammar: str, name: str) -> str:
    """Return text from `<name> ::=` until the next `<other> ::=` line.

    Handles multi-line rule bodies (e.g., `category ::= "A" | "B"\n           | "C"`).
    """
    lines = grammar.splitlines()
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith(f"{name} ::="))
    body = [lines[start]]
    for l in lines[start + 1:]:
        # New rule starts when a line begins with `<word> ::=` at column 0
        # (no leading whitespace).
        if l and not l.startswith((" ", "\t")) and "::=" in l:
            break
        body.append(l)
    return "\n".join(body)


def test_category_enum_has_no_duplicate_alternatives():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CATEGORIES
    body = _rule_body(build_label_grammar(), "category")
    for cat in CATEGORIES:
        assert body.count(_gbnf_literal(cat)) == 1, \
            f"category {cat!r} appears != 1 time in its rule body"


def test_type_enum_has_no_duplicate_alternatives():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import TYPES
    body = _rule_body(build_label_grammar(), "type")
    for t in TYPES:
        assert body.count(_gbnf_literal(t)) == 1, \
            f"type {t!r} appears != 1 time in its rule body"


def test_currency_enum_has_no_duplicate_alternatives():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CURRENCIES
    body = _rule_body(build_label_grammar(), "currency")
    for c in CURRENCIES:
        assert body.count(_gbnf_literal(c)) == 1, \
            f"currency {c!r} appears != 1 time in its rule body"


# ---------------------------------------------------------------------------
# module load-time validation
# ---------------------------------------------------------------------------


def test_module_raises_on_empty_enum(monkeypatch):
    """If _lib.CATEGORIES ever becomes empty, importing grammar must fail loud."""
    import _lib
    monkeypatch.setattr(_lib, "CATEGORIES", [])
    sys.modules.pop("grammar", None)
    try:
        with pytest.raises(ValueError, match="non-empty"):
            importlib.import_module("grammar")
    finally:
        sys.modules.pop("grammar", None)


# ---------------------------------------------------------------------------
# load_label_grammar — requires llama_cpp
# ---------------------------------------------------------------------------


def test_load_label_grammar_compiles_and_caches(monkeypatch):
    pytest.importorskip("llama_cpp")
    import grammar
    monkeypatch.setattr(grammar, "_LABEL_GRAMMAR", None)
    g1 = grammar.load_label_grammar()
    g2 = grammar.load_label_grammar()
    assert g1 is g2   # cache returns same object


def test_load_label_grammar_failure_is_loud(monkeypatch):
    """Skipped when llama_cpp is missing: this test patches the real
    LlamaGrammar class to force a compile failure, which requires the
    real module to exist."""
    pytest.importorskip("llama_cpp")
    import grammar
    monkeypatch.setattr(grammar, "_LABEL_GRAMMAR", None)

    def _explode(_s):
        raise ValueError("forced compile failure")

    import llama_cpp
    monkeypatch.setattr(llama_cpp.LlamaGrammar, "from_string", _explode)
    with pytest.raises(RuntimeError, match="--no-grammar"):
        grammar.load_label_grammar()


# ---------------------------------------------------------------------------
# load_label_grammar — coverage via injected fake llama_cpp (runs anywhere)
# ---------------------------------------------------------------------------


class _FakeLlamaGrammar:
    @classmethod
    def from_string(cls, _grammar: str):
        return cls()


def _inject_fake_llama_cpp(monkeypatch):
    import types
    fake = types.SimpleNamespace(LlamaGrammar=_FakeLlamaGrammar)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)


def test_load_label_grammar_with_fake_llama_cpp_compiles_and_caches(monkeypatch):
    """Runs anywhere (no real llama_cpp required) via injected fake module."""
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    monkeypatch.setattr(grammar, "_LABEL_GRAMMAR", None)
    g1 = grammar.load_label_grammar()
    g2 = grammar.load_label_grammar()
    assert g1 is g2
    assert isinstance(g1, _FakeLlamaGrammar)


def test_load_label_grammar_with_fake_failure_is_loud(monkeypatch):
    """Runs anywhere: force the fake LlamaGrammar.from_string to raise,
    confirm the RuntimeError wrapping fires."""
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    monkeypatch.setattr(grammar, "_LABEL_GRAMMAR", None)

    def _explode(_s):
        raise ValueError("forced fake compile failure")

    monkeypatch.setattr(_FakeLlamaGrammar, "from_string", _explode)
    with pytest.raises(RuntimeError, match="--no-grammar"):
        grammar.load_label_grammar()
