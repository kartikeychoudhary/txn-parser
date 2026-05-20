"""GBNF grammar for transaction-parser output, derived at runtime from
_lib.CATEGORIES/TYPES/CURRENCIES.

Pure-stdlib at import time. The lazy `load_label_grammar()` is the only
function that pulls in `llama_cpp`; it caches the compiled grammar
module-level so it's built exactly once per process.

Spec: docs/superpowers/specs/2026-05-17-grammar-constrained-decoding-design.md
"""
from __future__ import annotations

from _lib import CATEGORIES, TYPES, CURRENCIES


if not CATEGORIES or not TYPES or not CURRENCIES:
    raise ValueError(
        "grammar.py: _lib enums (CATEGORIES, TYPES, CURRENCIES) must be non-empty"
    )


def _gbnf_literal(s: str) -> str:
    """Build a GBNF terminal string that emits the JSON token `"<s>"`.

    Escaping order matters: backslash first (so we don't escape the
    backslashes we add for quotes), then double-quote.

    Returns a Python string. When that string is rendered into the GBNF
    text, it produces a terminal like `"\\"INR\\""`, which is how GBNF
    spells the literal three-char sequence `"INR"`.
    """
    # Backslashes: one backslash becomes 4 (for GBNF + JSON escaping)
    # Quotes: one quote becomes \" (2 chars: backslash + quote), but the backslash
    #   also needs GBNF escaping, so it becomes \\\" (3 chars in Python source = 2 in actual string)
    escaped = s.replace("\\", "\\\\\\\\").replace('"', '\\\\\\"')
    return '"\\"' + escaped + '\\""'


def _enum_rule(name: str, values: list[str]) -> str:
    """Return a GBNF line like:
        category ::= "\\"Food\\"" | "\\"Drinks\\"" | ...
    """
    alts = " | ".join(_gbnf_literal(v) for v in values)
    return f"{name} ::= {alts}"


def build_label_grammar() -> str:
    """Return the full GBNF text. Pure string building."""
    # Standard JSON-string and JSON-number rules, adapted from llama.cpp's
    # grammars/json.gbnf reference set.
    return "\n".join([
        'root ::= "{" ws "\\"transactions\\":" ws "[" ws transactions ws "]" ws "}"',
        'transactions ::= transaction (ws "," ws transaction)* | ""',
        '',
        'transaction ::= "{"',
        '    ws "\\"amount\\":" ws number ws ","',
        '    ws "\\"currency\\":" ws currency ws ","',
        '    ws "\\"item\\":" ws string ws ","',
        '    ws "\\"category\\":" ws category ws ","',
        '    ws "\\"type\\":" ws type',
        '    ws "}"',
        '',
        _enum_rule("currency", list(CURRENCIES)),
        _enum_rule("type", list(TYPES)),
        _enum_rule("category", list(CATEGORIES)),
        '',
        'string ::= "\\"" char* "\\""',
        'char ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt] | "\\\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]',
        'number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
        'ws ::= [ \\t\\n\\r]*',
        '',
    ])


_LABEL_GRAMMAR = None    # cached LlamaGrammar instance


def load_label_grammar():
    """Return the cached LlamaGrammar. Lazy import of `llama_cpp`.

    Failure to import or compile raises RuntimeError with a hint about
    --no-grammar so the operator knows the escape hatch exists. No silent
    fallback — that would defeat the entire point of grammar-on-by-default.
    """
    global _LABEL_GRAMMAR
    if _LABEL_GRAMMAR is not None:
        return _LABEL_GRAMMAR
    try:
        from llama_cpp import LlamaGrammar
        _LABEL_GRAMMAR = LlamaGrammar.from_string(build_label_grammar())
    except Exception as e:    # noqa: BLE001
        raise RuntimeError(
            f"Failed to load label grammar: {e}. "
            "Use --no-grammar to run unconstrained."
        ) from e
    return _LABEL_GRAMMAR
