# Grammar-Constrained Decoding (Slice 2 of validator roadmap) — Design Spec

**Date:** 2026-05-17
**Status:** Approved, ready for implementation plan
**Scope:** Slice 2 of 6 in the validator + transaction-parser quality roadmap.
**Prerequisite:** Slice 1 (validator + amount_parser) merged on `main`; the multi-provider PR (#1) carrying Slices 1–4 of the multi-provider roadmap is in flight but not required for this work.

---

## 1. Problem

The student Q5_K_M GGUF lands at ~94% JSON-valid on the 50-example README eval and ~72% schema-valid. The biggest single class of failures is the model emitting JSON-shaped output that drifts after the first object — trailing prose, a second object, or unfinished tokens past `max_tokens`. The Gemini provider already enforces structure via `structured_output: true` (Slice 3 of the multi-provider roadmap); the llama.cpp inference paths do not.

This slice adds GBNF grammar-constrained decoding to every llama.cpp call site, default-on, with a `--no-grammar` opt-out for baseline comparisons.

## 2. Solution overview

One new module + three modified call sites + one new test file pair:

```
scripts/
  grammar.py                       (NEW)    build_label_grammar + load_label_grammar
  04_eval.py                       (MODIFY) GgufBackend takes use_grammar=True; --no-grammar flag
  predict_one.py                   (MODIFY) --no-grammar flag; default applies grammar

viewer/
  inference_worker.py              (MODIFY) --no-grammar arg; default applies grammar

tests/
  test_grammar.py                  (NEW)    ~14 cases: pure builder + optional compile
  test_eval_grammar_wiring.py      (NEW)    ~5 cases: GgufBackend plumbing without real GGUF

README.md                          (MODIFY) one-line note on Stage 4/7 grammar default
```

Default behavior changes for GGUF inference: every `create_chat_completion` call now passes a `grammar=<LlamaGrammar>` kwarg unless `--no-grammar` is set.

## 3. Out of scope

- Schema changes (`_lib.CATEGORIES`/`TYPES`/`CURRENCIES` are inputs, not outputs).
- Training-loop changes — teacher/student training don't see grammar.
- API-provider grammar — Gemini already has `structured_output: true` (Slice 3 of multi-provider roadmap); DeepSeek doesn't support GBNF.
- `TransformersBackend` grammar — Unsloth/transformers uses a separate constrained-generation library; out of scope here.
- Per-model grammar variation — one grammar for all GGUFs.
- Stage 7 server.js — playground inherits grammar from `inference_worker.py` defaults; no server-side toggle.
- Eval-set expansion — Slice 3 of the validator roadmap.

## 4. Module: `scripts/grammar.py`

Pure stdlib. Importing this module does NOT pull in `llama_cpp`. Reuses `_lib.CATEGORIES/TYPES/CURRENCIES` as the single source of truth.

### 4.1 Public surface

```python
from _lib import CATEGORIES, TYPES, CURRENCIES

if not CATEGORIES or not TYPES or not CURRENCIES:
    raise ValueError("grammar.py: _lib enums must be non-empty")

def _gbnf_literal(s: str) -> str:
    """Escape a Python string into a GBNF string literal terminal that emits
    JSON `"<s>"`. Handles backslashes and embedded quotes."""

def build_label_grammar() -> str:
    """Return the GBNF text. Pure stdlib."""

_LABEL_GRAMMAR = None    # cached LlamaGrammar instance

def load_label_grammar():
    """Return the cached LlamaGrammar. Lazy llama_cpp import.
    Compile failure raises RuntimeError with a hint about --no-grammar."""
```

### 4.2 Grammar shape

```ebnf
root ::= "{" ws "\"transactions\":" ws "[" ws transactions ws "]" ws "}"
transactions ::= transaction (ws "," ws transaction)*
              | ""

transaction ::= "{"
    ws "\"amount\":" ws number ws ","
    ws "\"currency\":" ws currency ws ","
    ws "\"item\":" ws string ws ","
    ws "\"category\":" ws category ws ","
    ws "\"type\":" ws type
    ws "}"

currency ::= "\"INR\"" | "\"USD\""
type     ::= "\"expense\"" | "\"income\""
category ::= "\"Food & Drinks\"" | "\"Groceries\"" | "\"Travel\"" | "\"Shopping\""
           | "\"Bills\"" | "\"Entertainment\"" | "\"Health\"" | "\"Other\""

string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F]{4}
number ::= "-"? ([0-9] | [1-9][0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws     ::= [ \t\n\r]*
```

Properties:

- **Single top-level object.** `root` consumes exactly one `{...transactions...}`. The grammar has no valid continuation after the top-level closing brace, so llama.cpp terminates constrained generation there. Existing `max_tokens` and `extract_json` extraction remain as defensive backups.
- **Empty `transactions` array allowed.** `transactions ::= transaction (...)* | ""` covers `{"transactions": []}` when input is junk.
- **Fixed field order:** `amount → currency → item → category → type`. Matches the canonical serializer used for constrained output. Downstream JSON parsing is order-independent; no data migration needed.
- **Strict enums.** Each `_lib.CATEGORIES` / `TYPES` / `CURRENCIES` value becomes one alternative in its enum rule. `_gbnf_literal` handles future entries with quotes/backslashes/spaces.
- **`item` free-form.** `string ::= "\"" char* "\""` allows empty `item` (`""`); semantic checks remain downstream in the validator.
- **`ws` permissive.** `[ \t\n\r]*` — won't penalize CRLF or different whitespace styles.

### 4.3 `_gbnf_literal` examples

```python
_gbnf_literal("INR")              == '"\\"INR\\""'
_gbnf_literal("a\\b")              == '"\\"a\\\\\\\\b\\""'
_gbnf_literal('he said "hi"')     == '"\\"he said \\\\\\"hi\\\\\\"\\""'
```

The escape order is: backslash first, then double-quote. The result is a Python string that, when emitted into the GBNF text, produces a terminal that matches the JSON string token.

### 4.4 Failure modes

- `_lib` enums empty at import time → `ValueError` at module load.
- `llama_cpp.LlamaGrammar.from_string` raises (typo, ABI change) → `load_label_grammar` re-raises as `RuntimeError("Failed to load label grammar: <orig>. Use --no-grammar to run unconstrained.")`.
- `llama_cpp` not installed → the `from llama_cpp import LlamaGrammar` line raises `ImportError`. Same `RuntimeError` wrapping applies.

## 5. Integration

### 5.1 `scripts/04_eval.py::GgufBackend`

Constructor gains `use_grammar: bool = True`. Grammar loaded once during `__init__`; the per-call hot path stays unchanged except for one conditional kwarg.

```python
class GgufBackend:
    def __init__(self, gguf_path, *, n_ctx, n_gpu_layers, use_grammar=True):
        # ...existing init unchanged...
        self._grammar = None
        if use_grammar:
            from grammar import load_label_grammar
            self._grammar = load_label_grammar()    # raises if compile fails

    def batch_infer(self, inputs, max_tokens):
        out = []
        for s in inputs:
            t0 = time.perf_counter()
            kwargs = dict(messages=build_messages(s), temperature=0.0,
                          top_p=1.0, max_tokens=max_tokens)
            if self._grammar is not None:
                kwargs["grammar"] = self._grammar
            resp = self.llm.create_chat_completion(**kwargs)
            # ...rest unchanged...
```

`main()` adds:

```python
p.add_argument("--no-grammar", action="store_true",
               help="Disable GBNF grammar-constrained decoding for GGUF inference.")
# ...
backend = GgufBackend(..., use_grammar=not args.no_grammar)
logging.info("Grammar (GGUF backend): %s",
             "enabled" if not args.no_grammar else "disabled")
```

If `--all` mode runs both backends, the log line is emitted only when the GGUF backend is constructed (TransformersBackend is unaffected).

Per-row JSONL output (`eval_results/<name>.jsonl`) gains a `"_grammar": true|false` field on each row so result files self-document.

Sampling parameters (`temperature=0.0`, `top_p=1.0`, `max_tokens=...`) are **unchanged**. Only the `grammar` kwarg is added.

### 5.2 `scripts/predict_one.py`

Adds `--no-grammar`. Grammar import is lazy (joins the existing lazy `from llama_cpp import Llama` pattern at line 29):

```python
p.add_argument("--no-grammar", action="store_true",
               help="Disable GBNF grammar-constrained decoding.")
# ...
from llama_cpp import Llama
llm = Llama(...)
kwargs = dict(messages=build_messages(args.input), temperature=0.0,
              top_p=1.0, max_tokens=args.max_tokens)
if not args.no_grammar:
    from grammar import load_label_grammar
    kwargs["grammar"] = load_label_grammar()
resp = llm.create_chat_completion(**kwargs)

print(f"GRAMMAR: {'off' if args.no_grammar else 'on'}")
print(f"MODEL  : {args.model}")
# ... rest unchanged ...
```

### 5.3 `viewer/inference_worker.py`

Adds `--no-grammar`. Grammar loaded once at startup right after the `Llama(...)` load; attached to every request:

```python
parser.add_argument("--no-grammar", action="store_true",
                    help="Disable GBNF grammar-constrained decoding.")
# ...
grammar_obj = None
if not args.no_grammar:
    from grammar import load_label_grammar
    grammar_obj = load_label_grammar()
log(f"grammar: {'enabled' if grammar_obj is not None else 'disabled'}")
send({"ready": True, "model": model_path.name})

for raw in sys.stdin:
    # ...
    kwargs = dict(messages=build_messages(input_text), temperature=0.0,
                  top_p=1.0, max_tokens=max_tokens)
    if grammar_obj is not None:
        kwargs["grammar"] = grammar_obj
    resp = llm.create_chat_completion(**kwargs)
```

`scripts/` is already on `sys.path` (line 25), so `from grammar import load_label_grammar` works.

The `[worker] grammar: enabled|disabled` stderr line surfaces the choice in the playground's worker log.

### 5.4 `viewer/server.js`

Unchanged in this slice. The playground uses grammar by default because `inference_worker.py` defaults to grammar-on. There is no server-side toggle. To run unconstrained manually, invoke `inference_worker.py` with `--no-grammar`.

### 5.5 Thread/concurrency note

The `LlamaGrammar` object is reused within a single backend/worker instance. Current GGUF eval and worker paths are sequential. No concurrent grammar use today; future concurrent code would need to revisit if llama.cpp's grammar object proves stateful per-call.

## 6. Testing

Net new tests: ~19 (14 in `test_grammar.py` + 5 in `test_eval_grammar_wiring.py`). Target ~353 passing; treat as smoke signal, not a hard blocker.

### 6.1 `tests/test_grammar.py`

**Pure-stdlib helpers and presence:**

```python
def _json_key_terminal(key: str) -> str:
    return f'"\\"{key}\\":"'

def test_gbnf_literal_wraps_json_string_quotes():
    from grammar import _gbnf_literal
    assert _gbnf_literal("INR") == '"\\"INR\\""'

def test_gbnf_literal_escapes_backslash():
    from grammar import _gbnf_literal
    assert _gbnf_literal("a\\b") == '"\\"a\\\\\\\\b\\""'

def test_gbnf_literal_escapes_embedded_quote():
    from grammar import _gbnf_literal
    assert _gbnf_literal('he said "hi"') == '"\\"he said \\\\\\"hi\\\\\\"\\""'

def test_build_label_grammar_contains_every_category():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CATEGORIES
    g = build_label_grammar()
    for cat in CATEGORIES:
        assert _gbnf_literal(cat) in g

def test_build_label_grammar_contains_every_type():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import TYPES
    g = build_label_grammar()
    for t in TYPES:
        assert _gbnf_literal(t) in g

def test_build_label_grammar_contains_every_currency():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CURRENCIES
    g = build_label_grammar()
    for c in CURRENCIES:
        assert _gbnf_literal(c) in g

def test_build_label_grammar_has_required_top_level_shape():
    from grammar import build_label_grammar
    g = build_label_grammar()
    assert 'root ::= "{"' in g
    assert _json_key_terminal("transactions") in g

def test_build_label_grammar_has_required_transaction_keys():
    from grammar import build_label_grammar
    g = build_label_grammar()
    for key in ("amount", "currency", "item", "category", "type"):
        assert _json_key_terminal(key) in g

def test_build_label_grammar_uses_crlf_safe_whitespace():
    from grammar import build_label_grammar
    g = build_label_grammar()
    assert "[ \\t\\n\\r]*" in g
```

**Enum-rule uniqueness:** parse each rule body and assert every value appears exactly once.

```python
def _rule_body(grammar: str, name: str) -> str:
    lines = grammar.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(f"{name} ::="))
    body = [lines[start]]
    for l in lines[start + 1:]:
        if "::=" in l:
            break
        body.append(l)
    return "\n".join(body)

def test_category_enum_has_no_duplicate_alternatives():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CATEGORIES
    body = _rule_body(build_label_grammar(), "category")
    for cat in CATEGORIES:
        assert body.count(_gbnf_literal(cat)) == 1

def test_type_enum_has_no_duplicate_alternatives():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import TYPES
    body = _rule_body(build_label_grammar(), "type")
    for t in TYPES:
        assert body.count(_gbnf_literal(t)) == 1

def test_currency_enum_has_no_duplicate_alternatives():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CURRENCIES
    body = _rule_body(build_label_grammar(), "currency")
    for c in CURRENCIES:
        assert body.count(_gbnf_literal(c)) == 1
```

**Module load-time validation:**

```python
def test_module_raises_on_empty_enum(monkeypatch):
    """If _lib.CATEGORIES ever becomes empty, grammar import must fail loud."""
    import importlib, sys, _lib
    monkeypatch.setattr(_lib, "CATEGORIES", [])
    sys.modules.pop("grammar", None)
    try:
        with pytest.raises(ValueError, match="non-empty"):
            importlib.import_module("grammar")
    finally:
        sys.modules.pop("grammar", None)
```

**Compile-step and loud-failure tests (use `pytest.importorskip`):**

```python
def test_load_label_grammar_compiles(monkeypatch):
    pytest.importorskip("llama_cpp")
    import grammar
    monkeypatch.setattr(grammar, "_LABEL_GRAMMAR", None)    # bust cache cleanly
    g1 = grammar.load_label_grammar()
    g2 = grammar.load_label_grammar()
    assert g1 is g2    # cache works

def test_load_label_grammar_failure_is_loud(monkeypatch):
    pytest.importorskip("llama_cpp")
    import grammar as G
    monkeypatch.setattr(G, "_LABEL_GRAMMAR", None)
    monkeypatch.setattr(
        "llama_cpp.LlamaGrammar.from_string",
        lambda _s: (_ for _ in ()).throw(ValueError("forced fail")),
    )
    with pytest.raises(RuntimeError, match="--no-grammar"):
        G.load_label_grammar()
```

### 6.2 `tests/test_eval_grammar_wiring.py`

Tests `GgufBackend` plumbing without real GGUF I/O. Injects a fake `llama_cpp` module so tests run even if `llama-cpp-python` isn't installed.

```python
import sys, types
import pytest
from pathlib import Path


class _FakeLlama:
    def __init__(self, *args, **kwargs):
        self.calls = []
    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": '{"transactions":[]}'}}],
                "usage": {"completion_tokens": 3}}


class _FakeGrammar:
    pass


def _inject_fake_llama_cpp(monkeypatch):
    fake = types.SimpleNamespace(Llama=_FakeLlama, LlamaGrammar=_FakeGrammar)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)


def test_gguf_backend_with_grammar_true_loads_grammar(monkeypatch):
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    sentinel = object()
    monkeypatch.setattr(grammar, "load_label_grammar", lambda: sentinel)
    from importlib import import_module
    eval_mod = import_module("04_eval")
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=True)
    assert backend._grammar is sentinel


def test_gguf_backend_with_grammar_false_skips_load(monkeypatch):
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    called = []
    monkeypatch.setattr(grammar, "load_label_grammar",
                        lambda: called.append(1) or _FakeGrammar())
    from importlib import import_module
    eval_mod = import_module("04_eval")
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=False)
    assert called == []
    assert backend._grammar is None


def test_gguf_backend_grammar_load_failure_is_loud(monkeypatch):
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    def _raises():
        raise RuntimeError("forced grammar load failure")
    monkeypatch.setattr(grammar, "load_label_grammar", _raises)
    from importlib import import_module
    eval_mod = import_module("04_eval")
    with pytest.raises(RuntimeError, match="grammar"):
        eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                              n_gpu_layers=0, use_grammar=True)


def test_gguf_backend_batch_infer_passes_grammar(monkeypatch):
    """When grammar is enabled, create_chat_completion gets the grammar kwarg."""
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    sentinel = object()
    monkeypatch.setattr(grammar, "load_label_grammar", lambda: sentinel)
    from importlib import import_module
    eval_mod = import_module("04_eval")
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=True)
    backend.batch_infer(["500 beer"], max_tokens=128)
    assert backend.llm.calls[0]["grammar"] is sentinel


def test_gguf_backend_batch_infer_omits_grammar_when_disabled(monkeypatch):
    _inject_fake_llama_cpp(monkeypatch)
    from importlib import import_module
    eval_mod = import_module("04_eval")
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=False)
    backend.batch_infer(["500 beer"], max_tokens=128)
    assert "grammar" not in backend.llm.calls[0]
```

### 6.3 No new E2E grammar tests

Real-GGUF inference tests are out of scope for this slice — Slice 3 (expanded eval set) is the right slice for that. Slice 2's responsibility ends at "grammar is wired correctly into every call site" plus manual smokes.

### 6.4 No `predict_one.py` / `inference_worker.py` unit tests

Both are thin CLI wrappers. Coverage comes from manual smokes. The grammar path through them is identical to `GgufBackend`'s and already tested.

## 7. Manual smokes

Five smokes. Linux/WSL is the supported environment (Windows GPU build of `llama-cpp-python` is fragile).

**Smoke 1 — `predict_one.py` with grammar (default):**
```bash
python scripts/predict_one.py \
    --model models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf \
    "500 rs on beer 50 rs on candy"
```
Expected: `GRAMMAR: on`; `RAW` is a single `{"transactions":[...]}` with two transactions; `PARSED` non-null.

**Smoke 2 — `predict_one.py --no-grammar`:**
```bash
python scripts/predict_one.py --no-grammar \
    --model models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf \
    "500 rs on beer 50 rs on candy"
```
Expected: `GRAMMAR: off`; output may be less constrained. This is a baseline comparison, not proof that grammar changed the output for this one prompt.

**Smoke 3 — `04_eval.py` with grammar:**
```bash
python scripts/04_eval.py --gguf models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf
```
Expected: log line `Grammar (GGUF backend): enabled`; `eval_results/<name>.jsonl` rows include `"_grammar": true`; `json_valid` is at or near 100%.

**Smoke 4 — `04_eval.py --no-grammar`:**
```bash
python scripts/04_eval.py --gguf models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf --no-grammar
```
Expected: log line `Grammar (GGUF backend): disabled`; rows include `"_grammar": false`; `json_valid` shows the current unconstrained baseline for this model. Compare against smoke 3 to quantify lift.

**Smoke 5 — `inference_worker.py` startup line:**
```bash
python viewer/inference_worker.py \
    --model models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf < /dev/null 2>&1 | head -5
```
Expected: stderr includes `[worker] grammar: enabled`. With `--no-grammar`: `[worker] grammar: disabled`. This smoke only verifies startup/logging, not request handling — stdin EOF may shut the worker down right after the ready/log lines.

### 7.1 Latency expectation

Grammar enforcement adds per-token logit masking. Expected overhead is likely small for this grammar, but not gated. Smoke 3 vs Smoke 4 is the measurement.

## 8. Documentation

- **`README.md`** — one new line in the Stage 4 / Stage 7 area: *"By default, GGUF-based inference (Stages 4, 7, and `predict_one.py`) constrains output via a GBNF grammar derived from the validator's enum constants. Pass `--no-grammar` to disable for baseline comparisons."*
- **No new section.** The grammar is an implementation detail; it doesn't change the user-facing pipeline.
- **No `docs/provider_config.md` change** — that file documents the multi-provider config, which doesn't touch GBNF.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `LlamaGrammar.from_string` fails (typo in builder, llama.cpp ABI change) | Loud `RuntimeError` with `--no-grammar` escape hatch. Compile-step test catches typos. |
| Grammar over-constrains and the model can't sample anything valid → empty output | `max_tokens` still applies; output may be empty or truncated. Validator catches malformed output downstream. Smoke 3 verifies normal inputs complete. |
| Adding a category in `_lib.py` silently drifts the grammar | Doesn't drift — grammar is rebuilt from `_lib` at every `load_label_grammar()`. Cache is per-process. |
| Windows playground worker can't find `scripts/` | Already on `sys.path` via `viewer/inference_worker.py:25`. Verified. |
| `llama-cpp-python` version-pinning incompatibility | `LlamaGrammar.from_string` has been stable across 0.2.x+. If a future SDK breaks this, it's an environment fix, not a grammar-decoding slice concern. |
| Concurrent use of one `LlamaGrammar` object | Current paths are sequential. Future concurrent callers would need to revisit. |

## 10. Single-commit landing

Plan will target a single additive commit on `feat/grammar-decoding` from `main` HEAD. Spec + amendments live on the branch as their own commits; the implementation lands as one additive commit.

Files staged: `scripts/grammar.py`, `scripts/04_eval.py`, `scripts/predict_one.py`, `viewer/inference_worker.py`, `tests/test_grammar.py`, `tests/test_eval_grammar_wiring.py`, `README.md`.

Files NEVER staged: `data/distill/*`, `eval_results/*`, `.coverage`, `reports/`, `htmlcov/`.
