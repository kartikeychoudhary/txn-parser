# Multi-Provider Generation — Slice 1 Design Spec

**Date:** 2026-05-16
**Status:** Approved, ready for implementation plan
**Scope:** Slice 1 of 4 in the multi-provider synthetic-data roadmap.
**Prerequisite:** The validator + amount_parser slice is already merged (commits `9f253b7`, `1bf3a14`, `7b05da3` on `main`).

---

## 1. Problem

Stage 5 currently calls one provider (DeepSeek) for synthetic inputs and one local fine-tuned teacher for labels. We want to widen this to N configurable providers with weighted scheduling, run them concurrently, and gate every label through the new semantic validator before it lands in `data/distill/train.jsonl`. The full program spans four slices:

1. **Slice 1 (this spec):** provider abstraction + config schema + dry-run + validator probe. Scaffolding only, no real provider calls.
2. **Slice 2:** multi-provider input generation (real API calls, threaded execution, global dedupe).
3. **Slice 3:** validator-gated output labeling (candidate generation, scoring, structured output).
4. **Slice 4:** metrics, cost tracking, per-provider analysis.

Splitting this way lets every component land with unit tests and a runnable smoke before any provider SDK is added to requirements or any real API budget is touched.

## 2. Solution overview (Slice 1 only)

New pure-Python modules under `scripts/`, pure stdlib dependencies:

```
scripts/
  llm_providers.py            (NEW)  Protocol + FakeProvider + UnimplementedProvider + create_provider
  generation_config.py        (NEW)  dataclasses + load_generation_config
  generation_orchestrator.py  (NEW)  allocate_quota, RoundRobinScheduler, render_dry_run
  probe_validator.py          (NEW)  standalone CLI: JSONL pairs → validator report
  05_generate_distillation_data.py  (MODIFY) +--provider-config, +--dry-run-quota, +--multi-provider

configs/
  test_providers.json         (NEW)  fake-only 10-input config for smoke
  example_providers.json      (NEW)  realistic config with unimplemented provider types

docs/
  provider_config.md          (NEW)  human-readable companion explaining the JSON schema

tests/
  test_generation_config.py   (NEW)
  test_provider_scheduler.py  (NEW)
  test_fake_provider.py       (NEW)
  test_probe_validator.py     (NEW)
  test_stage5_flag_contract.py (NEW)
  fixtures/
    fake_inputs.jsonl         (NEW)  20 rows of canned input strings
    fake_labels.jsonl         (NEW)  13 rows of (input, output) + raw_output failure rows
    probe_examples.jsonl      (NEW)  11 rows hitting every validator error code
```

Default behavior is unchanged: invocations of `scripts/05_generate_distillation_data.py` that don't pass `--provider-config` run the existing Stage 5 flow identically.

**Out of scope for Slice 1** (deferred to later slices):

- Any real DeepSeek / Gemini API call from code or tests.
- Adding `google-generativeai` or `openai` to `requirements-*.txt`.
- The multi-provider input *generation* loop (Slice 2).
- The multi-provider output *labeling* loop (Slice 3).
- Retry-with-stricter-prompt repair logic (Slice 3).
- Metrics, latency tracking, cost estimation (Slice 4).

---

## 3. Module: `scripts/llm_providers.py`

### 3.1 Public API

```python
from typing import Protocol


class LLMProvider(Protocol):
    """Minimal provider surface. All methods must be thread-safe — callers
    will run them concurrently via ThreadPoolExecutor in slices 2 and 3."""
    name: str
    provider_type: str   # "fake" | "deepseek" | "gemini" | "local_teacher"

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        """Return up to n raw input strings, one per example."""
        ...

    def generate_label(self, input_text: str) -> str:
        """Return raw provider text. May be invalid JSON."""
        ...


class ProviderError(RuntimeError):
    """Provider-level failures (API down, quota exceeded, bad fixture).
    Distinct from validator/JSON failures."""
```

### 3.2 `FakeProvider`

```python
class FakeProvider:
    """In-process provider backed by JSONL fixture files.

    Fixture contracts:
      inputs_path JSONL: each row is {"input": "<string>"} (extra keys ignored).
      labels_path JSONL: each row is one of:
        {"input": "...", "output": {...}}         valid label, serialized via json.dumps
        {"input": "...", "raw_output": "..."}     literal text returned (may be bad JSON)
        Extra keys ignored.

    Both files load once at __init__:
      - Blank lines are ignored.
      - Invalid JSON raises ProviderError with line number.
      - Missing required field ("input") raises ProviderError with line number.

    Determinism:
      - generate_inputs ignores `prompt`; returns deterministic cursor slices.
      - Cursor advances per call and wraps around modulo fixture length:
          fixtures=[a,b,c]; generate_inputs(n=5) -> [a,b,c,a,b]
          next generate_inputs(n=2) -> [c,a]
      - Cursor protected by threading.Lock for concurrent callers.
      - generate_label looks up input in an immutable map built at __init__;
        no lock needed.

    Label resolution rule:
      1. If row has 'raw_output': return row['raw_output'] verbatim.
      2. Else if row has 'output': return json.dumps(row['output'], separators=(',', ':')).
      3. Else: return miss_payload (default "").

    If input_text is not found in labels: return miss_payload.
    """

    provider_type = "fake"

    def __init__(
        self,
        name: str,
        *,
        inputs_path: Path | None = None,
        labels_path: Path | None = None,
        seed: int = 0,                  # reserved; FakeProvider in Slice 1 is purely
                                        # cursor-deterministic and does not randomize.
                                        # Slice 2/3 may use it if a randomized fake is needed.
        miss_payload: str = "",
    ) -> None: ...

    def generate_inputs(self, prompt: str, n: int) -> list[str]: ...
    def generate_label(self, input_text: str) -> str: ...
```

### 3.3 `UnimplementedProvider`

```python
class UnimplementedProvider:
    """Placeholder returned by create_provider for type in
    {deepseek, gemini, local_teacher}. Construction succeeds (so configs
    parse and dry-run works); any method call raises NotImplementedError
    with a pointer to the right slice. NO SDK imports."""

    def __init__(self, name: str, provider_type: str, model: str | None = None) -> None:
        self.name = name
        self.provider_type = provider_type
        self.model = model

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        raise NotImplementedError(
            f"{self.provider_type!r} provider {self.name!r}: real execution lands in Slice 2."
        )

    def generate_label(self, input_text: str) -> str:
        raise NotImplementedError(
            f"{self.provider_type!r} provider {self.name!r}: real execution lands in Slice 3."
        )
```

### 3.4 Factory

```python
def create_provider(cfg: "ProviderConfig") -> LLMProvider:
    """Map ProviderConfig.provider_type → concrete provider.
    Only 'fake' is callable in Slice 1; other types return
    UnimplementedProvider so dry-run-quota and config parsing work."""
    if cfg.provider_type == "fake":
        # Loader (generation_config.py §4.5) stores resolved absolute paths.
        return FakeProvider(
            name=cfg.name,
            inputs_path=Path(cfg.fixture_inputs) if cfg.fixture_inputs else None,
            labels_path=Path(cfg.fixture_labels) if cfg.fixture_labels else None,
            seed=cfg.seed or 0,
        )
    if cfg.provider_type in {"deepseek", "gemini", "local_teacher"}:
        return UnimplementedProvider(
            name=cfg.name, provider_type=cfg.provider_type, model=cfg.model,
        )
    raise ValueError(  # defensive — generation_config.py already rejects unknown types
        f"Unknown provider type {cfg.provider_type!r} for provider {cfg.name!r}"
    )
```

### 3.5 No SDK imports

`scripts/llm_providers.py` imports only from `pathlib`, `threading`, `json`, `typing`, `dataclasses`, and (forward-string-typed) `generation_config`. No `google.generativeai`, no `openai`, no `requests`, no `unsloth`. Slice 2/3 will add those inside their concrete provider implementations.

---

## 4. Module: `scripts/generation_config.py`

### 4.1 Dataclasses

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    provider_type: str                       # JSON key "type"
    weight: int = 1                          # positive integer
    threads: int = 1                         # positive integer
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int = 3
    structured_output: bool = False
    seed: int | None = None
    fixture_inputs: str | None = None        # FakeProvider only — path
    fixture_labels: str | None = None        # FakeProvider only — path


@dataclass(frozen=True)
class InputGenerationConfig:
    enabled: bool = True
    target_inputs: int = 0
    batch_size: int = 100
    dedupe: bool = True
    providers: tuple[ProviderConfig, ...] = ()


@dataclass(frozen=True)
class OutputGenerationConfig:
    enabled: bool = True
    label_attempts_per_input: int = 1
    selection_policy: str = "first_valid_then_score"
    providers: tuple[ProviderConfig, ...] = ()
    provider_priority: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationGateConfig:
    schema: bool = True
    semantic_validator: bool = True
    reject_invalid: bool = True
    retry_invalid_with_stricter_prompt: bool = False
    max_repair_attempts: int = 0


@dataclass(frozen=True)
class RateLimitsConfig:
    global_max_workers: int = 8
    write_flush_every: int = 50


@dataclass(frozen=True)
class GenerationConfig:
    version: int
    input_generation: InputGenerationConfig
    output_generation: OutputGenerationConfig
    validation: ValidationGateConfig
    rate_limits: RateLimitsConfig
    source_path: str | None = None           # set by loader at construction time
```

`source_path` is populated during construction by `load_generation_config(...)`. The dataclass is frozen — no post-construction mutation.

### 4.2 Loader

```python
import logging

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised with field path in the message (e.g. 'input_generation.providers[1].weight')."""


_VALID_PROVIDER_TYPES = {"fake", "deepseek", "gemini", "local_teacher"}
_VALID_SELECTION_POLICIES = {"first_valid_then_score"}


def load_generation_config(path: Path | str) -> GenerationConfig: ...
```

### 4.3 Validation rules — all raise `ConfigError`

| Field path | Rule |
|---|---|
| `version` | int, must equal `1` if present; missing → treated as `1` |
| `input_generation.target_inputs` | int ≥ 0 |
| `input_generation.batch_size` | int ≥ 1 |
| `input_generation.providers` | non-empty when `enabled=true` AND `target_inputs ≥ 1` |
| `output_generation.providers` | non-empty when `enabled=true` |
| `output_generation.label_attempts_per_input` | int ≥ 1 |
| `output_generation.selection_policy` | must be in `_VALID_SELECTION_POLICIES` |
| `output_generation.provider_priority[i]` | each must reference a declared `output_generation.providers[*].name` |
| `<phase>.providers[i].name` | non-empty string |
| `<phase>.providers[i].name` | unique within its phase |
| `<phase>.providers[i].type` | must be in `_VALID_PROVIDER_TYPES` |
| `<phase>.providers[i].weight` | int ≥ 1 |
| `<phase>.providers[i].threads` | int ≥ 1 |
| `<phase>.providers[i].temperature` | null or `float ≥ 0` |
| `<phase>.providers[i].max_tokens` | null or `int ≥ 1` |
| `<phase>.providers[i].max_retries` | int ≥ 0 |
| `providers[i] where type='fake'` | requires `fixture_inputs` when used in input phase, `fixture_labels` when used in output phase |
| `providers[i] where type in {deepseek, gemini}` | requires non-null `model` |
| `validation.max_repair_attempts` | int ≥ 0 |
| `rate_limits.global_max_workers` | int ≥ 1 |
| `rate_limits.write_flush_every` | int ≥ 1 |

When `enabled=false` for a phase, all `providers` and `target_inputs` rules for that phase are skipped — empty lists and zero values are allowed.

### 4.4 Validation rules that produce warnings (via `logger.warning`)

- Unknown top-level keys in the JSON dict.
- Unknown keys inside any `ProviderConfig` dict.
- `structured_output=true` on a provider where `provider_type != "gemini"`.

Warnings do not block load.

### 4.5 Fixture path resolution

For any provider with `fixture_inputs` or `fixture_labels`:

- Absolute paths used as-is.
- Relative paths resolved **relative to the config file's directory** (`Path(config_path).parent / fixture_path`).
- Loader asserts the resolved path exists AND is a file; missing → `ConfigError("fixture not found at <path>")`.
- The resolved absolute path is what the loader passes through to `ProviderConfig.fixture_inputs/fixture_labels` (i.e., the dataclass stores the resolved path, not the original relative string).

Example: `configs/test_providers.json` containing `"fixture_inputs": "../tests/fixtures/fake_inputs.jsonl"` resolves to `<repo>/tests/fixtures/fake_inputs.jsonl`.

### 4.6 Example minimal valid config (`configs/test_providers.json`)

```json
{
  "version": 1,
  "input_generation": {
    "enabled": true,
    "target_inputs": 10,
    "batch_size": 5,
    "providers": [
      {"name": "fake_a", "type": "fake", "weight": 60, "threads": 1,
       "fixture_inputs": "../tests/fixtures/fake_inputs.jsonl"},
      {"name": "fake_b", "type": "fake", "weight": 40, "threads": 1,
       "fixture_inputs": "../tests/fixtures/fake_inputs.jsonl"}
    ]
  },
  "output_generation": {
    "enabled": true,
    "label_attempts_per_input": 1,
    "providers": [
      {"name": "fake_a", "type": "fake", "weight": 50, "threads": 1,
       "fixture_labels": "../tests/fixtures/fake_labels.jsonl"},
      {"name": "fake_b", "type": "fake", "weight": 50, "threads": 1,
       "fixture_labels": "../tests/fixtures/fake_labels.jsonl"}
    ]
  },
  "validation": {"schema": true, "semantic_validator": true,
                 "reject_invalid": true,
                 "retry_invalid_with_stricter_prompt": false,
                 "max_repair_attempts": 0},
  "rate_limits": {"global_max_workers": 4, "write_flush_every": 10}
}
```

`configs/example_providers.json` mirrors this shape with `type: "deepseek"` / `type: "gemini"` / `type: "local_teacher"`, plus realistic `model`/`temperature`/`max_tokens` values. It loads cleanly in Slice 1 because `create_provider` returns `UnimplementedProvider` for those types.

---

## 5. Module: `scripts/generation_orchestrator.py`

Pure stdlib (`itertools`, `math`, `threading`). No I/O, no concurrency execution, no provider calls.

### 5.1 `allocate_quota`

```python
def allocate_quota(
    total: int,
    providers: Sequence[ProviderConfig],
) -> dict[str, int]:
    """Allocate `total` units across providers by weight.

    Algorithm (largest-remainder method):
      1. raw_share[i] = total * weight[i] / sum(weights)
      2. floor each to int → preliminary[i]
      3. distribute the leftover (total - sum(preliminary)) by largest
         fractional remainder, ties broken by declaration order.

    Guarantees:
      - sum(result.values()) == total
      - all values >= 0
      - deterministic given fixed provider order and weights

    Edge cases:
      - total == 0 → every provider gets 0
      - single provider → it gets total
      - empty providers → raises ValueError
    """
```

Example traces:

| `total` | weights | result |
|---:|---|---|
| 100000 | `[deepseek=70, gemini=30]` | `{deepseek: 70000, gemini: 30000}` |
| 10 | `[a=60, b=40]` | `{a: 6, b: 4}` |
| 10 | `[a=1, b=1, c=1]` | `{a: 4, b: 3, c: 3}` (leftover by order) |

### 5.2 `RoundRobinScheduler`

```python
class RoundRobinScheduler:
    """Per-call weighted round-robin used by output-label generation.

    Behavior:
      - Normalizes provider weights by GCD before building the cycle.
        Example: weights [60000, 40000] become [3, 2]; cycle length 5.
      - Builds a deterministic Bresenham-style interleaved cycle.
      - next_provider() advances a cursor through the cycle, wrapping.
      - Cursor protected by threading.Lock; concurrent callers receive
        serialized consecutive scheduler positions; cursor is never
        corrupted.
      - Scheduler state is in-memory only and resets between runs.

    Distribution guarantees:
      - Over every full normalized cycle, counts exactly match
        normalized weights.
      - Over partial cycles, counts deviate by at most one slot per
        provider from proportional.
    """

    def __init__(self, providers: Sequence[ProviderConfig]) -> None: ...
    def next_provider(self) -> str: ...
    def reset(self) -> None: ...
    @property
    def cycle_length(self) -> int: ...
```

### 5.3 `render_dry_run`

```python
def render_dry_run(cfg: GenerationConfig) -> str:
    """Human-readable multi-line summary. No execution, no side effects."""
```

Output format (pinned for the smoke test — tests use `"X in output"` substring matches, not full-string equality):

```
Multi-provider dry run
Config: configs/test_providers.json  (version 1)

Input generation: enabled
  target_inputs: 10
  batch_size: 5
  dedupe: True
  providers:
    fake_a  type=fake  weight=60  threads=1  quota=6
    fake_b  type=fake  weight=40  threads=1  quota=4

Output generation: enabled
  label_attempts_per_input: 1
  selection_policy: first_valid_then_score
  provider_priority: (default declaration order)
  providers:
    fake_a  type=fake  weight=50  threads=1
    fake_b  type=fake  weight=50  threads=1
  scheduler preview (first 10): fake_a fake_b fake_a fake_b fake_a fake_b fake_a fake_b fake_a fake_b

Validation gate:
  schema: True
  semantic_validator: True
  reject_invalid: True
  retry_invalid_with_stricter_prompt: False
  max_repair_attempts: 0

Execution limits:
  global_max_workers: 4
  write_flush_every: 10

Execution: not started. This was a dry run.
```

Rules:
- `enabled=false` for a phase prints `Input generation: disabled` (or `Output generation: disabled`) and skips the inner block (no quota allocation, no scheduler preview).
- Scheduler preview always shows the **first 10** names from `RoundRobinScheduler.next_provider()`.
- ASCII only; renders the same in PowerShell, bash, Windows cmd.
- Rendered "type=fake" string is a human label backed by `ProviderConfig.provider_type`.

---

## 6. Module: `scripts/probe_validator.py`

Standalone CLI: feeds a JSONL of `(input, output)` pairs through `validate_example` and prints a per-code report. No GPU. No API.

### 6.1 CLI

```bash
python scripts/probe_validator.py \
    --input <jsonl-path> \
    [--output <report-path>] \
    [--limit N] \
    [--mode strict|warn] \
    [--fail-on-malformed] \
    [--quiet]
```

- `--input` (required): path to a JSONL file.
- `--output` (optional): per-row report JSONL. Parent directories are created automatically (`mkdir(parents=True, exist_ok=True)`).
- `--limit N` (optional, default 0): only probe first N rows; 0 means all.
- `--mode strict|warn` (default `strict`): passed to `validate_example`. **Only affects the serialized `validation.ok` in per-row reports; does NOT affect summary OK/Failed classification.**
- `--fail-on-malformed` (default off): when set, malformed rows cause exit code 3.
- `--quiet`: suppress per-row progress output; print only the final summary.

Exit codes:

- `0` — ran to completion (default).
- `2` — input file missing or unreadable.
- `3` — malformed rows found AND `--fail-on-malformed` is set.

### 6.2 Streaming JSONL reader

`probe_validator.py` uses its own streaming reader instead of `_lib.load_jsonl` (which raises on bad JSON):

```python
def iter_jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                yield line_no, raw, json.loads(text), None
            except json.JSONDecodeError as e:
                yield line_no, raw.rstrip("\n"), None, f"json_parse_error: {e}"
```

### 6.3 Summary classification (severity-based)

```
Malformed = bad JSON OR missing 'input' OR non-dict 'output'
OK = well-formed rows with zero error-severity errors
Failed = well-formed rows with ≥ 1 error-severity error
Warnings by code = warning-severity errors counted across ALL well-formed rows
```

`--mode warn` flips `result.ok` to True for all rows; the summary classification uses severity directly, so warn mode does NOT hide failures.

### 6.4 Summary output (pinned)

```
Validator probe — data/distill/train.jsonl
Rows scanned:      4823
Malformed:            0
OK:                3614   (74.9%)
Failed:            1209   (25.1%)

Failures by code:
  AMOUNT_NOT_IN_INPUT          612
  SUSPICIOUS_DUPLICATE         287
  SUPERSEDED_AMOUNT_USED        81
  CURRENCY_MISMATCH             49
  TXN_COUNT_EXCEEDS_CANDIDATES 145
  NO_AMOUNT_IN_INPUT            35

Warnings by code:
  TXN_COUNT_BELOW_CANDIDATES   208
```

Failure codes listed in **descending count order**; ties broken alphabetically.

### 6.5 Per-row report shape (`--output` path)

Well-formed rows:
```json
{
  "line": 42,
  "input": "500 beer",
  "output": {"transactions": [...]},
  "validator_ok": true,
  "validation": {
    "ok": true,
    "amount_values_match_active_candidates": true,
    "txn_count_matches_active_candidates": true,
    "duplicate_transactions_found": false,
    "superseded_amount_used": false,
    "errors": []
  }
}
```

Malformed rows:
```json
{
  "line": 42,
  "raw": "{bad json",
  "validator_ok": null,
  "validation": null,
  "malformed": "json_parse_error: Expecting property name..."
}
```

---

## 7. Stage 5 CLI integration

### 7.1 New flags

Added to `parse_args()` after the existing `--retry-validation-failed`:

```python
    p.add_argument("--provider-config", type=Path, default=None,
                   help="Path to a multi-provider generation config JSON. "
                        "Slice 1: only --dry-run-quota is supported.")
    p.add_argument("--dry-run-quota", action="store_true",
                   help="With --provider-config: parse + validate config, "
                        "print quota allocations and scheduler preview, exit 0. "
                        "No generation runs.")
    p.add_argument("--multi-provider", action="store_true",
                   help="Reserved for Slice 2; in Slice 1 every combination "
                        "involving this flag exits via parser.error.")
```

### 7.2 `parse_args` returns `(parser, namespace)`

```python
def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    p = argparse.ArgumentParser(...)
    # ... existing args ...
    return p, p.parse_args()
```

### 7.3 Flag matrix

| Flags | Behavior |
|---|---|
| no `--provider-config` | legacy Stage 5 behavior, fully unchanged |
| `--provider-config --dry-run-quota` | load config, render dry-run, exit 0 |
| `--provider-config` only | `parser.error("Slice 1: --provider-config requires --dry-run-quota...")`, exit 2 |
| `--dry-run-quota` only | `parser.error("--dry-run-quota requires --provider-config.")`, exit 2 |
| `--multi-provider` only | `parser.error("--multi-provider requires --provider-config.")`, exit 2 |
| `--provider-config --multi-provider` | `parser.error("Multi-provider execution lands in Slice 2...")`, exit 2 |
| `--provider-config --multi-provider --dry-run-quota` | same `parser.error` as above, exit 2 |

`parser.error` is used everywhere — no `NotImplementedError`, no tracebacks for known CLI contract violations.

### 7.4 main() shape

```python
def main() -> int:
    parser, args = parse_args()
    _enforce_slice1_flag_contract(args, parser)   # may parser.error and exit 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"05_generate_distillation_data_{ts}.log")

    if args.dry_run_quota:
        assert args.provider_config is not None
        return _run_dry_run(args.provider_config)

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Stage 5: phase=%s args=%s", args.phase, vars(args))

    if args.phase in ("all", "inputs"):
        phase_inputs(args)
    if args.phase in ("all", "label"):
        phase_label(args)
    if args.phase in ("all", "eval"):
        phase_eval(args)

    logging.info("Stage 5 done.")
    return 0
```

Contract enforcement runs BEFORE `setup_logging` so invalid-CLI invocations don't create empty log files.

### 7.5 `_run_dry_run` config-load error handling

```python
def _run_dry_run(config_path: Path) -> int:
    from generation_config import ConfigError, load_generation_config
    from generation_orchestrator import render_dry_run

    try:
        cfg = load_generation_config(config_path)
    except ConfigError as e:
        logging.error("Invalid provider config: %s", e)
        print(f"Invalid provider config: {e}", file=sys.stderr)
        return 2

    logging.info("Loaded config from %s (version %d)", config_path, cfg.version)
    print(render_dry_run(cfg))
    return 0
```

Bad config produces a clean message and exit 2 — no traceback. `sys` import added at top of file.

---

## 8. Testing

### 8.1 Test file inventory

All pure-Python, no GPU, no API, < 5s total. Use `pytest`.

| File | What it covers |
|---|---|
| `tests/test_generation_config.py` | Loader happy-path; missing-field errors with field paths in message; unknown-type rejection; unknown-key warnings (caplog); fixture-path resolution relative to config dir; fixture-missing → ConfigError; `enabled=false` short-circuits provider-list requirement; version mismatch; `example_providers.json` loads cleanly and renders dry-run without any SDK imports |
| `tests/test_provider_scheduler.py` | `allocate_quota` invariants (sum-to-total, single-provider, zero-total, tie-break order, empty providers raises); `RoundRobinScheduler` GCD-normalized cycle length; distribution exact over `N × cycle_length` calls; thread-safety with 8 workers × 1000 calls; `render_dry_run` substring checks (`"Config:"`, `"(version 1)"`, `"Input generation:"`, `"scheduler preview"`); `enabled=false` phase rendering |
| `tests/test_fake_provider.py` | Cursor wrap-around (`[a,b,c]`, `n=5` → `[a,b,c,a,b]`; next call continues from `c`); `generate_label` resolution order (raw_output → output → miss_payload); miss returns miss_payload when input not in fixture; bad-fixture loader raises `ProviderError` at construction time with line number; thread-safe cursor under concurrent calls; immutable label-map (no lock needed) |
| `tests/test_probe_validator.py` | All-pass fixture → `Failed: 0`, exit 0; mixed fixture → expected per-code counts; malformed row → counted in `Malformed`, default exit 0; `--fail-on-malformed` flips exit to 3; `--mode warn` doesn't hide errors from summary; `--output` creates parent directory; `--limit N` truncates correctly |
| `tests/test_stage5_flag_contract.py` | Every row of the §7.3 flag matrix; uses `subprocess.run` against the actual CLI to verify end-to-end (no SDK imports); confirms exit codes and stderr substrings |

### 8.2 Fixture files

```
tests/fixtures/
  fake_inputs.jsonl       20 rows of canned input strings
                          (7 from _lib.SAMPLE_INPUTS + 13 generated variants)

  fake_labels.jsonl       13 rows: 10 valid (input, output) pairs + 3 raw_output
                          failure rows covering JSON-parse-fail, schema-fail,
                          validator-fail. Miss behavior tested by querying an
                          input absent from the fixture (no special miss row).

  probe_examples.jsonl    11 rows: SCHEMA_INVALID + six semantic error rows
                          (AMOUNT_NOT_IN_INPUT, SUSPICIOUS_DUPLICATE,
                          SUPERSEDED_AMOUNT_USED, CURRENCY_MISMATCH,
                          TXN_COUNT_EXCEEDS_CANDIDATES, NO_AMOUNT_IN_INPUT) +
                          3 known-OK rows + 1 row triggering only
                          TXN_COUNT_BELOW_CANDIDATES (warning).
```

### 8.3 Coverage targets

- `generation_config.py`: ≥ 90%
- `generation_orchestrator.py`: ≥ 95%
- `llm_providers.py`: ≥ 85% (UnimplementedProvider's raise paths counted)
- `probe_validator.py`: ≥ 85%

---

## 9. Smoke commands (no API, no GPU)

Human-runnable verification commands; included in the README's "Multi-provider scaffolding (Slice 1)" subsection.

```bash
# 1. Validate the test config, see quotas
python scripts/05_generate_distillation_data.py \
    --provider-config configs/test_providers.json \
    --dry-run-quota

# 2. Confirm CLI help/import path still works (legacy untouched)
python scripts/05_generate_distillation_data.py --help

# 3. Each error-exit path (all should exit 2 with a clear stderr message)
python scripts/05_generate_distillation_data.py --provider-config configs/test_providers.json
python scripts/05_generate_distillation_data.py --dry-run-quota
python scripts/05_generate_distillation_data.py --multi-provider
python scripts/05_generate_distillation_data.py --provider-config configs/test_providers.json --multi-provider --dry-run-quota

# 4. Probe the validator against the hand-picked fixture
python scripts/probe_validator.py \
    --input tests/fixtures/probe_examples.jsonl

# 5. Probe the validator against the EXISTING distillation training data —
#    the high-value diagnostic: what fraction of current teacher labels pass?
python scripts/probe_validator.py \
    --input data/distill/train.jsonl \
    --output reports/validator_probe_existing_train.jsonl

# 6. Probe with --limit to confirm exit behavior on truncated data
python scripts/probe_validator.py \
    --input data/distill/train.jsonl \
    --limit 100
```

`probe_validator.py` creates `reports/` automatically.

---

## 10. Rollout

Single commit on `feat/multi-provider-generation`. This is purely additive scaffolding.

1. Add `scripts/llm_providers.py` + `tests/test_fake_provider.py` → green.
2. Add `scripts/generation_config.py` + `tests/test_generation_config.py` → green.
3. Add `scripts/generation_orchestrator.py` + `tests/test_provider_scheduler.py` → green.
4. Add `configs/test_providers.json`, `configs/example_providers.json`, `tests/fixtures/{fake_inputs,fake_labels,probe_examples}.jsonl`.
5. Modify `scripts/05_generate_distillation_data.py` — flags + `_enforce_slice1_flag_contract` + `_run_dry_run` + `parse_args` returns `(parser, namespace)`.
6. Add `tests/test_stage5_flag_contract.py` → green.
7. Add `scripts/probe_validator.py` + `tests/test_probe_validator.py` → green.
8. Add `docs/provider_config.md` (companion explaining the JSON config schema).
9. Update `README.md`:
   - New "Multi-provider scaffolding (Slice 1)" subsection with the smoke commands from §9.
   - Verify the prior validator-slice README content (Stage 4 aggregates paragraph, Stage 5 validator-gate paragraph, Dev dependencies subsection) is still present and accurate; touch up any wording gaps left from the prior slice.
10. Run full suite (`pytest tests/`) and verify all existing + new tests pass.
11. Single commit.

---

## 11. Decision points after rollout

After running smoke command 5 (probe against existing `data/distill/train.jsonl`):

| Pass rate | Action |
|---|---|
| < 50% | Validator is more aggressive than expected. Triage `reports/validator_probe_existing_train.jsonl` for top failure codes BEFORE planning Slice 2. May need parser tweaks before regenerating any data. |
| 50–80% | Plausible range. Inspect top failure codes; if failures are explainable (known parser gaps, known v1-vocab limitations), proceed with Slice 2/3 as planned. |
| > 90% | Validator may be too lenient. Manually inspect a sample of OK rows to confirm we're not under-rejecting. |

These thresholds are heuristics, not release gates.

---

## 12. Public-surface dependency graph

```
llm_providers.py
  ├─ stdlib only (pathlib, threading, json, dataclasses)
  ├─ forward-string-references generation_config.ProviderConfig
  └─ NO SDK imports

generation_config.py
  ├─ stdlib only
  └─ ConfigError, GenerationConfig, load_generation_config

generation_orchestrator.py
  ├─ stdlib only (math, threading, itertools)
  ├─ runtime import: `from generation_config import ProviderConfig, GenerationConfig`
  │   (both modules are stdlib-only, so a runtime import does not bloat anything)
  └─ allocate_quota, RoundRobinScheduler, render_dry_run

probe_validator.py
  ├─ stdlib only
  ├─ flat-import setup (matches repo convention):
  │     sys.path.insert(0, str(Path(__file__).resolve().parent))
  │     from _lib import validate_example, serialize_validation_result
  └─ standalone CLI

scripts/05_generate_distillation_data.py
  └─ +imports generation_config, generation_orchestrator inside _run_dry_run only

NO new third-party deps.
NO changes to _lib.py.
NO changes to amount_parser.py or validator.py.
NO changes to scripts/04_eval.py.
```
