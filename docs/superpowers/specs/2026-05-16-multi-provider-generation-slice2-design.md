# Multi-Provider Generation — Slice 2 Design Spec

**Date:** 2026-05-16
**Status:** Approved, ready for implementation plan
**Scope:** Slice 2 of 4 in the multi-provider generation roadmap — real DeepSeek + Gemini providers, threaded multi-provider input generation, global dedupe, provider metadata in `inputs_raw.jsonl`.
**Prerequisite:** Slice 1 (`a28a69a`, `3ef4875`, `e110a14`, `93811f7`, `55e0af9`, `e937c7e` on `feat/multi-provider-generation`) already merged or branched-off. Slice 2 lands on `feat/multi-provider-slice2`.

---

## 1. Problem

Slice 1 shipped the scaffolding: `LLMProvider` Protocol, `FakeProvider`, `UnimplementedProvider`, config dataclasses with strict validation, `allocate_quota`, `RoundRobinScheduler`, `render_dry_run`, `probe_validator.py`, and the `--provider-config` / `--dry-run-quota` / `--multi-provider` CLI flags. But `--multi-provider` currently exits via `parser.error` — no real generation runs.

Slice 2 makes the command actually run for the input phase:

```bash
python scripts/05_generate_distillation_data.py \
    --phase inputs \
    --provider-config configs/<config>.json \
    --multi-provider
```

Real DeepSeek + Gemini API calls happen, concurrently, across multiple providers, with backoff/retry, with global dedupe against existing `inputs_raw.jsonl`, and with provider metadata recorded in each new row.

Output labeling (Slice 3) and metrics/cost tracking (Slice 4) stay out.

## 2. Solution overview

```
scripts/
  _retry.py                       (NEW)     Exponential backoff with jitter, injectable sleep
  llm_providers.py                (MODIFY)  Add DeepSeekProvider, GeminiProvider, _parse_input_lines.
                                            Update create_provider factory. local_teacher stays
                                            UnimplementedProvider.
  _lib.py                         (MODIFY)  Move clean_input_line here with fixed regex; add
                                            normalize_input.
  05_generate_distillation_data.py (MODIFY) Add phase_inputs_multi_provider; rename + extend the
                                            flag-contract enforcer; honor DISTILL_DIR_OVERRIDE;
                                            update read_inputs_jsonl malformed-row policy.

configs/
  example_providers.json          (MODIFY)  Make safe by default: target_inputs=10, enabled=false.
  smoke_real_providers.json       (NEW)     Opt-in 100-input real-API smoke config.

tests/
  test_retry.py                   (NEW)
  test_real_providers.py          (NEW)
  test_multi_provider_inputs.py   (NEW)
  test_clean_input_line.py        (NEW)
  test_stage5_flag_contract.py    (MODIFY)  Update rows; new helper env support.

requirements-train.txt            (MODIFY)  +openai>=1.50, +google-genai (floor pinned at impl time)
README.md                         (MODIFY)  New "Slice 2: Multi-provider input generation" subsection.
docs/provider_config.md           (MODIFY)  Update "What loads successfully" — now includes
                                            running, not just loading, of deepseek/gemini providers.
```

### Out of scope

- `LocalTeacherProvider` real implementation (stays UnimplementedProvider; Slice 3).
- `generate_label` real implementations (stay NotImplementedError; Slice 3).
- Gemini structured-output JSON-schema mode (Slice 3).
- Per-provider cost tracking, latency metrics, token counters (Slice 4).
- Streaming responses (neither slice plans to use them).

---

## 3. Real provider interfaces

### 3.1 SDK choice

- **DeepSeek**: `openai>=1.50` (DeepSeek exposes an OpenAI-compatible endpoint; the SDK client takes a `base_url`).
- **Gemini**: `google-genai` (the newer "Gen AI SDK"). Active development; structured-output support that Slice 3 will use. Version floor pinned by the implementation plan after running `from google import genai; from google.genai import types, errors` successfully against the installed SDK.

### 3.2 Lazy SDK imports

`scripts/llm_providers.py` imports neither `openai` nor `google.genai` at module load. SDK imports happen inside `DeepSeekProvider.__init__` / `GeminiProvider.__init__`. Importing `llm_providers` for tests, dry-run, or any other purpose remains SDK-free.

### 3.3 `DeepSeekProvider`

```python
class DeepSeekProvider:
    provider_type = "deepseek"

    def __init__(
        self,
        name: str,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        temperature: float | None = 1.0,
        max_tokens: int | None = 8000,
        max_retries: int = 3,
        timeout: int = 300,
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self._sleep = time.sleep   # test hook

        from openai import OpenAI                                       # lazy
        from openai import APITimeoutError, RateLimitError, APIConnectionError

        # Intentionally do not read OPENAI_API_KEY; DeepSeek has its own key.
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ProviderError(
                f"DeepSeekProvider {name!r}: DEEPSEEK_API_KEY not set"
            )
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._retryable_excs = (APITimeoutError, RateLimitError, APIConnectionError)

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        if n < 0:
            raise ProviderError(f"generate_inputs: n must be >= 0, got {n}")
        if n == 0:
            return []
        text = self._call_with_retry(prompt)
        return _parse_input_lines(text, expected=n)

    def generate_label(self, input_text: str) -> str:
        raise NotImplementedError(
            f"deepseek provider {self.name!r}: output labeling lands in Slice 3"
        )

    def _call_with_retry(self, prompt: str) -> str:
        return retry_with_backoff(
            lambda: self._call_api(prompt),
            retryable=self._retryable_excs,
            max_retries=self.max_retries,
            logger_name=f"deepseek.{self.name}",
            sleep=self._sleep,
        )

    def _call_api(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.timeout,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
```

Invariants:

- `__init__` performs no network I/O (only client construction, which is local).
- `__init__` raises `ProviderError` (not arbitrary auth error) when the env key is missing.
- `OPENAI_API_KEY` is **never** used as a fallback.
- `temperature` / `max_tokens` are **omitted** from the SDK call when `None`, not passed as `None`.

### 3.4 `GeminiProvider`

```python
class GeminiProvider:
    provider_type = "gemini"

    def __init__(
        self,
        name: str,
        *,
        model: str,
        api_key: str | None = None,
        temperature: float | None = 1.0,
        max_tokens: int | None = 8000,
        max_retries: int = 3,
        structured_output: bool = False,   # ignored in Slice 2; Slice 3 wires it
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.structured_output = structured_output
        self._sleep = time.sleep

        from google import genai                                       # lazy
        from google.genai import errors as genai_errors

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ProviderError(
                f"GeminiProvider {name!r}: GOOGLE_API_KEY (or GEMINI_API_KEY) not set"
            )
        self._client = genai.Client(api_key=key)
        # Predicate filters: retry only on 429 + 5xx. 4xx (auth, bad model, bad request)
        # are non-retryable and must bubble up immediately.
        self._retryable_excs = (genai_errors.APIError, genai_errors.ClientError)

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        if n < 0:
            raise ProviderError(f"generate_inputs: n must be >= 0, got {n}")
        if n == 0:
            return []
        text = self._call_with_retry(prompt)
        return _parse_input_lines(text, expected=n)

    def generate_label(self, input_text: str) -> str:
        raise NotImplementedError(
            f"gemini provider {self.name!r}: output labeling lands in Slice 3"
        )

    def _call_with_retry(self, prompt: str) -> str:
        return retry_with_backoff(
            lambda: self._call_api(prompt),
            retryable=self._retryable_excs,
            is_retryable=_is_retryable_gemini_error,
            max_retries=self.max_retries,
            logger_name=f"gemini.{self.name}",
            sleep=self._sleep,
        )

    def _call_api(self, prompt: str) -> str:
        from google.genai import types
        cfg_kwargs = {}
        if self.temperature is not None:
            cfg_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = self.max_tokens
        config = types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return resp.text or ""


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """Defensive against SDK variation: some SDK versions expose status as
    int, others as string. Coerce to int; on failure, do not retry."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        return False
    return status in {429, 500, 502, 503, 504}
```

Timeout handling for Gemini: verified during implementation (the `google-genai` SDK exposes request options). If the SDK supports a per-call timeout, wire it; otherwise leave it as a documented gap.

### 3.5 `_parse_input_lines`

Lives in `scripts/llm_providers.py`. Imports `clean_input_line` from `_lib` lazily inside the function to avoid the digit-prefix module name issue.

```python
def _parse_input_lines(text: str, *, expected: int) -> list[str]:
    """Provider output → cleaned, locally-deduped input strings.

    Local dedupe (within this provider response) catches the common case
    where a single API call returns the same line twice. Global dedupe
    (against existing inputs_raw.jsonl + this run's accepted set) happens
    in the orchestrator. Both layers are needed: local saves the queue
    from getting flooded, global is the source of truth.
    """
    from _lib import clean_input_line, normalize_input
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        cleaned = clean_input_line(raw)
        if cleaned is None:
            continue
        key = normalize_input(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= expected:
            break
    return out
```

### 3.6 Factory update

`create_provider` in `llm_providers.py` (Slice 1's branch on `fake` stays unchanged):

```python
def create_provider(cfg) -> LLMProvider:
    if cfg.provider_type == "fake":
        return FakeProvider(...)   # unchanged
    if cfg.provider_type == "deepseek":
        return DeepSeekProvider(
            name=cfg.name,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.max_retries,
        )
    if cfg.provider_type == "gemini":
        return GeminiProvider(
            name=cfg.name,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.max_retries,
            structured_output=cfg.structured_output,
        )
    if cfg.provider_type == "local_teacher":
        return UnimplementedProvider(
            name=cfg.name, provider_type="local_teacher", model=cfg.model,
        )
    raise ValueError(
        f"Unknown provider type {cfg.provider_type!r} for provider {cfg.name!r}"
    )
```

---

## 4. `scripts/_retry.py`

Pure stdlib. Used by both real providers (and any future one). Slice 4 will likely add observability hooks here.

```python
def retry_with_backoff(
    fn: Callable[[], T],
    *,
    retryable: tuple[Type[BaseException], ...],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.5,
    is_retryable: Callable[[BaseException], bool] | None = None,
    logger_name: str = "retry",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `fn`; on a retryable exception, sleep and retry up to max_retries.

    Retry predicate:
      - Exception type must be in `retryable` tuple, AND
      - If `is_retryable` is provided, it must return True for the exception.

    Sleep duration: min(max_delay, base_delay * 2**attempt) + uniform(0, jitter).
    On final failure: re-raises the original exception unchanged.
    """
```

Tests inject `sleep=lambda _: None` to avoid real delays. Providers pass `sleep=self._sleep`, which tests override per-instance.

---

## 5. `phase_inputs_multi_provider` orchestrator

Lives in `scripts/05_generate_distillation_data.py`. Concurrency model:

- One shared `ThreadPoolExecutor` sized to `min(sum(p.threads for p in providers), cfg.rate_limits.global_max_workers)`.
- For each provider with `quota[p.name] > 0`, submit `worker_count = min(p.threads, max(1, ceil(provider_quota / batch_size)))` worker tasks. Providers with zero quota (which can happen on small `pending` values) are skipped entirely — no workers spawned for them.
- Each worker generates batches and submits results to a shared `queue.Queue`. Workers never write to disk.
- The **main thread** is the sole writer to `inputs_raw.jsonl`. It consumes the queue, dedupes by `normalize_input`, writes accepted rows, advances the progress bar.
- Main thread sets `stop_event` when `accepted_total >= pending`. Workers stop on next iteration.

### 5.1 Stop conditions (precedence)

1. **Success**: `accepted_total >= pending` → `stop_event.set()`, break.
2. **Providers exhausted**: all futures done AND queue empty AND `accepted_total < pending` → log warning, break.
3. **Per-worker streak caps**:
   - `empty_response_streak >= MAX_CONSECUTIVE_EMPTY_BATCHES (=3)` — provider returned 0 cleaned lines.
   - `provider_error_streak >= MAX_CONSECUTIVE_PROVIDER_ERRORS (=3)` — provider raised through retries.
   - Workers log and exit; main thread continues until conditions 1 or 2 hit.
4. **Duplicate-stall guard**: `consecutive_dupes_since_last_accept >= MAX_CONSECUTIVE_DUPLICATES_BEFORE_GIVEUP (=200)` → log warning, `stop_event.set()`, break. Protects against targets that exceed provider unique supply.

**Counter semantics**:
- Incremented when a queued line is rejected as duplicate.
- Reset to `0` only when a row is accepted and written.
- NOT incremented on `queue.Empty` timeouts — temporary provider slowness must not trigger the stall guard.

### 5.2 Quota as soft scheduling hint

```text
Quota is a soft scheduling hint in Slice 2:
- It determines how many worker tasks each provider gets.
- It does not cap accepted rows per provider.
- Completion is global: accepted_unique >= pending.
- Final provider distribution may differ from configured weights because
  providers have different latency, duplicate rate, and failure rate.
- Slice 4 metrics will report actual accepted distribution.
```

### 5.3 Resume safety

On startup, read existing `inputs_raw.jsonl`, build `existing_normalized = {normalize_input(s) for s in existing}`, compute `pending = max(0, cfg.input_generation.target_inputs - len(existing_normalized))`. If `pending == 0`, log and return without launching workers. Quotas use `pending`, not `target_inputs`, so resumed runs only split the remaining work.

Mid-run interruption is safe: only the main thread writes, in append mode, with periodic flush. The file is intact through the last flush.

### 5.4 `--n-inputs` precedence

In multi-provider mode, `cfg.input_generation.target_inputs` is the source of truth. If `args.n_inputs` is set and differs from `cfg.input_generation.target_inputs`, log a `logger.warning` and ignore `--n-inputs`. The warning includes both values.

### 5.5 Provider metadata in `inputs_raw.jsonl`

Each new row from multi-provider path:

```json
{
  "input": "500 rs on beer 50 rs on candy",
  "_source": "synthetic_input",
  "_provider": "deepseek_v4_pro",
  "_model": "deepseek-chat",
  "_batch_id": "20260516_134022"
}
```

- `_source`: literal string `"synthetic_input"`.
- `_provider`: matches `ProviderConfig.name`.
- `_model`: model id passed to SDK. **Always emitted**, even when `null` (FakeProvider).
- `_batch_id`: timestamp at run start, one value per `phase_inputs_multi_provider` invocation.

Legacy rows (`{"input": "..."}`) stay valid. Consumers reading `obj["input"]` continue to work.

### 5.6 Helpers in `05_generate_distillation_data.py`

- `_build_input_prompt(n, batch_focus)`: wraps the existing `INPUT_GEN_PROMPT` template with a focus from `_lib.BATCH_FOCUSES`. Caller is responsible for instructing the model on how many lines to return.
- `_next_focus(provider_name)`: deterministic round-robin over `BATCH_FOCUSES`, keyed per-provider via dict + lock. No randomness.
- `_provider_worker(...)`: per-provider thread function. Loops until `stop_event` is set or its own failure streak caps hit. Maintains `empty_response_streak` and `provider_error_streak` separately.

`phase_inputs_multi_provider` uses `normalize_input` directly from `_lib` for global dedupe (no local wrapper). The orchestrator and `_parse_input_lines` share the same canonical normalization function.

---

## 6. Data shape changes

### 6.1 `read_inputs_jsonl` — malformed-row policy

```python
def read_inputs_jsonl(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    bad_json = 0
    bad_shape = 0
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning("read_inputs_jsonl: %s:%d invalid JSON: %s",
                                path.name, line_no, e)
                bad_json += 1
                continue
            if not isinstance(obj, dict):
                logging.warning("read_inputs_jsonl: %s:%d row is not an object, skipping",
                                path.name, line_no)
                bad_shape += 1
                continue
            if "input" not in obj or not isinstance(obj["input"], str):
                logging.warning(
                    "read_inputs_jsonl: %s:%d missing or non-string 'input' field, skipping",
                    path.name, line_no,
                )
                bad_shape += 1
                continue
            out.append(obj["input"])
    if bad_json or bad_shape:
        logging.info(
            "read_inputs_jsonl: loaded %d inputs from %s; skipped bad_json=%d bad_shape=%d",
            len(out), path, bad_json, bad_shape,
        )
    return out
```

### 6.2 `clean_input_line` migration to `_lib.py`

**Bug fix included in the migration**: the original regex `r"^[\s\-\*\d]+[.)\s]+"` could corrupt digit-led natural content (`"500 beer"` → `"beer"`) because the leading digit class is too aggressive. The migration uses a safer regex:

```python
# scripts/_lib.py (added near the bottom of the existing helpers)
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)")


def clean_input_line(line: str) -> str | None:
    """Strip leading bullets/numbering and surrounding whitespace.

    Strips:
      "- 500 beer"   -> "500 beer"
      "* 500 beer"   -> "500 beer"
      "1. 500 beer"  -> "500 beer"
      "2) 500 beer"  -> "500 beer"

    Does NOT strip digit-led natural content:
      "500 beer"     -> "500 beer"
      "1 lakh rent"  -> "1 lakh rent"
    """
    s = line.strip()
    if not s:
        return None
    if s.startswith("```") or s.lower().startswith("output:") or s.lower().startswith("example"):
        return None
    s = _LINE_PREFIX_RE.sub("", s).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if len(s) < 3 or len(s) > 300:
        return None
    return s


def normalize_input(text: str) -> str:
    """Canonical dedupe key. Lower-cased, whitespace-collapsed."""
    return " ".join(text.lower().split())
```

The old definition + `_LINE_PREFIX_RE` constant in `05_generate_distillation_data.py` are deleted; the file gets a re-export:

```python
from _lib import clean_input_line   # noqa: F401  (used by legacy phase_inputs)
```

**Behavior change for legacy `phase_inputs`**: the legacy single-DeepSeek path also gets the safer regex. This is a deliberate fix, not a regression — the old regex was silently corrupting digit-led lines whenever DeepSeek returned clean (non-numbered) output. The new regex preserves the documented intent of `clean_input_line` (strip bullets/numbering) while removing the digit-stripping bug.

### 6.3 `DISTILL_DIR_OVERRIDE` env var

Tests need to write `inputs_raw.jsonl` to a tmp dir, not the repo's real `data/distill/`. Implementation:

```python
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_EVAL = REPO_ROOT / "data" / "clean" / "eval.jsonl"

DISTILL_DIR = Path(os.environ.get("DISTILL_DIR_OVERRIDE", REPO_ROOT / "data" / "distill"))
INPUTS_FILE = DISTILL_DIR / "inputs_raw.jsonl"
TRAIN_FILE = DISTILL_DIR / "train.jsonl"
DISTILL_EVAL_FILE = DISTILL_DIR / "eval.jsonl"
FAILED_FILE = DISTILL_DIR / "failed.jsonl"
```

All derived paths read `DISTILL_DIR` **after** env resolution. Tests using subprocess pass `env={**os.environ, "DISTILL_DIR_OVERRIDE": str(tmp_path)}`.

`DISTILL_DIR_OVERRIDE` is documented in the README as a **test/dev-only** knob. CLI users should rely on the default `data/distill/`; the env var is not surfaced in `--help`.

---

## 7. Stage 5 CLI updates

### 7.1 Flag matrix

| Invocation | Behavior |
|---|---|
| no flags | Legacy Stage 5 default behavior, unchanged. |
| `--phase <X>` only (no provider flags) | Legacy path for phase X, unchanged. |
| `--provider-config <path> --dry-run-quota` | Slice 1 dry-run, phase value ignored. Exit 0. |
| `--provider-config <path>` alone | `parser.error("--provider-config requires --dry-run-quota or --multi-provider")`. Exit 2. |
| `--dry-run-quota` alone | `parser.error("--dry-run-quota requires --provider-config")`. Exit 2. |
| `--multi-provider` alone | `parser.error("--multi-provider requires --provider-config")`. Exit 2. |
| **`--provider-config <path> --multi-provider --phase inputs`** | **NEW. Runs `phase_inputs_multi_provider`. Exit 0.** |
| `--provider-config <path> --multi-provider --phase label` | `parser.error("--multi-provider --phase label is reserved for Slice 3")`. Exit 2. |
| `--provider-config <path> --multi-provider --phase eval` | `parser.error("--multi-provider --phase eval is not supported; use legacy mode")`. Exit 2. |
| `--provider-config <path> --multi-provider --phase all` | `parser.error("--multi-provider does not support --phase all; specify --phase inputs or use legacy mode")`. Exit 2. |
| `--provider-config <path> --multi-provider --dry-run-quota --phase <any>` | Dry-run wins; phase ignored. Exit 0. |

Dry-run wins because `_enforce_flag_contract` skips the multi-provider phase restrictions when `args.dry_run_quota` is true, and `main()` then dispatches to `_run_dry_run` before any execution branch.

### 7.2 `_enforce_flag_contract` (renamed from `_enforce_slice1_flag_contract`)

```python
def _enforce_flag_contract(args, parser):
    # Standalone-flag errors (unchanged from Slice 1).
    if args.multi_provider and not args.provider_config:
        parser.error("--multi-provider requires --provider-config.")
    if args.dry_run_quota and not args.provider_config:
        parser.error("--dry-run-quota requires --provider-config.")
    if args.provider_config and not (args.dry_run_quota or args.multi_provider):
        parser.error(
            "--provider-config requires --dry-run-quota or --multi-provider."
        )
    # Multi-provider phase restrictions for Slice 2.
    # Dry-run wins: phase checks skipped when --dry-run-quota is present.
    if args.multi_provider and args.provider_config and not args.dry_run_quota:
        if args.phase == "label":
            parser.error(
                "--multi-provider --phase label is reserved for Slice 3 "
                "(validator-gated output labeling)."
            )
        if args.phase == "eval":
            parser.error(
                "--multi-provider --phase eval is not supported; "
                "use legacy mode (no --multi-provider) for eval."
            )
        if args.phase == "all":
            parser.error(
                "--multi-provider does not support --phase all in Slice 2; "
                "specify --phase inputs, or use legacy mode (no --multi-provider)."
            )
        # --phase inputs is the only allowed combo; falls through to main().
```

### 7.3 `main()` dispatch

```python
def main() -> int:
    parser, args = parse_args()
    _enforce_flag_contract(args, parser)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"05_generate_distillation_data_{ts}.log")

    if args.dry_run_quota:
        assert args.provider_config is not None
        return _run_dry_run(args.provider_config)

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Stage 5: phase=%s args=%s", args.phase, vars(args))

    if args.multi_provider:
        # Slice 2: only --phase inputs reaches here (others rejected by contract).
        assert args.phase == "inputs"
        assert args.provider_config is not None
        phase_inputs_multi_provider(args)
        logging.info("Stage 5 done.")
        return 0

    # Legacy path — unchanged.
    if args.phase in ("all", "inputs"):
        phase_inputs(args)
    if args.phase in ("all", "label"):
        phase_label(args)
    if args.phase in ("all", "eval"):
        phase_eval(args)

    logging.info("Stage 5 done.")
    return 0
```

### 7.4 Help-text refresh

- `--provider-config`: "Path to a multi-provider generation config JSON. Use with `--dry-run-quota` to inspect the config, or with `--multi-provider --phase inputs` (Slice 2) to run real input generation."
- `--multi-provider`: "Run multi-provider execution. Slice 2 supports `--phase inputs` only; `label` and `all` exit via parser.error."
- `--dry-run-quota`: unchanged.

---

## 8. Tests

All pytest; no real API calls. Real-API smoke is a manual CLI command, not in pytest.

### 8.1 New test files

| File | Coverage |
|---|---|
| `tests/test_retry.py` | First-try success, retry-then-success, exhaustion, non-retryable propagation, `is_retryable` predicate filter, exponential delay growth (injected sleep collector) |
| `tests/test_real_providers.py` | DeepSeek + Gemini construction, missing-key error, retry on transient (fake exception), max-retries exhaustion, omit-None kwargs, no-fallback to OPENAI_API_KEY, Gemini 400 not retried, direct `_parse_input_lines` test |
| `tests/test_multi_provider_inputs.py` | End-to-end via subprocess + FakeProvider: target count, metadata fields, resume appends only new, already-at-target no-op, `--n-inputs` ignored warning (assert on row count) |
| `tests/test_clean_input_line.py` | Regex regression — digit-led naturals NOT stripped, real bullets/numbering ARE stripped, normalize_input idempotency |
| `tests/test_stage5_flag_contract.py` (modify) | Drop "multi-provider always errors" rows; add Slice 2 phase-rule rows; `run_cli` accepts `env=` |

### 8.2 Test hygiene rules pinned for implementation

- Retry tests inject `sleep=lambda _: None`. Providers store `self._sleep` and pass it explicitly to `retry_with_backoff(..., sleep=self._sleep)`; tests do `p._sleep = lambda _: None`.
- Retry-behavior tests use **fake exception classes** (`class FakeRateLimitError(Exception): pass`), not real SDK exception constructors. Providers' `_retryable_excs` is reassigned to `(FakeError,)` in the test setup.
- Constructor tests verify `_retryable_excs` contains the real SDK exception types but do not raise them.
- Tests assert `__init__` does no network I/O by mocking SDK Client constructors.
- All subprocess tests pass `env={**os.environ, "DISTILL_DIR_OVERRIDE": str(tmp_path)}`. Repo `data/distill/` is never touched.
- `test_deepseek_omits_none_kwargs` uses small fake classes (`FakeMessage`, `FakeChoice`, `FakeResp`), not tuple-update hacks.

### 8.3 Coverage targets

- `_retry.py`: ≥ 95%.
- `llm_providers.py` (after additions): ≥ 80%.
- `phase_inputs_multi_provider` and helpers in `05_generate_distillation_data.py`: exercised end-to-end via subprocess; not in `--cov` target.

---

## 9. Config files

### 9.1 `configs/example_providers.json` — make safe by default

Replace the Slice 1 version's `target_inputs: 10000` and `enabled: true` with a safe-by-default shape:

- `input_generation.enabled: false`
- `input_generation.target_inputs: 10`
- Other fields unchanged.

Even with valid API keys, running `--multi-provider --phase inputs` against this config does nothing (enabled=false). Documented in `docs/provider_config.md` as "example only, not for real runs."

**Note on `enabled=false`**: The config loader's strict-type and field validation still runs over the provider entries (model names, weights, types, etc.) — `enabled=false` only short-circuits *execution* and *quota allocation*, not validation. So `example_providers.json` still needs valid-looking model names like `"deepseek-chat"` even though it never sends requests.

### 9.2 `configs/smoke_real_providers.json` — opt-in real smoke

New file. Same shape as `example_providers.json` but:

- `input_generation.enabled: true`
- `input_generation.target_inputs: 100`
- `input_generation.providers: [deepseek_v4_pro, gemini_flash]` with weight 50/50.
- Models pinned to known defaults (`"deepseek-chat"`, `"gemini-2.5-flash"`).

This is the file users explicitly opt into for a real-API spend. Cost: single-digit cents for 100 inputs.

---

## 10. Requirements

`requirements-train.txt` additions:

```text
openai>=1.50
google-genai
```

`google-genai` is left unpinned in the spec. The implementation plan's Task 0 includes:

```bash
pip install google-genai
python -c "from google import genai; from google.genai import types, errors; print('ok')"
python -c "import importlib.metadata as m; print(m.version('google-genai'))"
```

The third command prints the installed version; pin that as the floor in `requirements-train.txt` (e.g., `google-genai>=0.4.0` if the installed version is `0.4.x`).

---

## 11. Rollout

Single commit on `feat/multi-provider-slice2`. Same pattern as Slice 1.

1. **Task 0** — Preflight: verify prior slice imports, install + verify SDKs, pin `google-genai` floor.
2. **Task 1** — `scripts/_retry.py` + `tests/test_retry.py` → green.
3. **Task 2** — Move `clean_input_line` + add `normalize_input` to `_lib.py` (with the safer regex). Re-export from `05_generate_distillation_data.py`. Add `tests/test_clean_input_line.py`.
4. **Task 3** — Update `read_inputs_jsonl` malformed-row policy. Inline test.
5. **Task 4** — Add `DeepSeekProvider` + `_parse_input_lines` to `scripts/llm_providers.py`. Wire `create_provider` factory branch. DeepSeek half of `tests/test_real_providers.py`.
6. **Task 5** — Add `GeminiProvider`. Gemini half of `test_real_providers.py`.
7. **Task 6** — Add `DISTILL_DIR_OVERRIDE` support to constants block.
8. **Task 7** — Add `phase_inputs_multi_provider` + helpers.
9. **Task 8** — Rename `_enforce_slice1_flag_contract` → `_enforce_flag_contract`, add Slice 2 phase rules, update `main()` dispatch. Update `test_stage5_flag_contract.py` rows.
10. **Task 9** — Add `tests/test_multi_provider_inputs.py` end-to-end suite.
11. **Task 10** — Update `requirements-train.txt`, configs, `docs/provider_config.md`, README.
12. **Task 11** — Full-suite green + coverage + manual smokes (dry-run; fake-only run; `--help` confirms new wording).
13. **Task 12** — Single commit.

---

## 12. Manual smokes

```bash
# 1. Dry-run still works (unchanged from Slice 1)
python scripts/05_generate_distillation_data.py \
    --provider-config configs/test_providers.json \
    --dry-run-quota

# 2. Multi-provider input generation against FakeProvider (no API spend)
DISTILL_DIR_OVERRIDE=/tmp/slice2_smoke \
python scripts/05_generate_distillation_data.py \
    --phase inputs \
    --provider-config configs/test_providers.json \
    --multi-provider
# Expected: 10 inputs in /tmp/slice2_smoke/inputs_raw.jsonl,
#           each row has _source, _provider, _model:null, _batch_id

# 3. Confirm phase restrictions
python scripts/05_generate_distillation_data.py \
    --provider-config configs/test_providers.json \
    --multi-provider --phase label
# exit 2, stderr mentions "Slice 3"

# 4. Real API smoke (opt-in, costs real money)
export DEEPSEEK_API_KEY=sk-...
export GOOGLE_API_KEY=...
DISTILL_DIR_OVERRIDE=/tmp/real_smoke \
python scripts/05_generate_distillation_data.py \
    --phase inputs \
    --provider-config configs/smoke_real_providers.json \
    --multi-provider
```

Only smoke #4 costs real money. Expected: ~100 unique inputs in `/tmp/real_smoke/inputs_raw.jsonl`, mix of `_provider: deepseek_v4_pro` and `_provider: gemini_flash` rows.

---

## 13. Public-surface dependency graph

```
_retry.py
  └─ stdlib only (logging, random, time, typing)

llm_providers.py
  ├─ stdlib (json, os, threading, pathlib, typing)
  ├─ defines _parse_input_lines
  ├─ _parse_input_lines lazily imports clean_input_line and normalize_input from _lib
  ├─ lazy SDK imports inside DeepSeekProvider/GeminiProvider __init__
  └─ no top-level SDK imports — `import llm_providers` stays SDK-free

generation_config.py    (unchanged from Slice 1)
generation_orchestrator.py    (unchanged from Slice 1)
probe_validator.py    (unchanged from Slice 1)

_lib.py
  └─ +clean_input_line, +normalize_input, +_LINE_PREFIX_RE (with safer regex)

05_generate_distillation_data.py
  ├─ +phase_inputs_multi_provider + helpers
  ├─ +_enforce_flag_contract (renamed)
  ├─ +DISTILL_DIR_OVERRIDE
  ├─ +read_inputs_jsonl malformed-row policy
  └─ legacy phase_inputs/phase_label/phase_eval unchanged

NEW deps (lazy): openai>=1.50, google-genai
NO changes to: amount_parser, validator, 04_eval, 03_train_teacher, 06_train_student, _training
```
