# Multi-Provider Generation — Slice 3 Design Spec

**Date:** 2026-05-16
**Status:** Approved, ready for implementation plan
**Scope:** Slice 3 of 4 in the multi-provider generation roadmap — validator-gated multi-provider output labeling.
**Prerequisite:** Slices 1+2 already merged (commits on `feat/multi-provider-generation` up through `f999c4e`). Slice 3 branches off as `feat/multi-provider-slice3`.

---

## 1. Problem

Slice 2 made `--phase inputs --multi-provider` actually run with real DeepSeek + Gemini API calls, but `--phase label --multi-provider` still exits via `parser.error` with the "reserved for Slice 3" message. Stage 5 label generation remains single-teacher (legacy `phase_label`) regardless of `--provider-config`.

Slice 3 closes that gap: `--phase label --multi-provider --provider-config <path>` runs a validator-gated, multi-provider output-labeling loop with candidate scoring, best-of-N selection across providers, and repair-on-failure retry. Legacy `--phase label` (no `--multi-provider`) keeps working byte-identically.

Real `generate_label` implementations land on `DeepSeekProvider` + `GeminiProvider` (with Gemini structured output). A new `LocalTeacherProvider` wraps the existing fp16 Unsloth backend.

## 2. Solution overview

```
scripts/
  llm_providers.py            (MODIFY)  Real generate_label on DeepSeek + Gemini. New
                                        LocalTeacherProvider with lazy backend load + lock.
                                        Gemini structured-output schema for labels only.
                                        Factory updates: local_teacher requires cfg.model
                                        (path to adapter dir); ConfigError if missing.
  label_selection.py          (NEW)     Pure functions: score_candidate, pick_best,
                                        build_repair_prompt. CandidateOutcome dataclass.
                                        No model/API/I/O deps.
  _lib.py                     (MODIFY-ADDITIVE)  Add build_teacher_fp16_backend +
                                        _FP16Backend helper. Heavy imports inside the
                                        function only — `import _lib` stays light.
                                        Existing schema/SYSTEM_PROMPT/validator/parser
                                        helpers unchanged.
  05_generate_distillation_data.py  (MODIFY)  Add phase_label_multi_provider; remove
                                        the Slice 2 "label is reserved for Slice 3"
                                        contract error; main() dispatch handles both
                                        --phase inputs and --phase label under
                                        --multi-provider.

configs/
  example_providers.json      (UNCHANGED)  Stays safe: both phases enabled=false.
  smoke_real_providers.json   (MODIFY)  Add output_generation block (opt-in real labels).
  smoke_local_teacher_providers.json (NEW)  Opt-in local-teacher + Gemini labeling config.
                                        local_teacher provider's `model` field points to
                                        models/teacher/adapters.

tests/
  conftest.py                 (MODIFY)  Add shared SDK-stub fixture (opt-in, not autouse).
  test_candidate_scoring.py   (NEW)
  test_label_providers.py     (NEW)
  test_label_orchestrator.py  (NEW)
  test_multi_provider_labels.py (NEW)   Subprocess end-to-end.
  test_stage5_flag_contract.py (MODIFY)
  fixtures/
    fake_labels_with_failures.jsonl (NEW)

requirements-train.txt        (UNCHANGED)  openai + google-genai already present from Slice 2.
README.md                     (MODIFY)
docs/provider_config.md       (MODIFY)
```

### Out of scope (Slice 4 and later)

- Per-provider cost, latency, token, acceptance-rate metrics.
- Ensemble amount-voting across candidates.
- Eval-set expansion or back-cleaning of existing `train.jsonl`.
- Qwen3-0.6B student-candidate benchmark.

---

## 3. Provider-side label implementations

### 3.1 SDK choice

Already pinned by Slice 2:
- DeepSeek via `openai>=1.50` (OpenAI-compatible endpoint).
- Gemini via `google-genai>=1.73.1`.

No new top-level deps in this slice. Plan's Task 0 verifies SDK imports still resolve.

### 3.2 Lazy SDK imports

Unchanged from Slice 2: `scripts/llm_providers.py` imports no SDK at module load. SDK imports happen inside provider `__init__` only. `import llm_providers` stays SDK-free.

### 3.3 `DeepSeekProvider.generate_label`

Replace the existing `NotImplementedError` body:

```python
def generate_label(self, input_text: str) -> str:
    """Single chat-completion call to label this input. Returns raw text;
    the orchestrator validates."""
    from _lib import build_messages
    msgs = build_messages(input_text)
    return self._call_with_retry_label(msgs)


def _call_with_retry_label(self, messages: list[dict]) -> str:
    from _retry import retry_with_backoff
    return retry_with_backoff(
        lambda: self._call_api_messages(messages),
        retryable=self._retryable_excs,
        max_retries=self.max_retries,
        logger_name=f"deepseek.{self.name}",
        sleep=self._sleep,
    )


def _call_api_messages(self, messages: list[dict]) -> str:
    kwargs = {
        "model": self.model,
        "messages": messages,
        "timeout": self.timeout,
    }
    if self.temperature is not None:
        kwargs["temperature"] = self.temperature
    if self.max_tokens is not None:
        kwargs["max_tokens"] = self.max_tokens
    resp = self._client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
```

`_call_api_messages` is new (passes message list); `_call_api` from Slice 2 stays for `generate_inputs`. No `OPENAI_API_KEY` fallback. Same retry semantics as inputs.

### 3.4 `GeminiProvider.generate_label`

```python
def generate_label(self, input_text: str) -> str:
    return self._call_with_retry_label(input_text)


def _call_with_retry_label(self, input_text: str) -> str:
    from _retry import retry_with_backoff
    return retry_with_backoff(
        lambda: self._call_api_label(input_text),
        retryable=self._retryable_excs,
        is_retryable=_is_retryable_gemini_error,
        max_retries=self.max_retries,
        logger_name=f"gemini.{self.name}",
        sleep=self._sleep,
    )


def _call_api_label(self, input_text: str) -> str:
    from google.genai import types
    from _lib import SYSTEM_PROMPT
    cfg_kwargs = {"system_instruction": SYSTEM_PROMPT}
    if self.temperature is not None:
        cfg_kwargs["temperature"] = self.temperature
    if self.max_tokens is not None:
        cfg_kwargs["max_output_tokens"] = self.max_tokens
    if self.structured_output:
        cfg_kwargs["response_mime_type"] = "application/json"
        cfg_kwargs["response_schema"] = _label_schema_for_gemini()
    config = types.GenerateContentConfig(**cfg_kwargs)
    resp = self._client.models.generate_content(
        model=self.model,
        contents=input_text,
        config=config,
    )
    return resp.text or ""
```

`structured_output` applies only to `generate_label`; `generate_inputs` ignores it. The schema returned by `_label_schema_for_gemini()` is verified at implementation time against the installed `google-genai` SDK (raw dict expected to work; fall back to `types.Schema` if SDK rejects).

### 3.5 `_label_schema_for_gemini()`

```python
def _label_schema_for_gemini() -> dict:
    """Gemini-shaped schema for transaction labels. Enums sourced from
    _lib so they stay in lockstep with the validator."""
    from _lib import CATEGORIES, TYPES, CURRENCIES
    return {
        "type": "OBJECT",
        "properties": {
            "transactions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "amount":   {"type": "NUMBER"},
                        "currency": {"type": "STRING", "enum": list(CURRENCIES)},
                        "item":     {"type": "STRING"},
                        "category": {"type": "STRING", "enum": list(CATEGORIES)},
                        "type":     {"type": "STRING", "enum": list(TYPES)},
                    },
                    "required": ["amount", "currency", "item", "category", "type"],
                },
            },
        },
        "required": ["transactions"],
    }
```

No enum duplication — Slice 0's `_lib.CATEGORIES`/`TYPES`/`CURRENCIES` are the source of truth.

### 3.6 `LocalTeacherProvider`

```python
class LocalTeacherProvider:
    """Real local teacher via Unsloth + transformers fp16 backend.

    Lazy backend load: __init__ stashes config; the first generate_label
    call builds the backend. A threading.Lock protects both init and
    inference (single GPU is not safely concurrent).
    """

    provider_type = "local_teacher"

    def __init__(
        self,
        name: str,
        *,
        adapter_dir: Path,
        max_seq_length: int = 1024,
        max_new_tokens: int = 384,
    ) -> None:
        self.name = name
        self.model = None   # populated lazily after backend load
        self.adapter_dir = adapter_dir
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self._backend = None
        self._lock = threading.Lock()

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        raise NotImplementedError(
            f"local_teacher provider {self.name!r}: input generation is not "
            f"supported — local_teacher is label-only."
        )

    def generate_label(self, input_text: str) -> str:
        with self._lock:
            if self._backend is None:
                from _lib import build_teacher_fp16_backend
                if not self.adapter_dir.exists():
                    raise ProviderError(
                        f"LocalTeacherProvider {self.name!r}: adapter dir not found at {self.adapter_dir}"
                    )
                self._backend = build_teacher_fp16_backend(
                    adapter_dir=self.adapter_dir,
                    max_seq_length=self.max_seq_length,
                )
                self.model = str(self.adapter_dir.name)
            from _lib import build_messages
            msgs = build_messages(input_text)
            return self._backend.generate_label(msgs, max_new_tokens=self.max_new_tokens)
```

Invariants:
- `__init__` does zero I/O — no model load, no torch import.
- `generate_inputs` always raises (local teacher is label-only by design).
- `_lock` serializes both init and every inference call.
- Repair prompts go through the same path; quality may be weaker because local teacher was trained on raw utterances, not repair prompts. Acceptable for v1.

### 3.7 Factory update

In `create_provider`:

```python
if cfg.provider_type == "local_teacher":
    if not cfg.model:
        raise ConfigError(
            f"local_teacher provider {cfg.name!r} requires `model` "
            f"(path to the adapter directory)."
        )
    return LocalTeacherProvider(
        name=cfg.name,
        adapter_dir=Path(cfg.model),
        max_seq_length=1024,
        max_new_tokens=cfg.max_tokens or 384,
    )
```

`cfg.model` is repurposed for `local_teacher` as the adapter directory path. `max_seq_length` is fixed at 1024 — not user-tunable in v1 (no config field for it). `max_tokens` doubles as `max_new_tokens`. These overloads documented in `docs/provider_config.md`.

The `deepseek` and `gemini` factory branches from Slice 2 are unchanged (they continue to construct real providers with lazy SDK imports).

### 3.8 `_lib.build_teacher_fp16_backend` + `_FP16Backend`

Additive helpers in `_lib.py`. All heavy imports happen INSIDE the function bodies — `import _lib` stays light.

```python
def build_teacher_fp16_backend(
    *,
    adapter_dir: Path,
    max_seq_length: int = 1024,
) -> "_FP16Backend":
    """Build an fp16 transformers backend wrapping the Unsloth-trained
    teacher LoRA adapter. Heavy imports (unsloth, torch) happen inside
    this function so `import _lib` stays light.
    """
    from unsloth import FastLanguageModel
    model, processor = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return _FP16Backend(model=model, processor=processor, tokenizer=tokenizer)


class _FP16Backend:
    """Wraps the loaded model + tokenizer. NOT thread-safe;
    LocalTeacherProvider owns the lock."""

    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    def generate_label(self, messages: list[dict], *, max_new_tokens: int) -> str:
        import torch
        templater = (
            self.processor if hasattr(self.processor, "apply_chat_template") else self.tokenizer
        )
        prompt = templater.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = self.tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True,
            max_length=self.tokenizer.model_max_length or 1024,
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
```

No `temperature` / `top_p` arguments to `.generate()` — only `do_sample=False`, `max_new_tokens`, `pad_token_id`. Avoids HF deterministic-generation warnings.

Legacy `phase_label`'s `_build_label_backend` is refactored as a thin wrapper that calls `build_teacher_fp16_backend` for the transformers branch and continues using raw `llama-cpp-python` for the GGUF branch. Legacy behavior is unchanged.

---

## 4. `scripts/label_selection.py` — pure scoring logic

No I/O, no model deps. Imports only stdlib + `_lib` constants.

### 4.1 `CandidateOutcome`

```python
@dataclass(frozen=True)
class CandidateOutcome:
    provider: str                            # ProviderConfig.name
    model: str | None                        # provider.model
    raw_output: str                          # what the provider returned
    parsed_output: dict | None               # extract_json(raw_output); None on parse fail
    validation: dict | None                  # serialize_validation_result; None on parse fail
    failure_reason: str | None               # None if accepted; otherwise:
                                             #   provider_error, json_parse_failed,
                                             #   schema_invalid, validation_failed,
                                             #   superseded_amount_used, currency_mismatch,
                                             #   suspicious_duplicate
    score: int                               # 0..100; 0 for any failure
    is_repair_attempt: bool
    provider_priority_rank: int | None = None  # None if no provider_priority configured
    error: str | None = None                 # exception text for provider_error; else None
```

### 4.2 `score_candidate` — pinned formula (hard-reject on any validator error)

```python
def score_candidate(
    parsed_output: dict | None,
    validation: dict | None,
    failure_reason: str | None,
    provider_priority_rank: int | None = None,
) -> int:
    """Hard-reject any candidate with a failure_reason or any error-severity
    validator failure. Survivors get a base 80 (clean validator pass) plus
    bonuses for count match (+10), no warnings (+5), and provider priority
    rank 0 (+5). Max 100.
    """
    if failure_reason is not None or parsed_output is None or validation is None:
        return 0
    error_codes = {
        e["code"] for e in validation.get("errors", []) if e["severity"] == "error"
    }
    if error_codes:
        return 0
    score = 80
    if validation.get("txn_count_matches_active_candidates"):
        score += 10
    warning_codes = {
        e["code"] for e in validation.get("errors", []) if e["severity"] == "warning"
    }
    if not warning_codes:
        score += 5
    if provider_priority_rank == 0:
        score += 5
    return score
```

Possible scores: `0` (any failure or any error-severity validator code), `80` (clean pass, no count match, has warnings, not top priority), up to `100` (clean pass + count match + no warnings + top priority).

### 4.3 `pick_best`

```python
def pick_best(outcomes: list[CandidateOutcome]) -> CandidateOutcome | None:
    """Return the highest-scoring non-failure outcome, or None if all failed.

    Tie-break order:
      1. Higher score wins.
      2. Lower provider_priority_rank wins (None treated as +inf).
      3. Alphabetical provider name (stable, deterministic).
    """
    valid = [o for o in outcomes if o.failure_reason is None and o.score > 0]
    if not valid:
        return None
    valid.sort(key=lambda o: (
        -o.score,
        o.provider_priority_rank if o.provider_priority_rank is not None else float("inf"),
        o.provider,
    ))
    return valid[0]
```

### 4.4 `build_repair_prompt`

```python
_REPAIR_PROMPT_TEMPLATE = """The previous label attempt for this input failed validation.

INPUT:
{input_text}

PARSED AMOUNT CANDIDATES (from deterministic parser):
{candidates_summary}

FAILURE SUMMARY:
{failure_summary}

Generate a corrected JSON label that:
1. Uses ONLY amounts from the parsed candidates above (do not invent new amounts).
2. Marks the correct currency per the candidate's currency_hint.
3. Has the correct number of transactions matching the active candidates.
4. Does NOT include any superseded amounts (markers like "wait no", "actually").
5. Does NOT duplicate transactions unless the input explicitly repeats them.

Output ONLY the JSON object."""


def build_repair_prompt(
    input_text: str,
    *,
    candidates: list[dict] | None = None,
    failure_summary: str,
) -> str:
    if candidates:
        lines = [
            f"  - {c['raw']!r} -> {c['value']} "
            f"({c.get('currency_hint') or '(no hint, default INR)'}, {c['status']})"
            for c in candidates
        ]
        candidates_summary = "\n".join(lines)
    else:
        candidates_summary = "  (no amount candidates parsed)"
    return _REPAIR_PROMPT_TEMPLATE.format(
        input_text=input_text,
        candidates_summary=candidates_summary,
        failure_summary=failure_summary,
    )
```

`failure_summary` is built by the orchestrator (≤ 200 chars). Caller decides whether to pass the prompt as a user message or use the existing `build_messages` flow.

---

## 5. Orchestrator: `phase_label_multi_provider`

Lives in `scripts/05_generate_distillation_data.py`. Per-input ThreadPoolExecutor; main thread is sole writer.

### 5.1 Per-input flow

```
for each pending input:
    outcomes = []
    for attempt_idx in range(label_attempts_per_input):
        provider_name = scheduler.next_provider()
        outcomes.append(_try_one_attempt(input, provider, ...))
    best = pick_best(outcomes)
    if best is not None:
        emit train.jsonl row
        continue
    # All initial attempts failed
    if validation.retry_invalid_with_stricter_prompt and validation.max_repair_attempts > 0:
        repair_provider = _highest_priority_provider(...)
        for repair_idx in range(validation.max_repair_attempts):
            repair_outcome = _try_one_attempt(input, repair_provider, is_repair=True, ...)
            outcomes.append(repair_outcome)
            if repair_outcome.failure_reason is None:
                emit train.jsonl row (best = repair_outcome)
                break
        else:
            emit failed.jsonl row with reason='repair_exhausted'
    else:
        emit failed.jsonl row with reason='all_candidates_failed'
    # If every attempt was provider_error, top-level reason becomes 'provider_error'.
```

### 5.2 `_try_one_attempt` — never raises

```python
def _try_one_attempt(
    input_text: str,
    provider,
    pcfg,
    *,
    is_repair: bool = False,
    failure_summary: str = "",
    parser_candidates: list[dict] | None = None,
    provider_priority_rank: int | None = None,
) -> CandidateOutcome:
    """Run one provider attempt, validate, score. Never raises."""
    from label_selection import score_candidate, build_repair_prompt
    from _lib import extract_json, validate_example, serialize_validation_result

    try:
        if is_repair:
            prompt = build_repair_prompt(
                input_text,
                candidates=parser_candidates,
                failure_summary=failure_summary,
            )
            raw = provider.generate_label(prompt)
        else:
            raw = provider.generate_label(input_text)
    except Exception as e:    # noqa: BLE001 — unify all provider failures
        return CandidateOutcome(
            provider=pcfg.name,
            model=getattr(provider, "model", None),
            raw_output="",
            parsed_output=None,
            validation=None,
            failure_reason="provider_error",
            score=0,
            is_repair_attempt=is_repair,
            provider_priority_rank=provider_priority_rank,
            error=repr(e),
        )

    parsed = extract_json(raw)
    if parsed is None:
        return CandidateOutcome(
            provider=pcfg.name,
            model=getattr(provider, "model", None),
            raw_output=raw, parsed_output=None, validation=None,
            failure_reason="json_parse_failed",
            score=0, is_repair_attempt=is_repair,
            provider_priority_rank=provider_priority_rank,
        )

    result = validate_example(input_text, parsed, mode="strict")
    validation_dict = serialize_validation_result(result)

    error_codes = {
        e["code"] for e in validation_dict["errors"] if e["severity"] == "error"
    }
    if "SCHEMA_INVALID" in error_codes:
        failure_reason = "schema_invalid"
    elif "SUPERSEDED_AMOUNT_USED" in error_codes:
        failure_reason = "superseded_amount_used"
    elif "CURRENCY_MISMATCH" in error_codes:
        failure_reason = "currency_mismatch"
    elif "SUSPICIOUS_DUPLICATE" in error_codes:
        failure_reason = "suspicious_duplicate"
    elif error_codes:
        failure_reason = "validation_failed"   # catch-all for AMOUNT_NOT_IN_INPUT, NO_AMOUNT_IN_INPUT, etc.
    else:
        failure_reason = None

    score = score_candidate(
        parsed_output=parsed,
        validation=validation_dict,
        failure_reason=failure_reason,
        provider_priority_rank=provider_priority_rank,
    )
    return CandidateOutcome(
        provider=pcfg.name,
        model=getattr(provider, "model", None),
        raw_output=raw,
        parsed_output=parsed,
        validation=validation_dict,
        failure_reason=failure_reason,
        score=score,
        is_repair_attempt=is_repair,
        provider_priority_rank=provider_priority_rank,
    )
```

### 5.3 Per-input worker `_process_one_input`

```python
def _process_one_input(
    input_text: str,
    *,
    providers_by_name: dict,
    pcfgs_by_name: dict,
    scheduler,
    priority_map: dict[str, int],
    cfg,
    parser_candidates: list[dict],
) -> tuple[str, "CandidateOutcome | None", list[CandidateOutcome]]:
    from label_selection import pick_best
    outcomes: list[CandidateOutcome] = []

    for _ in range(cfg.output_generation.label_attempts_per_input):
        provider_name = scheduler.next_provider()
        outcomes.append(_try_one_attempt(
            input_text,
            provider=providers_by_name[provider_name],
            pcfg=pcfgs_by_name[provider_name],
            provider_priority_rank=priority_map.get(provider_name),
        ))

    best = pick_best(outcomes)
    if best is not None:
        return (input_text, best, outcomes)

    repair_enabled = (
        cfg.validation.retry_invalid_with_stricter_prompt
        and cfg.validation.max_repair_attempts > 0
    )
    if repair_enabled:
        repair_provider_name = _highest_priority_provider(cfg, providers_by_name)
        for _ in range(cfg.validation.max_repair_attempts):
            repair_outcome = _try_one_attempt(
                input_text,
                provider=providers_by_name[repair_provider_name],
                pcfg=pcfgs_by_name[repair_provider_name],
                is_repair=True,
                failure_summary=_summarize_failures(outcomes),
                parser_candidates=parser_candidates,
                provider_priority_rank=priority_map.get(repair_provider_name),
            )
            outcomes.append(repair_outcome)
            if repair_outcome.failure_reason is None:
                return (input_text, repair_outcome, outcomes)

    return (input_text, None, outcomes)


def _highest_priority_provider(cfg, providers_by_name: dict) -> str:
    """First entry of provider_priority, falling back to first declared provider.
    Raises ValueError if the resolved name isn't in providers_by_name (defensive
    check on top of config validation)."""
    if cfg.output_generation.provider_priority:
        name = cfg.output_generation.provider_priority[0]
    else:
        name = cfg.output_generation.providers[0].name
    if name not in providers_by_name:
        raise ValueError(
            f"_highest_priority_provider: {name!r} not in providers_by_name"
        )
    return name


def _summarize_failures(outcomes: list["CandidateOutcome"]) -> str:
    """≤200-char summary of why all attempts failed."""
    parts: list[str] = []
    for o in outcomes:
        if o.failure_reason == "provider_error":
            parts.append(f"{o.provider}: provider error")
        elif o.validation:
            codes = [e["code"] for e in o.validation["errors"] if e["severity"] == "error"]
            parts.append(f"{o.provider}: " + ", ".join(codes[:3]))
        elif o.failure_reason:
            parts.append(f"{o.provider}: {o.failure_reason}")
    return "; ".join(parts)[:200]
```

### 5.4 Top-level loop

```python
def phase_label_multi_provider(args: argparse.Namespace) -> None:
    from generation_config import load_generation_config
    from generation_orchestrator import RoundRobinScheduler
    from llm_providers import create_provider
    from _lib import parse_amounts, normalize_input
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cfg = load_generation_config(args.provider_config)
    og = cfg.output_generation
    if not og.enabled:
        logging.info("Phase label (multi-provider): output_generation.enabled=False — nothing to do.")
        return

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    all_inputs = read_inputs_jsonl(INPUTS_FILE)

    # Normalize-dedupe the input list (preserve first-seen order).
    seen_norm: set[str] = set()
    all_inputs_unique: list[str] = []
    for inp in all_inputs:
        key = normalize_input(inp)
        if key in seen_norm:
            continue
        seen_norm.add(key)
        all_inputs_unique.append(inp)

    labeled_keys = {
        normalize_input(s) for s in _read_labeled_inputs(TRAIN_FILE)
    }
    failed_skip_keys = {
        normalize_input(s) for s in _read_failed_skip_set(FAILED_FILE, args)
    }
    pending = [
        inp for inp in all_inputs_unique
        if normalize_input(inp) not in labeled_keys
        and normalize_input(inp) not in failed_skip_keys
    ]
    if args.limit > 0:
        pending = pending[:args.limit]
    logging.info(
        "Phase label (multi-provider): all_unique=%d labeled=%d failed_skip=%d pending=%d",
        len(all_inputs_unique), len(labeled_keys), len(failed_skip_keys), len(pending),
    )
    if not pending:
        return

    providers_by_name = {p.name: create_provider(p) for p in og.providers}
    pcfgs_by_name = {p.name: p for p in og.providers}
    scheduler = RoundRobinScheduler(og.providers)
    priority_map = _build_priority_map(og)

    pool_size = min(cfg.rate_limits.global_max_workers, max(1, len(pending)))

    n_kept = 0
    n_failed = 0
    n_repaired = 0
    failure_reasons: Counter[str] = Counter()
    attempt_failure_reasons: Counter[str] = Counter()

    with ThreadPoolExecutor(max_workers=pool_size) as pool, \
         TRAIN_FILE.open("a", encoding="utf-8") as train_out, \
         FAILED_FILE.open("a", encoding="utf-8") as fail_out, \
         tqdm(total=len(pending), desc="multi-provider labels") as bar:

        candidates_by_input = {
            inp: [_serialize_candidate(c) for c in parse_amounts(inp)]
            for inp in pending
        }

        futures = {
            pool.submit(
                _process_one_input,
                inp,
                providers_by_name=providers_by_name,
                pcfgs_by_name=pcfgs_by_name,
                scheduler=scheduler,
                priority_map=priority_map,
                cfg=cfg,
                parser_candidates=candidates_by_input[inp],
            ): inp
            for inp in pending
        }

        for fut in as_completed(futures):
            input_text, best, all_outcomes = fut.result()
            for o in all_outcomes:
                if o.failure_reason:
                    attempt_failure_reasons[o.failure_reason] += 1

            if best is not None:
                row = {
                    "input": input_text,
                    "output": best.parsed_output,
                    "_source": "multi_provider_label",
                    "_provider": best.provider,
                    "_model": best.model,
                    "_validation_score": best.score,
                    "_attempts": len(all_outcomes),
                }
                train_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_kept += 1
                if best.is_repair_attempt:
                    n_repaired += 1
            else:
                repair_enabled = (
                    cfg.validation.retry_invalid_with_stricter_prompt
                    and cfg.validation.max_repair_attempts > 0
                )
                if all(o.failure_reason == "provider_error" for o in all_outcomes):
                    top_reason = "provider_error"
                else:
                    top_reason = "repair_exhausted" if repair_enabled else "all_candidates_failed"
                failed_row = {
                    "input": input_text,
                    "reason": top_reason,
                    "candidates": candidates_by_input[input_text],
                    "attempts": [_serialize_outcome(o) for o in all_outcomes],
                }
                fail_out.write(json.dumps(failed_row, ensure_ascii=False) + "\n")
                n_failed += 1
                failure_reasons[top_reason] += 1
            if (n_kept + n_failed) % cfg.rate_limits.write_flush_every == 0:
                train_out.flush()
                fail_out.flush()
            bar.update(1)

        train_out.flush()
        fail_out.flush()

    keep_rate = (n_kept / max(1, n_kept + n_failed)) * 100
    logging.info(
        "Phase label (multi-provider) done. kept=%d (repair=%d) failed=%d keep_rate=%.1f%%",
        n_kept, n_repaired, n_failed, keep_rate,
    )
    if failure_reasons:
        logging.info("Top-level failure reasons: %s", dict(failure_reasons))
    if attempt_failure_reasons:
        logging.info("Attempt-level failure reasons: %s", dict(attempt_failure_reasons))
```

### 5.5 Resume helpers

```python
def _read_labeled_inputs(path: Path) -> set[str]:
    """Set of inputs already present in train.jsonl. Tolerant of legacy rows."""
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("input"), str):
                out.add(obj["input"])
    return out


def _row_has_validation_failure_attempt(obj: dict) -> bool:
    """True if any attempt in the row has a validator-class failure_reason."""
    validator_reasons = {
        "validation_failed", "schema_invalid", "superseded_amount_used",
        "currency_mismatch", "suspicious_duplicate",
    }
    for attempt in obj.get("attempts", []):
        if attempt.get("failure_reason") in validator_reasons:
            return True
    return False


def _read_failed_skip_set(path: Path, args: argparse.Namespace) -> set[str]:
    """Set of inputs to SKIP this run based on failed.jsonl + retry flags.

    - args.retry_failed: skip nothing (re-attempt all failed rows).
    - args.retry_validation_failed:
        * legacy validator-slice row (reason == 'validation_failed'): re-attempt.
        * Slice 3 aggregate row (reason in {all_candidates_failed, repair_exhausted})
          AND row has at least one validator-class attempt: re-attempt.
        * provider_error rows: skip (these need --retry-failed).
    - Default: skip every input present in failed.jsonl.
    """
    if not path.exists():
        return set()
    if args.retry_failed:
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("input"), str):
                continue
            reason = obj.get("reason")
            if args.retry_validation_failed:
                if reason == "validation_failed":
                    continue
                if reason in {"all_candidates_failed", "repair_exhausted"} \
                        and _row_has_validation_failure_attempt(obj):
                    continue
            out.add(obj["input"])
    return out


def _build_priority_map(og) -> dict[str, int]:
    if og.provider_priority:
        return {name: i for i, name in enumerate(og.provider_priority)}
    return {p.name: i for i, p in enumerate(og.providers)}


def _serialize_candidate(c) -> dict:
    return {
        "value": c.value, "raw": c.raw, "span": list(c.span),
        "status": c.status, "source": c.source, "currency_hint": c.currency_hint,
    }


def _serialize_outcome(o: "CandidateOutcome") -> dict:
    return {
        "provider": o.provider,
        "model": o.model,
        "raw_output": o.raw_output,
        "parsed_output": o.parsed_output,
        "validation": o.validation,
        "score": o.score,
        "failure_reason": o.failure_reason,
        "is_repair_attempt": o.is_repair_attempt,
        "provider_priority_rank": o.provider_priority_rank,
        "error": o.error,
    }
```

### 5.6 Concurrency invariants

- **Per-input worker pool**: each input is one task. Pool size = `min(global_max_workers, len(pending))`.
- **Main thread is sole writer**: workers return outcomes via `as_completed`; main thread writes JSONL.
- **`LocalTeacherProvider` serializes internally** via lock. Multiple workers calling it queue on the lock.
- **API providers** are thread-safe at the SDK level. Concurrent calls to DeepSeek/Gemini are fine.
- **`RoundRobinScheduler`** already thread-safe (Slice 1).
- **Provider exceptions are caught and produce `provider_error` outcomes.** True network hangs depend on provider-level timeouts (DeepSeek `timeout=300`; Gemini SDK default verified at implementation time; LocalTeacher GPU-bound). No executor-level timeout is added in Slice 3; `--limit` and per-input retry caps keep total wall time bounded.

---

## 6. Output file shapes

### 6.1 `train.jsonl` accepted-row shape

```json
{
  "input": "500 rs on beer 50 rs on candy",
  "output": {"transactions": [...]},
  "_source": "multi_provider_label",
  "_provider": "gemini_flash",
  "_model": "gemini-2.5-flash",
  "_validation_score": 95,
  "_attempts": 2
}
```

Legacy consumers reading only `input` + `output` continue to work. `_attempts` counts initial + repair attempts.

### 6.2 `failed.jsonl` failed-row shape (Slice 3 aggregate)

One row per input. Builds on the validator-slice shape; adds `attempts[]`.

```json
{
  "input": "...",
  "reason": "repair_exhausted",
  "candidates": [
    {"value": 500.0, "raw": "500", "span": [0, 3],
     "status": "active", "source": "digits", "currency_hint": null}
  ],
  "attempts": [
    {
      "provider": "gemini_flash",
      "model": "gemini-2.5-flash",
      "raw_output": "...",
      "parsed_output": {...},
      "validation": {...},
      "score": 0,
      "failure_reason": "validation_failed",
      "is_repair_attempt": false,
      "provider_priority_rank": 1,
      "error": null
    },
    {"provider": "deepseek_v4_pro", ..., "failure_reason": "json_parse_failed"},
    {"provider": "gemini_flash", ..., "failure_reason": "validation_failed", "is_repair_attempt": true}
  ]
}
```

Top-level `reason ∈ {all_candidates_failed, repair_exhausted, provider_error}`. Attempt-level `failure_reason ∈ {provider_error, json_parse_failed, schema_invalid, validation_failed, superseded_amount_used, currency_mismatch, suspicious_duplicate}`.

---

## 7. Stage 5 CLI updates

### 7.1 Flag matrix delta from Slice 2

Only one row changes:

| Invocation | Slice 2 | Slice 3 |
|---|---|---|
| `--provider-config <path> --multi-provider --phase label` | exit 2 ("reserved for Slice 3") | **runs `phase_label_multi_provider`. Exit 0.** |

Everything else (inputs runs, eval/all error, dry-run wins) stays exactly as Slice 2.

### 7.2 `_enforce_flag_contract` update

Delete the `if args.phase == "label":` block inside the multi-provider phase-restriction section. Keep `eval` and `all` blocks.

### 7.3 `main()` dispatch

```python
if args.multi_provider:
    assert args.phase in {"inputs", "label"}
    assert args.provider_config is not None
    if args.phase == "inputs":
        phase_inputs_multi_provider(args)
    else:
        phase_label_multi_provider(args)
    logging.info("Stage 5 done.")
    return 0
```

### 7.4 Help-text refresh

- `--provider-config`: "Path to a multi-provider generation config JSON. Use with `--dry-run-quota` to inspect the config, or with `--multi-provider --phase inputs` or `--phase label` to run real generation."
- `--multi-provider`: "Run multi-provider execution. Supports `--phase inputs` (real input generation) and `--phase label` (validator-gated output labeling). `--phase eval` is legacy-only — omit `--multi-provider` for eval. `--phase all` exits via parser.error."

### 7.5 Reused existing flags

No new flags. Slice 3 honors:
- `--limit` — truncates pending list.
- `--retry-failed` — re-attempts every failed row.
- `--retry-validation-failed` — re-attempts only rows whose attempts include a validator-class failure (see §5.5).

---

## 8. Tests

All pytest; no real API or GPU. Real-API + GPU smokes are manual CLI commands.

### 8.1 New + modified test files

| File | What it covers |
|---|---|
| `tests/test_candidate_scoring.py` (new) | `score_candidate`, `pick_best`, `build_repair_prompt`. Pure-function tests. ≥ 95% coverage on `label_selection.py`. |
| `tests/test_label_providers.py` (new) | DeepSeek + Gemini `generate_label` (with/without structured_output). LocalTeacherProvider construction + lazy backend load + lock serialization. SDK stubs from shared conftest fixture. |
| `tests/test_label_orchestrator.py` (new) | Direct unit tests for `_try_one_attempt` and `_process_one_input`. Each attempt-level failure-reason mapping. Repair-success and provider-raises paths (using raising-fake-provider stub). Fixture-verification test. |
| `tests/test_multi_provider_labels.py` (new) | End-to-end via subprocess + FakeProvider. Success path; scoring picks best; repair exhausted writes failed; all-json-parse-failed; resume safety; `--retry-validation-failed` re-attempts; `--limit` truncates. |
| `tests/test_stage5_flag_contract.py` (modify) | Delete the "label exits 2" test; add "label succeeds" test (reads first input from `fake_labels.jsonl` so the FakeProvider has a matching label). |
| `tests/conftest.py` (modify) | Add the SDK-stub fixture (opt-in, not autouse). Both `test_real_providers.py` and `test_label_providers.py` import it. |
| `tests/fixtures/fake_labels_with_failures.jsonl` (new) | 6 rows pairing inputs to specific attempt-level failure reasons. |

### 8.2 `fake_labels_with_failures.jsonl` — final 6 rows

```jsonl
{"input":"500 beer","output":{"transactions":[{"amount":500,"currency":"INR","item":"beer","category":"Drinks","type":"expense"}]}}
{"input":"schema fail case","raw_output":"{\"transactions\":[{\"amount\":100}]}"}
{"input":"json fail case","raw_output":"{not json"}
{"input":"no amount input","raw_output":"{\"transactions\":[{\"amount\":999,\"currency\":\"INR\",\"item\":\"thing\",\"category\":\"Other\",\"type\":\"expense\"}]}"}
{"input":"500 beer wait no 600 beer","raw_output":"{\"transactions\":[{\"amount\":500,\"currency\":\"INR\",\"item\":\"beer\",\"category\":\"Drinks\",\"type\":\"expense\"}]}"}
{"input":"500 beer once","raw_output":"{\"transactions\":[{\"amount\":500,\"currency\":\"INR\",\"item\":\"beer\",\"category\":\"Drinks\",\"type\":\"expense\"},{\"amount\":500,\"currency\":\"INR\",\"item\":\"beer\",\"category\":\"Drinks\",\"type\":\"expense\"}]}"}
```

Expected attempt-level failure reasons per row:

| Input | Expected `failure_reason` |
|---|---|
| `"500 beer"` | `None` (clean pass) |
| `"schema fail case"` | `schema_invalid` |
| `"json fail case"` | `json_parse_failed` |
| `"no amount input"` | `validation_failed` (NO_AMOUNT_IN_INPUT) |
| `"500 beer wait no 600 beer"` | `superseded_amount_used` |
| `"500 beer once"` | `suspicious_duplicate` |

The plan adds `test_fixture_failure_codes_match_expectation` to `test_label_orchestrator.py` to lock these in. If validator code changes alter any expected reason, that test breaks first.

### 8.3 Scoring tests — pinned expectations

Given the simplified scoring (any error-severity validator failure → 0):

```python
def test_score_perfect_candidate():
    """Clean validator pass + count match + no warnings + priority 0 → 100."""

def test_score_hard_reject_returns_zero():
    """failure_reason set → 0 regardless of validation."""

def test_score_any_error_severity_returns_zero():
    """Validator returned any error-severity code → 0."""

def test_score_clean_pass_no_priority_bonus():
    """Clean pass + count match + no warnings + priority=1 → 95."""

def test_score_warning_only_with_count_mismatch():
    """Warning + count mismatch + priority 0:
       80 base + 0 count + 0 warnings + 5 priority = 85."""
    # validation = _validation_with(
    #     [warning row],
    #     txn_count_matches_active_candidates=False,
    # )

def test_pick_best_returns_highest_score(): ...
def test_pick_best_returns_none_when_all_failed(): ...
def test_pick_best_ignores_zero_score_outcomes(): ...
def test_pick_best_tie_break_by_provider_priority_rank(): ...
def test_pick_best_tie_break_by_provider_name_when_priority_equal(): ...
def test_build_repair_prompt_includes_input_and_failures(): ...
def test_build_repair_prompt_handles_empty_candidates(): ...
```

`_validation_with(errors, **overrides)` helper accepts kwargs to vary `txn_count_matches_active_candidates` etc.

### 8.4 Test hygiene rules

- SDK fixture in `tests/conftest.py` is **opt-in** (not autouse). Tests reference it by parameter name when needed.
- Real-provider tests use fake exception classes for retry-behavior, not real SDK exception constructors.
- LocalTeacherProvider tests monkeypatch `_lib.build_teacher_fp16_backend` to return a fake backend. Unsloth/torch are NOT imported during these tests.
- Subprocess tests pass `env={"DISTILL_DIR_OVERRIDE": str(tmp_path)}`. Repo's `data/distill/` is never touched.
- `test_provider_error` uses a direct helper-level test (raising fake provider) rather than subprocess.

### 8.5 Coverage targets

- `scripts/label_selection.py`: ≥ 95% (pure functions; no excuses).
- `scripts/llm_providers.py`: ≥ 80% (real-provider label paths via mocked SDK; LocalTeacher backend mocked).
- `scripts/05_generate_distillation_data.py`: not in `--cov` target (subprocess-tested).

---

## 9. Config files

### 9.1 `configs/example_providers.json` — unchanged

Stays safe-by-default from Slice 2: both phases `enabled: false`. Documents shape only.

### 9.2 `configs/smoke_real_providers.json` — add output block

Slice 2 had `output_generation.enabled: false`. Slice 3 flips it to `true` with a real provider list:

```json
"output_generation": {
  "enabled": true,
  "label_attempts_per_input": 2,
  "selection_policy": "first_valid_then_score",
  "providers": [
    {"name": "deepseek_v4_pro", "type": "deepseek", "model": "deepseek-chat",
     "weight": 50, "threads": 4, "temperature": 0.0, "max_tokens": 512},
    {"name": "gemini_flash", "type": "gemini", "model": "gemini-2.5-flash",
     "weight": 50, "threads": 4, "temperature": 0.0, "max_tokens": 512,
     "structured_output": true}
  ],
  "provider_priority": ["gemini_flash", "deepseek_v4_pro"]
}
```

`validation` stays the same as Slice 2 (`retry_invalid_with_stricter_prompt: false`, `max_repair_attempts: 0`); users can enable repair manually.

### 9.3 `configs/smoke_local_teacher_providers.json` — new

```json
{
  "version": 1,
  "input_generation": {"enabled": false, "providers": []},
  "output_generation": {
    "enabled": true,
    "label_attempts_per_input": 2,
    "selection_policy": "first_valid_then_score",
    "providers": [
      {"name": "local_teacher", "type": "local_teacher",
       "model": "models/teacher/adapters",
       "weight": 70, "threads": 1, "max_tokens": 384},
      {"name": "gemini_flash", "type": "gemini", "model": "gemini-2.5-flash",
       "weight": 30, "threads": 4, "temperature": 0.0, "max_tokens": 512,
       "structured_output": true}
    ],
    "provider_priority": ["local_teacher", "gemini_flash"]
  },
  "validation": {"schema": true, "semantic_validator": true,
                 "reject_invalid": true,
                 "retry_invalid_with_stricter_prompt": true,
                 "max_repair_attempts": 1},
  "rate_limits": {"global_max_workers": 4, "write_flush_every": 5}
}
```

Requires GPU + trained teacher adapter at `models/teacher/adapters`. Opt-in only.

### 9.4 `docs/provider_config.md` — update

Update the "What loads and runs" section:

```markdown
**Slice 3 (real output labeling):**
- `--phase label --provider-config <path> --multi-provider` runs the validator-gated
  multi-provider labeling loop.
- `type: local_teacher` providers require `model` (path to the adapter directory).
- Gemini's `structured_output: true` activates response_schema + response_mime_type
  for label generation only. Input generation ignores it.
- `validation.retry_invalid_with_stricter_prompt: true` plus `max_repair_attempts: N`
  enables the repair loop: when all initial attempts fail, re-prompt the highest-
  priority provider with a stricter repair prompt up to N times.
```

---

## 10. Rollout

Single commit on `feat/multi-provider-slice3`.

1. **Task 0** — Preflight: confirm branch, prior slice imports, prior tests green, SDK imports resolve. If `unsloth` import fails: continue Slice 3 implementation/tests (LocalTeacherProvider tests monkeypatch the backend); mark GPU smoke as unavailable.
2. **Task 1** — `scripts/label_selection.py` + `tests/test_candidate_scoring.py` → green.
3. **Task 2** — `_lib.build_teacher_fp16_backend` + `_FP16Backend` (additive helpers; lazy imports inside the function).
4. **Task 3** — Move SDK-stub fixture to `tests/conftest.py`; Slice 2 tests still green.
5. **Task 4** — `DeepSeekProvider.generate_label` + `_call_api_messages` + tests.
6. **Task 5** — `GeminiProvider.generate_label` + `_call_api_label` + `_label_schema_for_gemini()` + tests.
7. **Task 6** — `LocalTeacherProvider` + factory wiring (requires `cfg.model`) + tests with monkeypatched backend.
8. **Task 7** — Add `phase_label_multi_provider` + helpers to `scripts/05_generate_distillation_data.py`.
9. **Task 8** — `tests/test_label_orchestrator.py` — direct unit tests + fixture-verification.
10. **Task 9** — Update `_enforce_flag_contract`; update `main()` dispatch; help-text refresh; update `tests/test_stage5_flag_contract.py`.
11. **Task 10** — Subprocess end-to-end suite in `tests/test_multi_provider_labels.py`.
12. **Task 11** — Add `tests/fixtures/fake_labels_with_failures.jsonl`. (May land with Task 8 if convenient.)
13. **Task 12** — Configs: `smoke_local_teacher_providers.json` (new), `smoke_real_providers.json` add output block, `provider_config.md` update.
14. **Task 13** — README "Slice 3: Multi-provider output labeling" subsection with corrected smoke command sequence.
15. **Task 14** — Full-suite green + coverage + manual smokes 1+2. Verify `git status --short` shows no `data/distill/*`, `.coverage`, `reports/`, or `htmlcov/` staged.
16. **Task 15** — Single commit.

---

## 11. Manual smokes

No API keys, no GPU:

```bash
# Smoke 1: input phase regression (Slice 2 still works)
DISTILL_DIR_OVERRIDE=/tmp/s3_smoke \
python scripts/05_generate_distillation_data.py \
    --phase inputs \
    --provider-config configs/test_providers.json \
    --multi-provider

# Smoke 2: label phase against fake providers (writes both train + failed)
DISTILL_DIR_OVERRIDE=/tmp/s3_smoke \
python scripts/05_generate_distillation_data.py \
    --phase label \
    --provider-config configs/test_providers.json \
    --multi-provider \
    --limit 5
# Expected: up to 5 processed rows across train.jsonl + failed.jsonl;
#           successful rows in train.jsonl have _source=multi_provider_label.

# Smoke 3: dry-run validates real-provider config without API calls
python scripts/05_generate_distillation_data.py \
    --provider-config configs/smoke_real_providers.json \
    --dry-run-quota
```

Opt-in (real API or GPU):

```bash
# Smoke 4 (real API): label phase with DeepSeek + Gemini
export DEEPSEEK_API_KEY=sk-...
export GOOGLE_API_KEY=...
DISTILL_DIR_OVERRIDE=/tmp/s3_real \
python scripts/05_generate_distillation_data.py \
    --phase inputs \
    --provider-config configs/smoke_real_providers.json \
    --multi-provider
DISTILL_DIR_OVERRIDE=/tmp/s3_real \
python scripts/05_generate_distillation_data.py \
    --phase label \
    --provider-config configs/smoke_real_providers.json \
    --multi-provider --limit 10
# Cost: small smoke run; includes real input-generation calls plus up to ~20 label calls. Cents.

# Smoke 5 (GPU): local teacher labeling
# Requires smoke_local_teacher_providers.json model path to point to an existing adapter directory.
DISTILL_DIR_OVERRIDE=/tmp/s3_local \
python scripts/05_generate_distillation_data.py \
    --phase label \
    --provider-config configs/smoke_local_teacher_providers.json \
    --multi-provider --limit 5
```

---

## 12. Decision points after rollout

For Smoke 4 (real API) output, run the validator probe against the produced labels:

```bash
python scripts/probe_validator.py \
    --input /tmp/s3_real/train.jsonl \
    --output reports/multi_provider_labels_probe.jsonl
```

The probe's pass-rate vs. legacy single-teacher labels is the headline diagnostic for Slice 3 quality. Thresholds (`>90%` good, `<70%` triage) apply to real-API smoke output only, not to deterministic fake fixtures.

For Smoke 2 (fake fixture), expected pass/fail is fixture-determined; tests assert the deterministic outcome.

---

## 13. What stays unchanged

- `scripts/_lib.py` schema / SYSTEM_PROMPT / validator / parser behavior unchanged. Only additive `build_teacher_fp16_backend` / `_FP16Backend` helper is added.
- `scripts/amount_parser.py`, `scripts/validator.py` — no edits.
- `scripts/04_eval.py`, `scripts/probe_validator.py` — no edits.
- `scripts/03_train_teacher.py`, `06_train_student.py`, `_training.py` — no edits.
- Slices 1+2 tests must stay green throughout (prior baseline at time of writing: ~244 tests).
- Legacy `phase_inputs` and `phase_eval` bodies unchanged. Legacy `phase_label` behavior is unchanged; its fp16 backend construction is refactored to call `_lib.build_teacher_fp16_backend` instead of constructing the backend inline.
