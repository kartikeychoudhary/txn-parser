"""Multi-provider abstraction for synthetic data + label generation.

Slice 1 ships:
  - LLMProvider Protocol (the public surface)
  - FakeProvider (fixture-backed; deterministic; used in tests + dry-runs)
  - UnimplementedProvider (factory placeholder for deepseek/gemini/local_teacher)
    -- added in Task 3
  - create_provider factory -- added in Task 3

No SDK imports. No real API calls. Real provider implementations land
in Slice 2 (input generation) and Slice 3 (output labeling).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal provider surface. All methods must be thread-safe."""
    name: str
    provider_type: str

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        """Return up to n raw input strings, one per example."""
        ...

    def generate_label(self, input_text: str) -> str:
        """Return raw provider text. May be invalid JSON."""
        ...


class ProviderError(RuntimeError):
    """Raised for provider-level failures (bad fixture, API down, etc.).
    Distinct from validator/JSON failures."""


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
      1. If row has 'raw_output' (must be a string): return verbatim.
      2. Else if row has 'output': return json.dumps(row['output'], separators=(',', ':')).
      3. Else: ProviderError at load.

    If input_text is not found in labels: return miss_payload (default "").
    """

    provider_type = "fake"

    def __init__(
        self,
        name: str,
        *,
        inputs_path: Path | None = None,
        labels_path: Path | None = None,
        seed: int = 0,
        miss_payload: str = "",
    ) -> None:
        self.name = name
        self._miss_payload = miss_payload
        self._seed = seed  # reserved; Slice 1 is cursor-deterministic.

        self._inputs: list[str] = []
        if inputs_path is not None:
            self._inputs = self._load_inputs(Path(inputs_path))

        self._labels: dict[str, str] = {}
        if labels_path is not None:
            self._labels = self._load_labels(Path(labels_path))

        self._cursor = 0
        self._cursor_lock = threading.Lock()

    @staticmethod
    def _load_inputs(path: Path) -> list[str]:
        rows: list[str] = []
        with path.open(encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ProviderError(
                        f"{path}: line {line_no}: invalid JSON: {e}"
                    ) from None
                if not isinstance(obj, dict) or "input" not in obj or not isinstance(obj["input"], str):
                    raise ProviderError(
                        f"{path}: line {line_no}: missing required 'input' string field"
                    )
                rows.append(obj["input"])
        return rows

    @staticmethod
    def _load_labels(path: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        with path.open(encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ProviderError(
                        f"{path}: line {line_no}: invalid JSON: {e}"
                    ) from None
                if not isinstance(obj, dict) or "input" not in obj or not isinstance(obj["input"], str):
                    raise ProviderError(
                        f"{path}: line {line_no}: missing required 'input' string field"
                    )
                if "raw_output" in obj:
                    if not isinstance(obj["raw_output"], str):
                        raise ProviderError(
                            f"{path}: line {line_no}: 'raw_output' must be a string"
                        )
                    out[obj["input"]] = obj["raw_output"]
                elif "output" in obj:
                    out[obj["input"]] = json.dumps(obj["output"], separators=(",", ":"))
                else:
                    raise ProviderError(
                        f"{path}: line {line_no}: row needs 'output' or 'raw_output'"
                    )
        return out

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        if n < 0:
            raise ProviderError(f"generate_inputs: n must be >= 0, got {n}")
        if n == 0 or not self._inputs:
            return []
        with self._cursor_lock:
            result: list[str] = []
            for _ in range(n):
                result.append(self._inputs[self._cursor % len(self._inputs)])
                self._cursor += 1
            return result

    def generate_label(self, input_text: str) -> str:
        # _labels is immutable after __init__ — no lock needed.
        return self._labels.get(input_text, self._miss_payload)


class UnimplementedProvider:
    """Placeholder for deepseek / gemini / local_teacher in Slice 1.

    Constructs without any SDK import so configs parse and dry-runs work.
    Any generation call raises NotImplementedError pointing at the slice
    where the real implementation lands.
    """

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


def create_provider(cfg) -> LLMProvider:
    """Map ProviderConfig.provider_type -> concrete provider.

    Only 'fake' is callable in Slice 1; other types return UnimplementedProvider
    so dry-run-quota and config parsing work with real-shaped configs.

    The loader (generation_config.py §4.5) stores resolved absolute paths in
    cfg.fixture_inputs / cfg.fixture_labels, so we wrap them in Path() directly.
    """
    if cfg.provider_type == "fake":
        return FakeProvider(
            name=cfg.name,
            inputs_path=Path(cfg.fixture_inputs) if cfg.fixture_inputs else None,
            labels_path=Path(cfg.fixture_labels) if cfg.fixture_labels else None,
            seed=cfg.seed if cfg.seed is not None else 0,
        )
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


def _parse_input_lines(text: str, *, expected: int) -> list[str]:
    """Provider output → cleaned, locally-deduped input strings.

    Local dedupe catches the common case of one API call returning the
    same line twice. Global dedupe (against existing inputs_raw.jsonl +
    this run's accepted set) happens in the orchestrator.
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


class DeepSeekProvider:
    """Real DeepSeek provider via the openai SDK (OpenAI-compatible endpoint).

    Lazy SDK import: `openai` is imported inside __init__, NOT at module
    load, so `import llm_providers` stays SDK-free for tests/dry-run.
    """

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
        from _retry import retry_with_backoff
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


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """Defensive against SDK variation: some versions expose status as int,
    others as string. Coerce to int; on failure, do not retry."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        status_int = int(status)
    except (TypeError, ValueError):
        return False
    return status_int in {429, 500, 502, 503, 504}


class GeminiProvider:
    """Real Gemini provider via the google-genai SDK.

    Lazy SDK import: `google.genai` is imported inside __init__, NOT at
    module load, so `import llm_providers` stays SDK-free.
    """

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

        from google import genai                                          # lazy
        from google.genai import errors as genai_errors

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ProviderError(
                f"GeminiProvider {name!r}: GOOGLE_API_KEY (or GEMINI_API_KEY) not set"
            )
        self._client = genai.Client(api_key=key)
        # Tuple is broad on purpose; the is_retryable predicate filters by status.
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
        from _retry import retry_with_backoff
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
