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
import threading
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
            seed=cfg.seed or 0,
        )
    if cfg.provider_type in {"deepseek", "gemini", "local_teacher"}:
        return UnimplementedProvider(
            name=cfg.name, provider_type=cfg.provider_type, model=cfg.model,
        )
    raise ValueError(
        f"Unknown provider type {cfg.provider_type!r} for provider {cfg.name!r}"
    )
