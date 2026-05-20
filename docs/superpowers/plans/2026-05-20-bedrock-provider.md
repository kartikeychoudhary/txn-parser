# Bedrock Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `BedrockProvider` to `scripts/llm_providers.py` that lets any AWS Bedrock model (Claude/Llama/Mistral/Cohere/etc.) be used for input generation and output labeling, with tool-use structured output, optional extended thinking, and optional prompt caching.

**Architecture:** New synchronous `BedrockProvider` class parallel to `DeepSeekProvider`/`GeminiProvider`. Uses boto3 `bedrock-runtime` Converse API; auth via the `AWS_BEARER_TOKEN_BEDROCK` env var only. In-flight concurrency only — the orchestrator's existing thread pool drives many concurrent Converse calls. No orchestrator changes.

**Tech Stack:** Python ≥ 3.10, `boto3 >= 1.35` (lazy-imported), existing `_retry.retry_with_backoff`, existing `_lib.SYSTEM_PROMPT` and `_lib.build_messages` / `_parse_input_lines`. Tests use stubbed `boto3.client` patched via `monkeypatch`.

**Spec reference:** `docs/superpowers/specs/2026-05-20-bedrock-provider-design.md`.

---

## File Map

**Modify**
- `requirements.txt` — add `boto3>=1.35`.
- `scripts/generation_config.py` — add `"bedrock"` provider type, three new `ProviderConfig` fields (`region`, `thinking_budget_tokens`, `cache_system_prompt`), parse + validate them.
- `scripts/llm_providers.py` — add `_label_schema_for_bedrock()` helper, `_is_retryable_bedrock_error` predicate, `BedrockProvider` class, and wire it into `create_provider()`.
- `configs/prices.json` — add Bedrock model pricing entries.
- `tests/conftest.py` — add a `_StubBedrockClient` and extend `patch_sdk_clients` to patch `boto3.client` for `bedrock-runtime`.
- `tests/test_generation_config.py` — add bedrock-specific parse/validate cases.

**Create**
- `tests/test_bedrock_provider.py` — full unit suite for `BedrockProvider`.
- `configs/example_bedrock_providers.json` — runnable example wiring Bedrock into input + output generation.

---

## Task 1: Add boto3 to requirements

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Read the file to find the right insertion point**

Run: open `requirements.txt`. boto3 is alphabetical-ish but the file already mixes provider SDKs (`openai`, `google-genai`). Add the line near the other client SDKs to keep them grouped.

- [ ] **Step 2: Add boto3 line**

Add a single line:

```
boto3>=1.35
```

- [ ] **Step 3: Verify the file parses with pip**

Run: `pip install --dry-run -r requirements.txt`
Expected: no errors; existing pins unchanged.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add boto3>=1.35 for Bedrock provider"
```

---

## Task 2: Add config fields and validation for the `bedrock` provider type

**Files:**
- Modify: `scripts/generation_config.py`
- Modify: `tests/test_generation_config.py`

- [ ] **Step 1: Write failing test — bedrock provider type parses with required fields**

Append to `tests/test_generation_config.py`:

```python
def test_bedrock_provider_parses(tmp_path):
    data = _minimal_dict()
    data["output_generation"]["providers"].append({
        "name": "br",
        "type": "bedrock",
        "weight": 1,
        "threads": 1,
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "region": "us-east-1",
        "temperature": 0.7,
        "max_tokens": 1000,
        "max_retries": 3,
        "structured_output": True,
        "thinking_budget_tokens": 2000,
        "cache_system_prompt": True,
    })
    path = _write_config(tmp_path, data)
    cfg = load_generation_config(path)
    p = cfg.output_generation.providers[1]
    assert p.provider_type == "bedrock"
    assert p.region == "us-east-1"
    assert p.thinking_budget_tokens == 2000
    assert p.cache_system_prompt is True


def test_bedrock_provider_requires_model(tmp_path):
    data = _minimal_dict()
    data["output_generation"]["providers"].append({
        "name": "br",
        "type": "bedrock",
        "weight": 1,
        "threads": 1,
        # model intentionally missing
        "region": "us-east-1",
    })
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "bedrock" in str(exc.value) and "model" in str(exc.value)


def test_bedrock_defaults_when_optional_fields_absent(tmp_path):
    data = _minimal_dict()
    data["output_generation"]["providers"].append({
        "name": "br",
        "type": "bedrock",
        "weight": 1,
        "threads": 1,
        "model": "anthropic.claude-3-5-haiku-20241022-v1:0",
    })
    path = _write_config(tmp_path, data)
    cfg = load_generation_config(path)
    p = cfg.output_generation.providers[1]
    assert p.region is None
    assert p.thinking_budget_tokens is None
    assert p.cache_system_prompt is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generation_config.py -k bedrock -v`
Expected: FAIL — `"bedrock" not in _VALID_PROVIDER_TYPES`.

- [ ] **Step 3: Extend `_VALID_PROVIDER_TYPES` and `_KNOWN_PROVIDER_FIELDS`**

In `scripts/generation_config.py`, near the top of the file:

```python
_VALID_PROVIDER_TYPES = {"fake", "deepseek", "gemini", "local_teacher", "bedrock"}
_KNOWN_PROVIDER_FIELDS = {
    "name", "type", "weight", "threads", "model", "temperature", "max_tokens",
    "max_retries", "structured_output", "seed", "fixture_inputs", "fixture_labels",
    "region", "thinking_budget_tokens", "cache_system_prompt",
}
```

- [ ] **Step 4: Extend `ProviderConfig` with the three new fields**

In `scripts/generation_config.py`, in the `ProviderConfig` dataclass (currently ends with `fixture_labels: str | None = None`), append:

```python
    region: str | None = None
    thinking_budget_tokens: int | None = None
    cache_system_prompt: bool = False
```

- [ ] **Step 5: Parse the new fields in `_parse_providers`**

In `scripts/generation_config.py`, inside `_parse_providers`, at the `out.append(ProviderConfig(...))` call, add the three new keyword args at the end (before the closing `))`):

```python
            region=_optional_str(item.get("region"), f"{prefix}.region"),
            thinking_budget_tokens=_optional_int(
                item.get("thinking_budget_tokens"),
                f"{prefix}.thinking_budget_tokens",
                min_value=1,
            ),
            cache_system_prompt=_require_bool(
                item.get("cache_system_prompt", False),
                f"{prefix}.cache_system_prompt",
            ),
```

- [ ] **Step 6: Extend `_validate_providers` to require `model` for bedrock**

In `scripts/generation_config.py`, find the existing check:

```python
        if p.provider_type in {"deepseek", "gemini"} and not p.model:
            raise ConfigError(f"{prefix}: {p.provider_type} provider requires model")
```

Replace `{"deepseek", "gemini"}` with `{"deepseek", "gemini", "bedrock"}`:

```python
        if p.provider_type in {"deepseek", "gemini", "bedrock"} and not p.model:
            raise ConfigError(f"{prefix}: {p.provider_type} provider requires model")
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_generation_config.py -v`
Expected: PASS for all tests (new bedrock tests AND all previously-passing tests).

- [ ] **Step 8: Commit**

```bash
git add scripts/generation_config.py tests/test_generation_config.py
git commit -m "config: add bedrock provider type with region/thinking/cache fields"
```

---

## Task 3: Add `_StubBedrockClient` fixture to conftest

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add the stub class and extend `patch_sdk_clients`**

In `tests/conftest.py`, after the `_StubGeminiClient` class:

```python
class _StubBedrockRuntimeClient:
    """Stand-in for boto3.client('bedrock-runtime', ...). Captures the last
    converse(**kwargs) call and returns a configurable response. Tests
    set `.next_response` to control what converse() returns and inspect
    `.last_kwargs` to verify the request shape."""

    def __init__(self, **client_kwargs):
        self.client_kwargs = client_kwargs
        self.last_kwargs: dict | None = None
        self.next_response: dict = {
            "output": {"message": {"role": "assistant", "content": [{"text": ""}]}},
            "usage": {"inputTokens": 0, "outputTokens": 0},
            "stopReason": "end_turn",
        }
        self.raise_on_call: BaseException | None = None

    def converse(self, **kwargs):
        self.last_kwargs = kwargs
        if self.raise_on_call is not None:
            exc = self.raise_on_call
            # one-shot by default; tests opt into multi-shot via raise_n
            self.raise_on_call = getattr(self, "_next_raise", None)
            raise exc
        return self.next_response
```

Then update `patch_sdk_clients` to also patch boto3 — replace the existing fixture body with:

```python
@_pytest.fixture
def patch_sdk_clients(monkeypatch):
    """Patch openai.OpenAI, google.genai.Client, and boto3.client so
    provider __init__ never makes network calls. Opt-in: tests reference
    `patch_sdk_clients` by parameter name when they want SDK construction
    stubbed.
    """
    import openai
    monkeypatch.setattr(openai, "OpenAI", _StubOpenAIClient)
    from google import genai
    monkeypatch.setattr(genai, "Client", _StubGeminiClient)

    # boto3 may not be installed in every test environment; only patch if
    # importable. Tests that need it should also be marked or import-guarded.
    try:
        import boto3
    except ImportError:
        return

    _stub_holder = {"client": None}

    def _fake_client(service_name, **kwargs):
        # Only Bedrock runtime is stubbed here; raise loudly for any other
        # service so tests catch surprise S3/STS calls.
        if service_name != "bedrock-runtime":
            raise AssertionError(
                f"Unexpected boto3.client({service_name!r}) in test"
            )
        c = _StubBedrockRuntimeClient(**kwargs)
        _stub_holder["client"] = c
        return c

    monkeypatch.setattr(boto3, "client", _fake_client)

    # Expose the latest-constructed stub via a fixture attribute so tests
    # can reach it: `request.getfixturevalue('patch_sdk_clients').latest()`.
    # Simpler: also expose it by mutating the fixture return — but pytest
    # fixtures don't return mutable state by default, so a tiny accessor
    # via monkeypatch suffices. Tests use llm_providers.BedrockProvider's
    # `._client` attribute directly after construction.
    return _stub_holder
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `pytest tests/test_real_providers.py -v`
Expected: PASS — `patch_sdk_clients` still patches openai + genai, and the new boto3 patching is no-op when boto3 isn't actually consumed.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: stub boto3.client('bedrock-runtime') in patch_sdk_clients"
```

---

## Task 4: Add `_label_schema_for_bedrock` and `_is_retryable_bedrock_error` helpers

**Files:**
- Modify: `scripts/llm_providers.py`

- [ ] **Step 1: Add the schema helper**

In `scripts/llm_providers.py`, immediately after `_label_schema_for_gemini()`:

```python
def _label_schema_for_bedrock() -> dict:
    """Bedrock Converse expects a standard JSON Schema in `tool.inputSchema.json`.
    Lowercase type names (unlike Gemini's uppercase variant)."""
    from _lib import CATEGORIES, TYPES, CURRENCIES
    return {
        "type": "object",
        "properties": {
            "transactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "amount":   {"type": "number"},
                        "currency": {"type": "string", "enum": list(CURRENCIES)},
                        "item":     {"type": "string"},
                        "category": {"type": "string", "enum": list(CATEGORIES)},
                        "type":     {"type": "string", "enum": list(TYPES)},
                    },
                    "required": ["amount", "currency", "item", "category", "type"],
                },
            },
        },
        "required": ["transactions"],
    }
```

- [ ] **Step 2: Add the retry predicate**

Immediately after `_label_schema_for_bedrock()`:

```python
_BEDROCK_RETRYABLE_CODES = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "ModelErrorException",
    "InternalServerException",
})


def _is_retryable_bedrock_error(exc: BaseException) -> bool:
    """ClientError carries an error code in .response['Error']['Code'];
    retry only the transient ones. ReadTimeoutError and EndpointConnectionError
    have no code — always retry those (the predicate only runs for the
    exception types in the retryable tuple)."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code is None:
            return True
        return code in _BEDROCK_RETRYABLE_CODES
    # Non-ClientError botocore exceptions: ReadTimeoutError, EndpointConnectionError.
    return True
```

- [ ] **Step 3: No tests yet — these are helpers; they get exercised by Task 5**

(Skip ahead to Task 5; no commit yet — keep helpers and `BedrockProvider` in one commit.)

---

## Task 5: Implement `BedrockProvider` — construction and auth/region

**Files:**
- Modify: `scripts/llm_providers.py`
- Create: `tests/test_bedrock_provider.py`

- [ ] **Step 1: Write failing construction tests**

Create `tests/test_bedrock_provider.py`:

```python
"""Tests for BedrockProvider — construction, generate_inputs, generate_label,
retry, usage stashing. boto3.client('bedrock-runtime') is patched by the
shared `patch_sdk_clients` fixture so no real AWS calls happen.
"""
import json

import pytest

from llm_providers import (
    BedrockProvider,
    ProviderError,
    _is_retryable_bedrock_error,
)


# ---- Construction ---------------------------------------------------------

def test_bedrock_constructs_with_env_token_and_region(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fake-token")
    p = BedrockProvider(
        name="br",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region="us-east-1",
    )
    assert p.name == "br"
    assert p.provider_type == "bedrock"
    assert p.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    # Stub client should have been constructed with the right region.
    assert p._client.client_kwargs["region_name"] == "us-east-1"


def test_bedrock_falls_back_to_aws_region_env(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fake-token")
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    p = BedrockProvider(name="br", model="anthropic.claude-3-5-haiku-20241022-v1:0")
    assert p._client.client_kwargs["region_name"] == "eu-central-1"


def test_bedrock_missing_token_raises_provider_error(monkeypatch, patch_sdk_clients):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    with pytest.raises(ProviderError) as exc:
        BedrockProvider(name="br", model="anthropic.claude-3-5-haiku-20241022-v1:0")
    assert "AWS_BEARER_TOKEN_BEDROCK" in str(exc.value)


def test_bedrock_missing_region_raises_provider_error(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fake-token")
    monkeypatch.delenv("AWS_REGION", raising=False)
    with pytest.raises(ProviderError) as exc:
        BedrockProvider(name="br", model="anthropic.claude-3-5-haiku-20241022-v1:0")
    assert "region" in str(exc.value).lower()


def test_bedrock_does_not_use_aws_access_key_for_auth(monkeypatch, patch_sdk_clients):
    """Provider must use the bearer token only, not the IAM credential chain."""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "should-not-be-used")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-be-used")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    with pytest.raises(ProviderError):
        BedrockProvider(name="br", model="anthropic.claude-3-5-haiku-20241022-v1:0")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bedrock_provider.py -v`
Expected: FAIL — `BedrockProvider` not importable from `llm_providers`.

- [ ] **Step 3: Implement `BedrockProvider.__init__`**

In `scripts/llm_providers.py`, after `GeminiProvider` (end of file), add:

```python
class BedrockProvider:
    """Real AWS Bedrock provider via the boto3 `bedrock-runtime` Converse API.

    Auth: `AWS_BEARER_TOKEN_BEDROCK` env var only (Bedrock long-term API key).
    No IAM credential chain, no profile, no key in config files.

    Region resolution: cfg.region -> AWS_REGION env var -> ProviderError.
    No silent default; wrong region routes traffic and bills wrong account.

    Lazy SDK import: `boto3` is imported inside __init__, NOT at module load.
    """

    provider_type = "bedrock"

    def __init__(
        self,
        name: str,
        *,
        model: str,
        region: str | None = None,
        temperature: float | None = 1.0,
        max_tokens: int | None = 4000,
        max_retries: int = 3,
        structured_output: bool = False,
        thinking_budget_tokens: int | None = None,
        cache_system_prompt: bool = False,
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.structured_output = structured_output
        self.thinking_budget_tokens = thinking_budget_tokens
        self.cache_system_prompt = cache_system_prompt
        self._sleep = time.sleep
        self._tls = threading.local()

        # Bearer-token auth: env var only. Bedrock's runtime client picks
        # AWS_BEARER_TOKEN_BEDROCK up automatically; we just gate construction
        # so missing config fails loudly here instead of inside .converse().
        if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            raise ProviderError(
                f"BedrockProvider {name!r}: AWS_BEARER_TOKEN_BEDROCK not set"
            )

        resolved_region = region or os.environ.get("AWS_REGION")
        if not resolved_region:
            raise ProviderError(
                f"BedrockProvider {name!r}: no region (set provider config "
                f"`region` or AWS_REGION env var)"
            )
        self.region = resolved_region

        import boto3                                                       # lazy
        from botocore.exceptions import (
            ClientError, ReadTimeoutError, EndpointConnectionError,
        )

        self._client = boto3.client("bedrock-runtime", region_name=resolved_region)
        self._retryable_excs = (ClientError, ReadTimeoutError, EndpointConnectionError)

    def pop_last_usage(self) -> dict | None:
        u = getattr(self._tls, "usage", None)
        self._tls.usage = None
        return u

    def _stash_usage(
        self,
        prompt_tokens=None,
        completion_tokens=None,
        cache_read_tokens=None,
    ) -> None:
        if prompt_tokens is None and completion_tokens is None:
            self._tls.usage = None
        else:
            self._tls.usage = {
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "cache_read_tokens": int(cache_read_tokens or 0),
            }
```

- [ ] **Step 4: Run construction tests**

Run: `pytest tests/test_bedrock_provider.py -v`
Expected: PASS for the five construction tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/llm_providers.py tests/test_bedrock_provider.py
git commit -m "providers: add BedrockProvider construction (bearer-token auth)"
```

---

## Task 6: Implement `generate_label` (text path)

**Files:**
- Modify: `scripts/llm_providers.py`
- Modify: `tests/test_bedrock_provider.py`

- [ ] **Step 1: Write failing tests for `generate_label` text path**

Append to `tests/test_bedrock_provider.py`:

```python
# ---- generate_label: text path ------------------------------------------

def test_bedrock_generate_label_returns_text_block(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(name="br", model="m", region="us-east-1")
    p._client.next_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": '{"transactions":[]}'}],
            }
        },
        "usage": {"inputTokens": 11, "outputTokens": 7},
        "stopReason": "end_turn",
    }
    raw = p.generate_label("500 beer")
    assert raw == '{"transactions":[]}'

    # Request shape: system prompt sent, messages contain the input, no toolConfig.
    kw = p._client.last_kwargs
    assert kw["modelId"] == "m"
    assert kw["messages"] == [{"role": "user", "content": [{"text": "500 beer"}]}]
    assert kw["system"][0]["text"]  # SYSTEM_PROMPT is non-empty
    assert "toolConfig" not in kw
    assert "additionalModelRequestFields" not in kw


def test_bedrock_generate_label_stashes_usage(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(name="br", model="m", region="us-east-1")
    p._client.next_response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "x"}]}},
        "usage": {"inputTokens": 13, "outputTokens": 5},
        "stopReason": "end_turn",
    }
    p.generate_label("any")
    u = p.pop_last_usage()
    assert u == {"prompt_tokens": 13, "completion_tokens": 5, "cache_read_tokens": 0}
    # second pop returns None
    assert p.pop_last_usage() is None


def test_bedrock_generate_label_empty_content_returns_empty_string(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(name="br", model="m", region="us-east-1")
    p._client.next_response = {
        "output": {"message": {"role": "assistant", "content": []}},
        "usage": {"inputTokens": 1, "outputTokens": 0},
        "stopReason": "end_turn",
    }
    assert p.generate_label("x") == ""
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_bedrock_provider.py::test_bedrock_generate_label_returns_text_block -v`
Expected: FAIL — `generate_label` not implemented.

- [ ] **Step 3: Implement `generate_label` and `_extract_label_text`**

In `scripts/llm_providers.py`, inside the `BedrockProvider` class, add:

```python
    def generate_label(self, input_text: str) -> str:
        """One Converse call to label this input. Returns raw provider text
        (or JSON-serialized tool-use input when structured_output=True)."""
        return self._call_with_retry_label(input_text)

    def _call_with_retry_label(self, input_text: str) -> str:
        from _retry import retry_with_backoff
        return retry_with_backoff(
            lambda: self._call_api_label(input_text),
            retryable=self._retryable_excs,
            is_retryable=_is_retryable_bedrock_error,
            max_retries=self.max_retries,
            logger_name=f"bedrock.{self.name}",
            sleep=self._sleep,
        )

    def _build_converse_kwargs(
        self,
        *,
        system_text: str | None,
        user_text: str,
        use_tool: bool,
    ) -> dict:
        inference_config: dict = {}
        if self.max_tokens is not None:
            inference_config["maxTokens"] = self.max_tokens
        if self.temperature is not None:
            inference_config["temperature"] = self.temperature

        kwargs: dict = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": user_text}]}],
        }

        if system_text is not None:
            system_blocks: list = [{"text": system_text}]
            if self.cache_system_prompt:
                system_blocks.append({"cachePoint": {"type": "default"}})
            kwargs["system"] = system_blocks

        if use_tool and self.structured_output:
            kwargs["toolConfig"] = {
                "tools": [{
                    "toolSpec": {
                        "name": "emit_label",
                        "description": "Emit the structured transaction label.",
                        "inputSchema": {"json": _label_schema_for_bedrock()},
                    }
                }],
                "toolChoice": {"tool": {"name": "emit_label"}},
            }

        if self.thinking_budget_tokens:
            kwargs["additionalModelRequestFields"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget_tokens,
                }
            }
            # Anthropic requires temperature=1.0 when thinking is enabled.
            inference_config["temperature"] = 1.0

        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        return kwargs

    def _call_api_label(self, input_text: str) -> str:
        from _lib import SYSTEM_PROMPT
        kwargs = self._build_converse_kwargs(
            system_text=SYSTEM_PROMPT,
            user_text=input_text,
            use_tool=True,
        )
        resp = self._client.converse(**kwargs)
        self._stash_from_response(resp)
        return self._extract_label_text(resp)

    @staticmethod
    def _extract_label_text(resp: dict) -> str:
        """Walk content blocks; prefer tool-use input (JSON-serialized) over
        text; skip reasoning blocks. Returns '' on empty content."""
        message = resp.get("output", {}).get("message", {})
        blocks = message.get("content", [])
        for block in blocks:
            if "reasoningContent" in block:
                continue
            if "toolUse" in block:
                tool_input = block["toolUse"].get("input", {})
                return json.dumps(tool_input, separators=(",", ":"))
        # No tool block; return first text block.
        for block in blocks:
            if "reasoningContent" in block:
                continue
            if "text" in block:
                return block["text"]
        return ""

    def _stash_from_response(self, resp: dict) -> None:
        usage = resp.get("usage") or {}
        self._stash_usage(
            prompt_tokens=usage.get("inputTokens"),
            completion_tokens=usage.get("outputTokens"),
            cache_read_tokens=usage.get("cacheReadInputTokens"),
        )
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/test_bedrock_provider.py -v`
Expected: PASS for construction tests + the three new generate_label text-path tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/llm_providers.py tests/test_bedrock_provider.py
git commit -m "providers: BedrockProvider.generate_label text path + usage stash"
```

---

## Task 7: `generate_label` — tool-use, thinking, and cache paths

**Files:**
- Modify: `tests/test_bedrock_provider.py`

- [ ] **Step 1: Write failing tests for the three modal paths**

Append to `tests/test_bedrock_provider.py`:

```python
# ---- generate_label: tool-use path ---------------------------------------

def test_bedrock_generate_label_tool_use_returns_json(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(
        name="br", model="m", region="us-east-1", structured_output=True,
    )
    p._client.next_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{
                    "toolUse": {
                        "toolUseId": "id1",
                        "name": "emit_label",
                        "input": {"transactions": [
                            {"amount": 500, "currency": "INR", "item": "beer",
                             "category": "food", "type": "expense"},
                        ]},
                    }
                }],
            }
        },
        "usage": {"inputTokens": 20, "outputTokens": 15},
        "stopReason": "tool_use",
    }
    raw = p.generate_label("500 beer")
    parsed = json.loads(raw)
    assert parsed == {"transactions": [
        {"amount": 500, "currency": "INR", "item": "beer",
         "category": "food", "type": "expense"},
    ]}

    # Request must include toolConfig with forced choice.
    kw = p._client.last_kwargs
    assert kw["toolConfig"]["toolChoice"] == {"tool": {"name": "emit_label"}}
    assert kw["toolConfig"]["tools"][0]["toolSpec"]["name"] == "emit_label"


# ---- generate_label: thinking path ---------------------------------------

def test_bedrock_thinking_sets_request_fields_and_skips_reasoning(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(
        name="br", model="m", region="us-east-1",
        thinking_budget_tokens=2000, temperature=0.2,
    )
    p._client.next_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
                    {"text": "final"},
                ],
            }
        },
        "usage": {"inputTokens": 4, "outputTokens": 6},
        "stopReason": "end_turn",
    }
    out = p.generate_label("any")
    assert out == "final"

    kw = p._client.last_kwargs
    assert kw["additionalModelRequestFields"]["thinking"] == {
        "type": "enabled", "budget_tokens": 2000,
    }
    # Temperature must be forced to 1.0 even though we set 0.2 on the provider.
    assert kw["inferenceConfig"]["temperature"] == 1.0


# ---- generate_label: cache path -----------------------------------------

def test_bedrock_cache_system_prompt_adds_cache_block(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(
        name="br", model="m", region="us-east-1", cache_system_prompt=True,
    )
    p._client.next_response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 2, "cacheReadInputTokens": 8},
        "stopReason": "end_turn",
    }
    p.generate_label("anything")

    kw = p._client.last_kwargs
    # Second system block is the cache point.
    assert kw["system"][0]["text"]  # the SYSTEM_PROMPT
    assert kw["system"][1] == {"cachePoint": {"type": "default"}}

    u = p.pop_last_usage()
    assert u["cache_read_tokens"] == 8


def test_bedrock_cache_disabled_omits_cache_block(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(
        name="br", model="m", region="us-east-1", cache_system_prompt=False,
    )
    p._client.next_response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "usage": {"inputTokens": 1, "outputTokens": 1},
        "stopReason": "end_turn",
    }
    p.generate_label("x")
    kw = p._client.last_kwargs
    # Only one block — the system text, no cache point.
    assert len(kw["system"]) == 1
```

- [ ] **Step 2: Run tests — verify they pass**

Run: `pytest tests/test_bedrock_provider.py -v`
Expected: PASS — Task 6's implementation already covers tool-use, thinking, and cache. If any fail, fix the implementation in `scripts/llm_providers.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_bedrock_provider.py
git commit -m "test: BedrockProvider tool-use, thinking, and cache paths"
```

---

## Task 8: `generate_inputs`

**Files:**
- Modify: `scripts/llm_providers.py`
- Modify: `tests/test_bedrock_provider.py`

- [ ] **Step 1: Write failing tests for `generate_inputs`**

Append to `tests/test_bedrock_provider.py`:

```python
# ---- generate_inputs -----------------------------------------------------

def test_bedrock_generate_inputs_parses_lines(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(name="br", model="m", region="us-east-1")
    p._client.next_response = {
        "output": {"message": {"role": "assistant", "content": [
            {"text": "500 beer\n200 chai\n100 samosa"},
        ]}},
        "usage": {"inputTokens": 5, "outputTokens": 10},
        "stopReason": "end_turn",
    }
    assert p.generate_inputs("ignored prompt", n=3) == ["500 beer", "200 chai", "100 samosa"]

    # No system block (input phase doesn't use SYSTEM_PROMPT); no toolConfig.
    kw = p._client.last_kwargs
    assert "system" not in kw
    assert "toolConfig" not in kw


def test_bedrock_generate_inputs_truncates_to_n(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(name="br", model="m", region="us-east-1")
    p._client.next_response = {
        "output": {"message": {"role": "assistant", "content": [
            {"text": "a x\nb y\nc z\nd q\ne r"},
        ]}},
        "usage": {"inputTokens": 1, "outputTokens": 5},
        "stopReason": "end_turn",
    }
    assert len(p.generate_inputs("ignored", n=3)) == 3


def test_bedrock_generate_inputs_zero_returns_empty(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(name="br", model="m", region="us-east-1")
    # Make a call fail loudly if invoked.
    def _no_call(**_kw):
        raise AssertionError("should not call converse for n=0")
    p._client.converse = _no_call
    assert p.generate_inputs("ignored", n=0) == []


def test_bedrock_generate_inputs_negative_n_raises(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(name="br", model="m", region="us-east-1")
    with pytest.raises(ProviderError):
        p.generate_inputs("ignored", n=-1)
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_bedrock_provider.py::test_bedrock_generate_inputs_parses_lines -v`
Expected: FAIL — `generate_inputs` not implemented.

- [ ] **Step 3: Implement `generate_inputs`**

In `scripts/llm_providers.py`, inside `BedrockProvider`, add:

```python
    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        if n < 0:
            raise ProviderError(f"generate_inputs: n must be >= 0, got {n}")
        if n == 0:
            return []
        text = self._call_with_retry(prompt)
        return _parse_input_lines(text, expected=n)

    def _call_with_retry(self, prompt: str) -> str:
        from _retry import retry_with_backoff
        return retry_with_backoff(
            lambda: self._call_api(prompt),
            retryable=self._retryable_excs,
            is_retryable=_is_retryable_bedrock_error,
            max_retries=self.max_retries,
            logger_name=f"bedrock.{self.name}",
            sleep=self._sleep,
        )

    def _call_api(self, prompt: str) -> str:
        kwargs = self._build_converse_kwargs(
            system_text=None,        # input phase: no SYSTEM_PROMPT
            user_text=prompt,
            use_tool=False,          # input phase: never use tool
        )
        resp = self._client.converse(**kwargs)
        self._stash_from_response(resp)
        # Always return first text block; no tool in input phase.
        message = resp.get("output", {}).get("message", {})
        for block in message.get("content", []):
            if "text" in block:
                return block["text"]
        return ""
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/test_bedrock_provider.py -v`
Expected: PASS for all tests so far.

- [ ] **Step 5: Commit**

```bash
git add scripts/llm_providers.py tests/test_bedrock_provider.py
git commit -m "providers: BedrockProvider.generate_inputs"
```

---

## Task 9: Retry behavior tests

**Files:**
- Modify: `tests/test_bedrock_provider.py`

- [ ] **Step 1: Write retry tests using a fake ClientError**

Append to `tests/test_bedrock_provider.py`:

```python
# ---- Retry behavior ------------------------------------------------------

class _FakeClientError(Exception):
    """Mimics botocore.exceptions.ClientError minimal surface: carries a
    .response dict with Error.Code."""
    def __init__(self, code: str):
        super().__init__(f"fake-{code}")
        self.response = {"Error": {"Code": code, "Message": code}}


def test_is_retryable_bedrock_error_recognizes_throttling():
    assert _is_retryable_bedrock_error(_FakeClientError("ThrottlingException"))
    assert _is_retryable_bedrock_error(_FakeClientError("ServiceUnavailableException"))
    assert _is_retryable_bedrock_error(_FakeClientError("InternalServerException"))


def test_is_retryable_bedrock_error_rejects_validation_error():
    assert not _is_retryable_bedrock_error(_FakeClientError("ValidationException"))
    assert not _is_retryable_bedrock_error(_FakeClientError("AccessDeniedException"))


def test_bedrock_retries_on_throttling(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(name="br", model="m", region="us-east-1", max_retries=2)
    # Reassign retryable to include our fake so retry_with_backoff catches it.
    p._retryable_excs = (_FakeClientError,)
    p._sleep = lambda _d: None

    state = {"calls": 0}
    success_resp = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "usage": {"inputTokens": 1, "outputTokens": 1},
        "stopReason": "end_turn",
    }
    def flaky(**_kw):
        state["calls"] += 1
        if state["calls"] < 2:
            raise _FakeClientError("ThrottlingException")
        return success_resp
    p._client.converse = flaky
    assert p.generate_label("x") == "ok"
    assert state["calls"] == 2


def test_bedrock_does_not_retry_on_validation_error(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(name="br", model="m", region="us-east-1", max_retries=3)
    p._retryable_excs = (_FakeClientError,)
    p._sleep = lambda _d: None

    state = {"calls": 0}
    def boom(**_kw):
        state["calls"] += 1
        raise _FakeClientError("ValidationException")
    p._client.converse = boom

    with pytest.raises(_FakeClientError):
        p.generate_label("x")
    assert state["calls"] == 1   # no retries on non-retryable error
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_bedrock_provider.py -k "retry or is_retryable" -v`
Expected: PASS — `_is_retryable_bedrock_error` and the retry machinery work end-to-end.

- [ ] **Step 3: Commit**

```bash
git add tests/test_bedrock_provider.py
git commit -m "test: BedrockProvider retry behavior (Throttling retries, Validation does not)"
```

---

## Task 10: Wire `bedrock` into `create_provider`

**Files:**
- Modify: `scripts/llm_providers.py`
- Modify: `tests/test_bedrock_provider.py`

- [ ] **Step 1: Write failing factory test**

Append to `tests/test_bedrock_provider.py`:

```python
# ---- create_provider wiring ---------------------------------------------

def test_create_provider_returns_bedrock(monkeypatch, patch_sdk_clients):
    from llm_providers import create_provider
    from generation_config import ProviderConfig
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    cfg = ProviderConfig(
        name="br",
        provider_type="bedrock",
        model="anthropic.claude-3-5-haiku-20241022-v1:0",
        region="us-east-1",
        temperature=0.5,
        max_tokens=500,
        max_retries=2,
        structured_output=True,
        thinking_budget_tokens=1024,
        cache_system_prompt=True,
    )
    p = create_provider(cfg)
    assert isinstance(p, BedrockProvider)
    assert p.model == "anthropic.claude-3-5-haiku-20241022-v1:0"
    assert p.region == "us-east-1"
    assert p.structured_output is True
    assert p.thinking_budget_tokens == 1024
    assert p.cache_system_prompt is True
```

- [ ] **Step 2: Run — verify it fails**

Run: `pytest tests/test_bedrock_provider.py::test_create_provider_returns_bedrock -v`
Expected: FAIL — `create_provider` raises `ValueError("Unknown provider type 'bedrock'")`.

- [ ] **Step 3: Add the `bedrock` branch to `create_provider`**

In `scripts/llm_providers.py`, in `create_provider`, immediately before the final `raise ValueError(...)`, insert:

```python
    if cfg.provider_type == "bedrock":
        return BedrockProvider(
            name=cfg.name,
            model=cfg.model,
            region=cfg.region,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.max_retries,
            structured_output=cfg.structured_output,
            thinking_budget_tokens=cfg.thinking_budget_tokens,
            cache_system_prompt=cfg.cache_system_prompt,
        )
```

- [ ] **Step 4: Run — verify it passes**

Run: `pytest tests/test_bedrock_provider.py -v`
Expected: PASS for all tests in the file.

- [ ] **Step 5: Commit**

```bash
git add scripts/llm_providers.py tests/test_bedrock_provider.py
git commit -m "providers: wire bedrock into create_provider factory"
```

---

## Task 11: Pricing entries and example config

**Files:**
- Modify: `configs/prices.json`
- Create: `configs/example_bedrock_providers.json`

- [ ] **Step 1: Add Bedrock pricing**

Replace `configs/prices.json` with:

```json
{
  "deepseek:deepseek-chat": {"input_per_million": 0.27, "output_per_million": 1.10},
  "deepseek:deepseek-v4-flash": {"input_per_million": 0.27, "output_per_million": 1.10},
  "gemini:gemini-2.5-flash": {"input_per_million": 0.30, "output_per_million": 2.50},
  "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0": {"input_per_million": 3.00, "output_per_million": 15.00},
  "bedrock:anthropic.claude-3-5-haiku-20241022-v1:0":  {"input_per_million": 1.00, "output_per_million": 5.00},
  "bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0": {"input_per_million": 3.00, "output_per_million": 15.00},
  "local_teacher": {"input_per_million": 0.0, "output_per_million": 0.0},
  "fake": {"input_per_million": 0.0, "output_per_million": 0.0},
  "default": {"input_per_million": 0.0, "output_per_million": 0.0}
}
```

- [ ] **Step 2: Create example config**

Create `configs/example_bedrock_providers.json`:

```json
{
  "version": 1,
  "input_generation": {
    "enabled": true,
    "target_inputs": 100,
    "batch_size": 20,
    "dedupe": true,
    "providers": [
      {
        "name": "bedrock_haiku_inputs",
        "type": "bedrock",
        "weight": 100,
        "threads": 4,
        "model": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "region": "us-east-1",
        "temperature": 1.0,
        "max_tokens": 2000,
        "max_retries": 3
      }
    ]
  },
  "output_generation": {
    "enabled": true,
    "label_attempts_per_input": 1,
    "providers": [
      {
        "name": "bedrock_sonnet_labels",
        "type": "bedrock",
        "weight": 100,
        "threads": 8,
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "region": "us-east-1",
        "temperature": 0.2,
        "max_tokens": 4000,
        "max_retries": 3,
        "structured_output": true,
        "cache_system_prompt": true
      }
    ]
  },
  "validation": {
    "schema": true,
    "semantic_validator": true,
    "reject_invalid": true,
    "retry_invalid_with_stricter_prompt": false,
    "max_repair_attempts": 0
  },
  "rate_limits": {
    "global_max_workers": 8,
    "write_flush_every": 20
  }
}
```

- [ ] **Step 3: Verify the example parses**

Run: `python -c "import sys; sys.path.insert(0, 'scripts'); from generation_config import load_generation_config; print(load_generation_config('configs/example_bedrock_providers.json').output_generation.providers[0])"`
Expected: prints a `ProviderConfig(...)` line with `provider_type='bedrock'` and `cache_system_prompt=True`.

- [ ] **Step 4: Commit**

```bash
git add configs/prices.json configs/example_bedrock_providers.json
git commit -m "configs: Bedrock pricing entries and example provider config"
```

---

## Task 12: Full test sweep + smoke

**Files:** none modified

- [ ] **Step 1: Run the entire test suite**

Run: `pytest -x -q`
Expected: PASS — all existing tests plus the new `test_bedrock_provider.py` and `test_generation_config.py::*bedrock*` tests.

- [ ] **Step 2: Confirm SDK-free import still works**

Run: `python -c "import sys; sys.path.insert(0, 'scripts'); import llm_providers; print('OK', list(c for c in dir(llm_providers) if 'Provider' in c))"`
Expected: prints `OK ['BedrockProvider', 'DeepSeekProvider', 'FakeProvider', 'GeminiProvider', 'LLMProvider', 'LocalTeacherProvider', 'UnimplementedProvider']` (order may differ) — no boto3 import error even if boto3 not installed in the test env, because the import is lazy.

- [ ] **Step 3: Confirm `create_provider` raises clearly when the env var is missing**

Run:
```bash
python -c "
import os, sys
os.environ.pop('AWS_BEARER_TOKEN_BEDROCK', None)
sys.path.insert(0, 'scripts')
from llm_providers import create_provider
from generation_config import ProviderConfig
try:
    create_provider(ProviderConfig(name='x', provider_type='bedrock', model='anthropic.claude-3-5-haiku-20241022-v1:0', region='us-east-1'))
except Exception as e:
    print('GOT:', type(e).__name__, e)
"
```
Expected: prints `GOT: ProviderError BedrockProvider 'x': AWS_BEARER_TOKEN_BEDROCK not set`.

- [ ] **Step 4: No commit — verification only**

If anything failed in steps 1-3, return to the failing task and fix it before considering this plan complete.

---

## Notes for the implementer

- **`boto3` may not be installed** in test environments that don't exercise Bedrock. The `try/except ImportError` in `conftest.py` Task 3 covers that. `llm_providers.BedrockProvider.__init__` will raise `ImportError` at runtime when `boto3` is missing — that's the correct behavior (matches DeepSeek's failure mode when `openai` is missing).
- **`AWS_BEARER_TOKEN_BEDROCK` consumption by boto3**: this env var is the AWS-documented bearer-token mechanism for Bedrock long-term API keys. If your installed boto3 version doesn't pick it up automatically, the symptom will be a `botocore.exceptions.NoCredentialsError` at the first `converse()` call. Mitigation: upgrade boto3, or — only if needed — switch to explicit `Session(profile_name=...)` later. The provider construction code does not need to set anything other than `region_name`; boto3 reads the bearer token from the env on its own.
- **Cross-region inference profile IDs** (`us.anthropic.claude-...`, `eu.anthropic.claude-...`) work without code changes — pass them as the `model` field. Pricing entries above are for the base model IDs; if profile IDs are used in production, add matching `bedrock:us.anthropic.claude-...` entries to `configs/prices.json`.
- **Cache token cost accounting** is intentionally not changed in this plan. `cache_read_tokens` is exposed via `pop_last_usage()` for downstream use; the cost reporter still prices all `prompt_tokens` at the input rate. A follow-up plan can refine that when the cost reporter is touched next.

---
