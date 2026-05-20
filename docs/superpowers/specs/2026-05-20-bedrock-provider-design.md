# Bedrock Provider — Design

**Status:** draft
**Date:** 2026-05-20
**Owner:** Kartikey Choudhary
**Branch:** feat/grammar-decoding

## Goal

Add `BedrockProvider` to the multi-provider abstraction in `scripts/llm_providers.py` so any model hosted on AWS Bedrock (Claude family, Llama, Mistral, Cohere, etc.) can be used for input generation and output labeling alongside the existing DeepSeek, Gemini, and local-teacher providers.

The provider must match the existing provider contract (synchronous `generate_inputs` / `generate_label`, thread-safe, lazy SDK import, thread-local usage stash) so it integrates with the current orchestrator and scheduler without orchestrator changes.

"Batching" in this spec means **in-flight concurrency** — many concurrent synchronous Converse calls driven by the orchestrator's existing thread pool. Bedrock's offline `CreateModelInvocationJob` (S3-in / S3-out) is **out of scope**.

## Non-goals

- Async batch inference via `CreateModelInvocationJob`.
- `ConverseStream` (token streaming). `generate_label` returns a final string.
- Guardrails, knowledge bases, agents.
- Refactoring DeepSeek/Gemini onto a shared base class.

## Architecture

One new class, `BedrockProvider`, in `scripts/llm_providers.py`, parallel in shape to `DeepSeekProvider` and `GeminiProvider`:

- Lazy `boto3` import inside `__init__` — keeps `import llm_providers` SDK-free for tests/dry-runs.
- `self._client = boto3.client("bedrock-runtime", region_name=...)` — one client per provider instance; boto3 clients are thread-safe for `converse` calls.
- Thread-local usage stash (`self._tls`) with `pop_last_usage()` returning `{"prompt_tokens", "completion_tokens", "cache_read_tokens"}`.
- Retry via `_retry.retry_with_backoff` with a Bedrock-specific `is_retryable` predicate.

### Config schema changes (`scripts/generation_config.py`)

- Add `"bedrock"` to `_VALID_PROVIDER_TYPES`.
- Add the following fields to `_KNOWN_PROVIDER_FIELDS` and to `ProviderConfig`:
  - `region: str | None` — AWS region (e.g. `"us-east-1"`). Optional in config; falls back to `AWS_REGION` env var at construction time.
  - `thinking_budget_tokens: int | None` — when set, enables Claude extended thinking with this budget. Optional.
  - `cache_system_prompt: bool` — when true, adds a Converse `cachePoint` block after the system prompt. Default `false`.
- Cross-field validation: `bedrock` requires `model`. Region is **not** validated at config-parse time — it's checked at provider construction so the existing pattern (DeepSeek/Gemini check env vars at construction) is preserved.

### Dependency

Add `boto3>=1.35` to `requirements.txt`. Lazy-imported; existing tests and dry-runs that don't touch Bedrock continue to work without `boto3` installed (matching the DeepSeek/Gemini approach with `openai`/`google-genai`).

### Pricing (`configs/prices.json`)

Per-model entries keyed `bedrock:<modelId>`. Initial entries to add (extend as needed):

```json
"bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0": {"input_per_million": 3.00, "output_per_million": 15.00},
"bedrock:anthropic.claude-3-5-haiku-20241022-v1:0":  {"input_per_million": 1.00, "output_per_million": 5.00},
"bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0": {"input_per_million": 3.00, "output_per_million": 15.00}
```

Cache reads (when `cache_system_prompt: true`) come back as `cache_read_tokens`; the cost reporter should price them at 10% of input rate per Anthropic's published cache pricing on Bedrock.

## Auth

**Bearer token only** — Bedrock long-term API keys. No IAM credential chain, no profile, no access-key/secret-key fields in the config.

- Read `AWS_BEARER_TOKEN_BEDROCK` (AWS's standard env var name for Bedrock API keys) at `__init__`.
- If absent → `ProviderError("BedrockProvider {name!r}: AWS_BEARER_TOKEN_BEDROCK not set")`.
- The env var is consumed by `boto3.client("bedrock-runtime", ...)` automatically — no manual signing.
- We deliberately do **not** write the key into provider config files or accept it via a config field.

## Region

Resolution order at construction:
1. `cfg.region` (provider config field) if set.
2. `AWS_REGION` env var.
3. Hard fail with `ProviderError` — no silent default.

## Converse call shape

### Label generation (`generate_label(input_text)`)

```python
system_blocks = [{"text": SYSTEM_PROMPT}]
if self.cache_system_prompt:
    system_blocks.append({"cachePoint": {"type": "default"}})

kwargs = {
    "modelId": self.model,
    "system": system_blocks,
    "messages": [{"role": "user", "content": [{"text": input_text}]}],
    "inferenceConfig": {
        "maxTokens": self.max_tokens,           # omit key if None
        "temperature": self.temperature,        # omit key if None; forced to 1.0 when thinking enabled
    },
}

if self.structured_output:
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
        "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget_tokens}
    }
    kwargs["inferenceConfig"]["temperature"] = 1.0   # required when thinking enabled

response = self._client.converse(**kwargs)
```

`_label_schema_for_bedrock()` mirrors `_label_schema_for_gemini()` but uses lowercase JSON-Schema types (`"object"`, `"array"`, `"number"`, `"string"`) — Bedrock's Converse tool schema is standard JSON Schema, unlike Gemini's uppercase variant.

### Response extraction

Walk `response["output"]["message"]["content"]` (a list of blocks):

- Skip blocks with key `"reasoningContent"` (extended-thinking blocks).
- If a block has `"toolUse"`: return `json.dumps(block["toolUse"]["input"], separators=(",", ":"))`. First match wins.
- Otherwise return the first block's `"text"` field.
- Empty content → return `""`.

This keeps `generate_label` returning a plain string, matching every other provider's contract; the orchestrator's validator does its own JSON parsing.

### Input generation (`generate_inputs(prompt, n)`)

Same Converse call but:
- No `system` blocks (matches DeepSeek/Gemini behavior).
- No `toolConfig`.
- `prompt` goes in the single user message.
- Response is run through the existing `_parse_input_lines(text, expected=n)` helper.
- `n == 0` short-circuits to `[]` without making an API call; `n < 0` raises `ProviderError` (matches existing providers).

## Retry, errors, usage

### Retryable errors

`is_retryable` predicate returns `True` for:

- `botocore.exceptions.ClientError` where `exc.response["Error"]["Code"]` in
  `{"ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException", "ModelErrorException", "InternalServerException"}`.
- `botocore.exceptions.ReadTimeoutError`.
- `botocore.exceptions.EndpointConnectionError`.

Non-retryable (let them propagate to the orchestrator's per-input error handling): `ValidationException`, `AccessDeniedException`, `ResourceNotFoundException`, all other `ClientError` codes.

`_retryable_excs` passed to `retry_with_backoff` is the broad tuple `(ClientError, ReadTimeoutError, EndpointConnectionError)`; the `is_retryable` predicate narrows it.

### Usage stashing

After each successful call:

```python
usage = response.get("usage", {})
self._stash_usage(
    prompt_tokens=usage.get("inputTokens"),
    completion_tokens=usage.get("outputTokens"),
    cache_read_tokens=usage.get("cacheReadInputTokens"),
)
```

`_stash_usage` is updated to accept the third field. `pop_last_usage()` returns the dict with the third key included (`0` when not present). DeepSeek/Gemini stashes are unaffected — their `_stash_usage` signature keeps the existing two args and the new helper takes a default of `0` for cache.

To avoid signature drift across providers, refactor `_stash_usage` into a small free helper or accept a kwargs-only third field with a default — the implementation plan will choose the smallest-diff approach.

## `create_provider` wiring

Add a `bedrock` branch in `create_provider()`:

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

## Tests

New file: `tests/test_bedrock_provider.py`. Patch `boto3.client` so no real network I/O occurs (parallel to how `OpenAI` and `genai.Client` are patched today in `tests/conftest.py`).

Cases:

1. **Construction**
   - Succeeds with `AWS_BEARER_TOKEN_BEDROCK` set and either `region=` or `AWS_REGION`.
   - Raises `ProviderError` when bearer token env var missing.
   - Raises `ProviderError` when no region available from config or env.
2. **`generate_inputs`**
   - Parses multi-line response into list.
   - Truncates to `n`.
   - `n == 0` returns `[]` without calling the API (assert the API stub is not invoked).
   - Negative `n` raises `ProviderError`.
3. **`generate_label` — text path**
   - Returns the text block when structured_output=False and no tool use.
4. **`generate_label` — tool-use path**
   - With `structured_output=True`, response contains a `toolUse` block; provider returns `json.dumps(...)` of the tool input.
5. **`generate_label` — thinking path**
   - With `thinking_budget_tokens=2000`, request kwargs include `additionalModelRequestFields.thinking` and `inferenceConfig.temperature == 1.0`.
   - Response with a leading `reasoningContent` block is skipped; the trailing text/tool block is returned.
6. **`generate_label` — cache path**
   - With `cache_system_prompt=True`, request kwargs include a `cachePoint` block after the system text.
   - `cacheReadInputTokens` in the response is exposed via `pop_last_usage()["cache_read_tokens"]`.
7. **Retry**
   - Transient `ClientError("ThrottlingException")` retries up to `max_retries` then returns success.
   - Non-retryable `ClientError("ValidationException")` propagates after the first failure (no retries).
8. **Usage stashing**
   - All three usage fields populated correctly from a Converse response.

Config-parsing test additions in `tests/test_generation_config.py`:

- `bedrock` without `model` fails at parse time.
- `bedrock` with all valid fields parses successfully.
- Unknown bedrock-only fields emit the standard "unknown field" warning (matches existing behavior for unknown provider fields).

## Example config

To be added at `configs/example_providers.json` (or as a new doc snippet — implementation plan will decide):

```json
{
  "name": "bedrock_claude_sonnet",
  "type": "bedrock",
  "weight": 100,
  "threads": 8,
  "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
  "region": "us-east-1",
  "temperature": 0.7,
  "max_tokens": 4000,
  "max_retries": 3,
  "structured_output": true,
  "cache_system_prompt": true,
  "thinking_budget_tokens": null
}
```

Cross-region inference profile IDs (`us.anthropic.claude-...`, `eu.anthropic.claude-...`) are accepted unchanged — Bedrock routes them transparently and no provider-side code is needed.

## Risks & open questions

- **boto3 version**: Converse and `cachePoint` blocks require boto3 ≥ 1.35.x. Pinning `>=1.35` should cover all currently shipped Claude models on Bedrock; the implementation plan should verify against the installed version.
- **`AWS_BEARER_TOKEN_BEDROCK` env var name**: spec assumes this is the var boto3 consumes. The implementation plan must verify against the boto3 docs for the target version and adjust the variable name if AWS uses a different one in the version we pin.
- **Pricing entries**: rates above are current as of writing; the cost reporter only uses `prices.json`, so updating the file is the only maintenance step needed when AWS changes prices.
- **Cache hit accounting**: cache read tokens are billed at a fraction of the input rate; the cost reporter currently multiplies `prompt_tokens` by the input rate. A follow-up may need to subtract `cache_read_tokens` from `prompt_tokens` and price them separately — flagged but out of scope for this spec.
