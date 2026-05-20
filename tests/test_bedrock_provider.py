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
    assert len(kw["system"]) == 1


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
    def _no_call(**_kw):
        raise AssertionError("should not call converse for n=0")
    p._client.converse = _no_call
    assert p.generate_inputs("ignored", n=0) == []


def test_bedrock_generate_inputs_negative_n_raises(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    p = BedrockProvider(name="br", model="m", region="us-east-1")
    with pytest.raises(ProviderError):
        p.generate_inputs("ignored", n=-1)


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
    assert state["calls"] == 1


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
