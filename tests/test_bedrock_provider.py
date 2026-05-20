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
