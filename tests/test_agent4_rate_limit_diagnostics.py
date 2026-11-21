from types import SimpleNamespace

from agents.legal_rag.agent import (
    LegalAnalysisAgent,
    llm_error_diagnostics,
)
from llm.config import LLMConfig


def test_diagnostics_capture_rate_limit_metadata():
    class FakeAPIError(Exception):
        status_code = 429
        request_id = "req-test-123"
        response = SimpleNamespace(
            headers={
                "retry-after": "2",
                "retry-after-ms": "2000",
                "x-ratelimit-reset-requests": "1s",
                "x-ratelimit-reset-tokens": "3s",
            }
        )

    result = llm_error_diagnostics(FakeAPIError("simulated failure"))

    assert result["error_type"] == "FakeAPIError"
    assert result["http_status"] == "429"
    assert result["retry_after"] == "2"
    assert result["retry_after_ms"] == "2000"
    assert result["rate_limit_reset_requests"] == "1s"
    assert result["rate_limit_reset_tokens"] == "3s"
    assert result["request_id"] == "req-test-123"


def test_diagnostics_do_not_include_error_body_or_message():
    class FakeAPIError(Exception):
        status_code = 429
        body = {"secret": "sensitive response"}
        response = SimpleNamespace(headers={})

    error = FakeAPIError("sensitive error message")
    result = llm_error_diagnostics(error)

    assert "body" not in result
    assert "secret" not in str(result)
    assert "sensitive" not in str(result)


def test_agent_uses_shared_retry_configuration(monkeypatch):
    config = LLMConfig(
        api_key="test-key",
        model="test-model",
        max_retries=2,
    )
    received = {}

    def fake_get_config():
        return config

    class FakeGroqClient:
        def __init__(self, config):
            received["config"] = config

    monkeypatch.setattr("llm.config.get_llm_config", fake_get_config)
    monkeypatch.setattr("llm.groq_client.GroqLLMClient", FakeGroqClient)

    LegalAnalysisAgent(retriever=object())

    assert received["config"] is config
    assert received["config"].max_retries == 2
