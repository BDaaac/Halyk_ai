import pytest
from pathlib import Path
from decimal import Decimal


def test_dotenv_populates_missing_values_without_overriding_environment():
    from config import load_dotenv

    env_file = Path(__file__).parent / "fixtures" / "dotenv.example"
    environment = {"ANTHROPIC_API_KEY": "from-environment"}

    load_dotenv(env_file, environment)

    assert environment == {
        "ANTHROPIC_API_KEY": "from-environment",
        "EXTRACT_MODEL": "test-model",
    }


def test_client_retries_transient_service_failure_once():
    from lib.anthropic_client import AnthropicClient, HttpResponse

    calls = []
    pauses = []

    def transport(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        if len(calls) == 1:
            return HttpResponse(status=503, body=b'{"error":"unavailable"}')
        return HttpResponse(status=200, body=b'{"id":"msg_1","content":[],"usage":{"input_tokens":1,"output_tokens":1}}')

    client = AnthropicClient(
        api_key="test-key",
        model="test-model",
        timeout_seconds=60,
        transport=transport,
        sleeper=pauses.append,
    )

    response = client.create_message(messages=[{"role": "user", "content": "."}], max_tokens=1)

    assert response["id"] == "msg_1"
    assert len(calls) == 2
    assert calls[0][3] == 60
    assert pauses == [1.0]


def test_client_does_not_retry_logical_api_error():
    from lib.anthropic_client import AnthropicApiError, AnthropicClient, HttpResponse

    calls = []

    def transport(*_):
        calls.append(1)
        return HttpResponse(status=401, body=b'{"error":"invalid key"}')

    client = AnthropicClient(api_key="bad-key", model="test-model", transport=transport, sleeper=lambda _: None)

    with pytest.raises(AnthropicApiError, match="401"):
        client.create_message(messages=[{"role": "user", "content": "."}], max_tokens=1)

    assert calls == [1]


def test_structured_message_returns_forced_tool_input():
    from lib.anthropic_client import AnthropicClient, HttpResponse

    captured = {}

    def transport(_url, payload, _headers, _timeout):
        captured.update(payload)
        return HttpResponse(
            status=200,
            body=(
                b'{"content":[{"type":"tool_use","name":"emit_result",'
                b'"input":{"covenants":[]}}],"usage":{"input_tokens":3,"output_tokens":2}}'
            ),
        )

    client = AnthropicClient(api_key="test-key", model="test-model", transport=transport)
    result = client.create_structured_message(
        system="extract",
        user="documents",
        tool_name="emit_result",
        input_schema={"type": "object", "properties": {"covenants": {"type": "array"}}},
    )

    assert result.output == {"covenants": []}
    assert result.usage == {"input_tokens": 3, "output_tokens": 2}
    assert captured["tool_choice"] == {"type": "tool", "name": "emit_result"}
    assert captured["tools"][0]["input_schema"]["type"] == "object"


def test_cost_estimate_uses_stage_pricing_from_environment(monkeypatch):
    from config import get_settings
    from lib.costs import estimate_cost

    monkeypatch.setenv("EXTRACT_INPUT_USD_PER_MTOK", "1")
    monkeypatch.setenv("EXTRACT_OUTPUT_USD_PER_MTOK", "5")

    settings = get_settings()

    assert estimate_cost({"input_tokens": 10_000, "output_tokens": 2_000}, settings.extract_pricing) == Decimal("0.020000")


def test_client_records_usage_before_structured_response_validation():
    from lib.anthropic_client import AnthropicClient, HttpResponse

    client = AnthropicClient(
        api_key="test-key",
        model="test-model",
        transport=lambda *_: HttpResponse(status=200, body=b'{"content":[],"usage":{"input_tokens":9,"output_tokens":4}}'),
    )

    with pytest.raises(ValueError, match="did not contain exactly one"):
        client.create_structured_message(system="x", user="y", tool_name="emit", input_schema={"type": "object"})

    assert client.usage_history == [{"input_tokens": 9, "output_tokens": 4}]
