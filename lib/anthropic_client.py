"""Small, dependency-free client for the Anthropic Messages API."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.anthropic.com/v1/messages"
RETRYABLE_STATUSES = {429, 503}


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


@dataclass(frozen=True)
class StructuredResult:
    output: dict[str, Any]
    usage: dict[str, int]


class AnthropicApiError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"Anthropic API error {status}: {detail}")
        self.status = status


def _default_transport(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> HttpResponse:
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return HttpResponse(status=response.status, body=response.read())
    except HTTPError as error:
        return HttpResponse(status=error.code, body=error.read())
    except URLError as error:
        raise AnthropicApiError(0, str(error.reason)) from error


class AnthropicClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: int = 60,
        transport: Callable[[str, dict[str, Any], dict[str, str], int], HttpResponse] = _default_transport,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured")
        if not model:
            raise ValueError("model is not configured")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.sleeper = sleeper
        self.usage_history: list[dict[str, int]] = []

    def create_message(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, str] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            payload["system"] = system
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        response = self._post(payload)
        self.usage_history.append(self._usage(response))
        return response

    def create_structured_message(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        input_schema: dict[str, Any],
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> StructuredResult:
        response = self.create_message(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            tools=[{"name": tool_name, "description": "Return the requested structured result.", "input_schema": input_schema}],
            tool_choice={"type": "tool", "name": tool_name},
            temperature=temperature,
        )
        tool_uses = [block for block in response.get("content", []) if block.get("type") == "tool_use" and block.get("name") == tool_name]
        if len(tool_uses) != 1 or not isinstance(tool_uses[0].get("input"), dict):
            raise ValueError(f"Anthropic response did not contain exactly one {tool_name} tool result")
        return StructuredResult(output=tool_uses[0]["input"], usage=self.usage_history[-1])

    def health_check(self) -> dict[str, int]:
        response = self.create_message(messages=[{"role": "user", "content": "."}], max_tokens=1)
        return self.usage_history[-1]

    @staticmethod
    def _usage(response: dict[str, Any]) -> dict[str, int]:
        usage = response.get("usage", {})
        return {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        for attempt in range(2):
            response = self.transport(API_URL, payload, headers, self.timeout_seconds)
            if 200 <= response.status < 300:
                try:
                    return json.loads(response.body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError("Anthropic API returned malformed JSON") from error
            detail = response.body.decode("utf-8", errors="replace")
            retryable = response.status in RETRYABLE_STATUSES or 500 <= response.status < 600
            if retryable and attempt == 0:
                self.sleeper(2.0**attempt)
                continue
            raise AnthropicApiError(response.status, detail)
        raise AssertionError("unreachable")
