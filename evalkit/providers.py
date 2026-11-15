"""Model providers.

``generate(prompt)`` returns the completion text. ``task_id`` is optional and is
how ``MockProvider`` looks up a canned answer.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import httpx

MOCK_ANSWERS: dict[str, str] = {
    "exact-right": "4",
    "exact-wrong": "Lyon",
    "contains-right": "The primary colors are red, blue, and yellow.",
    "contains-wrong": "Mars and Jupiter are planets.",
    "json-right": '{"name": "Ada", "city": "London"}',
    "json-wrong": '{"a": 1}',
    "python-right": "def add(a, b):\n    return a + b\n",
    "python-wrong": "def add(a, b):\n    return a - b\n",
}

OLLAMA_URL = "http://localhost:11434"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_HTTP_TIMEOUT_SECONDS = 60.0


class ProviderError(RuntimeError):
    """The requested provider cannot be constructed."""


class Provider(Protocol):
    """A model that turns a prompt into text."""

    name: str

    def generate(self, prompt: str, *, task_id: str | None = None) -> str:
        """Return the completion for ``prompt``."""


class MockProvider:
    """Return canned answers from a dict keyed by task id."""

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        *,
        name: str = "mock",
    ) -> None:
        self.name = name
        source = MOCK_ANSWERS if answers is None else answers
        self.answers = dict(source)

    def generate(self, prompt: str, *, task_id: str | None = None) -> str:
        if task_id is not None and task_id in self.answers:
            return self.answers[task_id]
        if prompt in self.answers:
            return self.answers[prompt]
        return "I don't know"


class OllamaProvider:
    """Call a local Ollama server."""

    def __init__(
        self,
        *,
        model: str = "llama3.2",
        base_url: str = OLLAMA_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "ollama"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client

    def generate(self, prompt: str, *, task_id: str | None = None) -> str:
        del task_id
        body = _post(
            self._client,
            f"{self.base_url}/api/generate",
            payload={"model": self.model, "prompt": prompt, "stream": False},
        )
        return str(body.get("response", ""))


class OpenAIProvider:
    """Call the OpenAI chat completions API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ProviderError("OPENAI_API_KEY is not set")
        self.name = "openai"
        self.api_key = api_key
        self.model = model
        self._client = client

    def generate(self, prompt: str, *, task_id: str | None = None) -> str:
        del task_id
        body = _post(
            self._client,
            OPENAI_URL,
            payload={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        message = body["choices"][0]["message"]
        content = message.get("content") if isinstance(message, dict) else ""
        return "" if content is None else str(content)


class AnthropicProvider:
    """Call the Anthropic messages API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-3-5-haiku-latest",
        max_tokens: int = 1024,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self.name = "anthropic"
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    def generate(self, prompt: str, *, task_id: str | None = None) -> str:
        del task_id
        body = _post(
            self._client,
            ANTHROPIC_URL,
            payload={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        content = body["content"]
        texts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(texts)


def create_openai_provider(*, model: str | None = None) -> OpenAIProvider | None:
    """Build an OpenAI provider only when ``OPENAI_API_KEY`` is set."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    if model:
        return OpenAIProvider(api_key, model=model)
    return OpenAIProvider(api_key)


def create_anthropic_provider(*, model: str | None = None) -> AnthropicProvider | None:
    """Build an Anthropic provider only when ``ANTHROPIC_API_KEY`` is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    if model:
        return AnthropicProvider(api_key, model=model)
    return AnthropicProvider(api_key)


def create_provider(
    name: str,
    *,
    answers: dict[str, str] | None = None,
    model: str | None = None,
) -> Provider:
    """Return the named provider. Remote providers require their API key."""
    normalized = name.strip().lower()
    chosen = model.strip() if isinstance(model, str) else ""
    if normalized == "mock":
        return MockProvider(answers)
    if normalized == "ollama":
        if chosen:
            return OllamaProvider(model=chosen)
        return OllamaProvider()
    if normalized == "openai":
        provider = create_openai_provider(model=chosen or None)
        if provider is None:
            raise ProviderError("OPENAI_API_KEY is not set")
        return provider
    if normalized == "anthropic":
        provider = create_anthropic_provider(model=chosen or None)
        if provider is None:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        return provider
    raise ProviderError(f"unknown provider: {name}")


def _post(
    client: httpx.Client | None,
    url: str,
    *,
    payload: dict[str, object],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    if client is None:
        with httpx.Client(timeout=_HTTP_TIMEOUT_SECONDS) as owned:
            response = owned.post(url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
    else:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise ProviderError("provider returned a non-object JSON response")
    return body
