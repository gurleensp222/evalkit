"""Model providers.

``generate(prompt)`` returns the completion text. ``task_id`` is optional and is
how ``MockProvider`` looks up a canned answer.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import httpx

MOCK_ANSWERS: dict[str, str] = {
    "two-plus-two": "4",
    "capital-of-france": "Lyon",
    "primary-colors": "The primary colors are red, blue, and yellow.",
    "closest-planets": "Mars and Jupiter are planets.",
    "name-and-city": '{"name": "Ada", "city": "London"}',
    "keys-a-and-b": '{"a": 1}',
    "add-function": "def add(a, b):\n    return a + b\n",
    "add-function-arguments": "def add(a, b):\n    return a - b\n",
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"
OLLAMA_URL = DEFAULT_OLLAMA_URL
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_HTTP_TIMEOUT_SECONDS = 60.0
_TAGS_TIMEOUT_SECONDS = 10.0
_NUM_PREDICT = 512


class ProviderError(RuntimeError):
    """The requested provider cannot be constructed."""


class Provider(Protocol):
    """A model that turns a prompt into text."""

    name: str

    def generate(
        self,
        prompt: str,
        *,
        task_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
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

    def generate(
        self,
        prompt: str,
        *,
        task_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        del timeout
        if task_id is not None and task_id in self.answers:
            return self.answers[task_id]
        if prompt in self.answers:
            return self.answers[prompt]
        return "I don't know"


class OllamaProvider:
    """Call a local Ollama server through ``/api/chat``.

    ``name`` is the model id (for example ``llama3.2``) so saved rows do not say
    ``ollama``. The base URL comes from ``OLLAMA_URL`` when one is not passed.
    """

    uses_http_timeout = True
    default_timeout = 120.0
    default_workers = 1

    def __init__(
        self,
        *,
        model: str = "llama3.2",
        base_url: str | None = None,
        client: httpx.Client | None = None,
        system: str | None = None,
        temperature: float | None = None,
        json_format: bool = False,
    ) -> None:
        resolved = base_url if base_url is not None else os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.model = model.strip()
        self.name = self.model
        self.base_url = resolved.rstrip("/")
        self.system = system.strip() if isinstance(system, str) and system.strip() else None
        self.temperature = temperature
        self.json_format = json_format
        self._client = client

    def check_model_available(self) -> None:
        """Raise ``ProviderError`` when ``/api/tags`` does not list this model."""
        try:
            body = _get(
                self._client,
                f"{self.base_url}/api/tags",
                timeout=_TAGS_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise ProviderError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"could not reach Ollama at {self.base_url}: {exc}") from exc
        names = _tag_names(body.get("models", []))
        if not _model_listed(self.model, names):
            raise ProviderError(f"run: ollama pull {self.model}")

    def generate(
        self,
        prompt: str,
        *,
        task_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        del task_id
        payload: dict[str, object] = {
            "model": self.model,
            "messages": _chat_messages(prompt, self.system),
            "stream": False,
            "options": _chat_options(self.temperature),
        }
        if self.json_format:
            payload["format"] = "json"
        body = _post(
            self._client,
            f"{self.base_url}/api/chat",
            payload=payload,
            timeout=timeout,
        )
        message = body.get("message")
        if not isinstance(message, dict):
            raise ProviderError("provider returned a chat response without a message")
        content = message.get("content")
        return "" if content is None else str(content)


class OpenAIProvider:
    """Call the OpenAI chat completions API."""

    uses_http_timeout = True

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        client: httpx.Client | None = None,
        temperature: float | None = None,
        json_format: bool = False,
    ) -> None:
        if not api_key.strip():
            raise ProviderError("OPENAI_API_KEY is not set")
        self.name = "openai"
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.json_format = json_format
        self._client = client

    def generate(
        self,
        prompt: str,
        *,
        task_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        del task_id
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.json_format:
            payload["response_format"] = {"type": "json_object"}
        body = _post(
            self._client,
            OPENAI_URL,
            payload=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )
        message = body["choices"][0]["message"]
        content = message.get("content") if isinstance(message, dict) else ""
        return "" if content is None else str(content)


class AnthropicProvider:
    """Call the Anthropic messages API."""

    uses_http_timeout = True

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-3-5-haiku-latest",
        max_tokens: int = 1024,
        client: httpx.Client | None = None,
        temperature: float | None = None,
        json_format: bool = False,
    ) -> None:
        if not api_key.strip():
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self.name = "anthropic"
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_format = json_format
        self._client = client

    def generate(
        self,
        prompt: str,
        *,
        task_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        del task_id
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        body = _post(
            self._client,
            ANTHROPIC_URL,
            payload=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=timeout,
        )
        content = body["content"]
        texts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(texts)


def create_openai_provider(
    *,
    model: str | None = None,
    temperature: float | None = None,
    json_format: bool = False,
) -> OpenAIProvider | None:
    """Build an OpenAI provider only when ``OPENAI_API_KEY`` is set."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    if model:
        return OpenAIProvider(
            api_key,
            model=model,
            temperature=temperature,
            json_format=json_format,
        )
    return OpenAIProvider(api_key, temperature=temperature, json_format=json_format)


def create_anthropic_provider(
    *,
    model: str | None = None,
    temperature: float | None = None,
    json_format: bool = False,
) -> AnthropicProvider | None:
    """Build an Anthropic provider only when ``ANTHROPIC_API_KEY`` is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    if model:
        return AnthropicProvider(
            api_key,
            model=model,
            temperature=temperature,
            json_format=json_format,
        )
    return AnthropicProvider(api_key, temperature=temperature, json_format=json_format)


def create_judge_provider(name: str, *, model: str | None = None) -> Provider:
    """Build a judge. It samples at temperature 0 and asks for JSON."""
    return create_provider(name, model=model, temperature=0, json_format=True)


def create_provider(
    name: str,
    *,
    answers: dict[str, str] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    system: str | None = None,
    json_format: bool = False,
) -> Provider:
    """Return the named provider. Remote providers require their API key."""
    normalized = name.strip().lower()
    chosen = model.strip() if isinstance(model, str) else ""
    if normalized == "mock":
        return MockProvider(answers)
    if normalized == "ollama":
        return OllamaProvider(
            **_ollama_kwargs(
                model=chosen or None,
                temperature=temperature,
                system=system,
                json_format=json_format,
            )
        )
    if normalized == "openai":
        provider = create_openai_provider(
            model=chosen or None,
            temperature=temperature,
            json_format=json_format,
        )
        if provider is None:
            raise ProviderError("OPENAI_API_KEY is not set")
        return provider
    if normalized == "anthropic":
        provider = create_anthropic_provider(
            model=chosen or None,
            temperature=temperature,
            json_format=json_format,
        )
        if provider is None:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        return provider
    raise ProviderError(f"unknown provider: {name}")


def _ollama_kwargs(
    *,
    model: str | None,
    temperature: float | None,
    system: str | None,
    json_format: bool,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if model:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature
    if system:
        kwargs["system"] = system
    if json_format:
        kwargs["json_format"] = True
    return kwargs


def _chat_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _chat_options(temperature: float | None) -> dict[str, object]:
    options: dict[str, object] = {"num_predict": _NUM_PREDICT}
    if temperature is not None:
        options["temperature"] = temperature
    return options


def _tag_names(models: object) -> list[str]:
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
    return names


def _model_listed(wanted: str, names: list[str]) -> bool:
    for name in names:
        if name == wanted:
            return True
        base, separator, _tag = name.partition(":")
        if separator and base == wanted:
            return True
    return False


def _post(
    client: httpx.Client | None,
    url: str,
    *,
    payload: dict[str, object],
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    return _exchange(client, "POST", url, payload=payload, headers=headers, timeout=timeout)


def _get(
    client: httpx.Client | None,
    url: str,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    return _exchange(client, "GET", url, timeout=timeout)


def _exchange(
    client: httpx.Client | None,
    method: str,
    url: str,
    *,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    limit = _HTTP_TIMEOUT_SECONDS if timeout is None else timeout
    try:
        if client is None:
            with httpx.Client(timeout=limit) as owned:
                response = _send(owned, method, url, payload=payload, headers=headers, timeout=None)
                response.raise_for_status()
                body = response.json()
        else:
            response = _send(client, method, url, payload=payload, headers=headers, timeout=limit)
            response.raise_for_status()
            body = response.json()
    except httpx.TimeoutException as exc:
        raise TimeoutError(f"timed out after {limit:g} seconds") from exc
    if not isinstance(body, dict):
        raise ProviderError("provider returned a non-object JSON response")
    return body


def _send(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    payload: dict[str, object] | None,
    headers: dict[str, str] | None,
    timeout: float | None,
) -> httpx.Response:
    kwargs: dict[str, object] = {}
    if headers is not None:
        kwargs["headers"] = headers
    if timeout is not None:
        kwargs["timeout"] = timeout
    if method == "GET":
        return client.get(url, **kwargs)
    kwargs["json"] = payload
    return client.post(url, **kwargs)
