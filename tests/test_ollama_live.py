"""Live Ollama checks. CI skips these with ``-m 'not ollama'``."""

from __future__ import annotations

import os

import pytest

from evalkit.providers import OllamaProvider

pytestmark = pytest.mark.ollama


def test_live_ollama_lists_the_model_and_answers() -> None:
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    provider = OllamaProvider(model=model, temperature=0)
    provider.check_model_available()
    text = provider.generate("Reply with the single word pong.", timeout=120)
    assert text.strip()
