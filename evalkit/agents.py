"""Load answer and grader settings from an agents YAML file."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError


class AgentSpec(BaseModel):
    """One model that either answers or grades."""

    provider: str
    model: str | None = None
    temperature: float | None = None
    system_prompt: str | None = None


class AgentsFile(BaseModel):
    """Settings for the answer model and the grader."""

    answer_agent: AgentSpec
    grader_agent: AgentSpec


class ResolvedAgents(BaseModel):
    """The settings a run will use after CLI flags override the file."""

    provider: str
    model: str | None = None
    temperature: float | None = None
    system_prompt: str | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_temperature: float = 0
    judge_system_prompt: str | None = None


def load_agents(path: Path) -> AgentsFile:
    """Read ``answer_agent`` and ``grader_agent`` from a YAML file."""
    if not path.is_file():
        raise ValueError(f"config not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("agents config must be a mapping")  # noqa: TRY004
    try:
        return AgentsFile.model_validate(loaded)
    except ValidationError as exc:
        raise ValueError(f"invalid agents config: {exc}") from exc


def resolve_agents(
    agents: AgentsFile | None,
    *,
    provider: str | None,
    model: str | None,
    temperature: float | None,
    judge_provider: str | None,
    judge_model: str | None,
) -> ResolvedAgents:
    """Use a CLI value when one was passed, and the file otherwise."""
    answer = None if agents is None else agents.answer_agent
    grader = None if agents is None else agents.grader_agent
    return ResolvedAgents(
        provider=provider or (answer.provider if answer is not None else "mock"),
        model=model if model is not None else (None if answer is None else answer.model),
        temperature=(
            temperature
            if temperature is not None
            else None
            if answer is None
            else answer.temperature
        ),
        system_prompt=None if answer is None else _text(answer.system_prompt),
        judge_provider=judge_provider or (None if grader is None else grader.provider),
        judge_model=(
            judge_model if judge_model is not None else None if grader is None else grader.model
        ),
        judge_temperature=(
            0 if grader is None or grader.temperature is None else grader.temperature
        ),
        judge_system_prompt=None if grader is None else _text(grader.system_prompt),
    )


def _text(value: str | None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
