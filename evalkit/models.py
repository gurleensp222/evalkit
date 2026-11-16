from typing import Any

from pydantic import BaseModel, Field


class Task(BaseModel):
    """One item in an evaluation dataset."""

    id: str
    category: str
    prompt: str
    scorer: str
    expected: Any


class ScoreResult(BaseModel):
    """Score for a single model output."""

    score: float = Field(ge=0, le=1)
    passed: bool
    reason: str


class TaskResult(BaseModel):
    """Recorded outcome of one task in a run."""

    run_id: str
    task_id: str
    category: str
    model: str
    output: str
    score: float = Field(ge=0, le=1)
    passed: bool
    reason: str
    latency_ms: float
    attempt: int = Field(default=1, ge=1)
    unstable: bool = False
    failure_type: str | None = None
