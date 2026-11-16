from typing import Any

from pydantic import BaseModel, Field, model_validator


class Task(BaseModel):
    """One item in an evaluation dataset.

    Set ``scorer`` for a built-in check, or ``rubric_id`` for a judge rubric.
    """

    id: str
    category: str
    prompt: str
    scorer: str | None = None
    rubric_id: str | None = None
    expected: Any = None

    @model_validator(mode="after")
    def exactly_one_grader(self) -> "Task":
        has_scorer = isinstance(self.scorer, str) and bool(self.scorer.strip())
        has_rubric = isinstance(self.rubric_id, str) and bool(self.rubric_id.strip())
        if has_scorer == has_rubric:
            raise ValueError("set exactly one of scorer or rubric_id")
        if has_scorer:
            self.scorer = self.scorer.strip()
            self.rubric_id = None
        else:
            assert self.rubric_id is not None
            self.rubric_id = self.rubric_id.strip()
            self.scorer = None
        return self


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
    output_varies: bool = False
    failure_type: str | None = None
    rubric_id: str | None = None
    rubric_version: int | None = None
    rubric_hash: str | None = None
