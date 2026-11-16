from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from evalkit.rubrics import Criterion


class Task(BaseModel):
    """One item in an evaluation dataset.

    Set ``scorer`` for a built-in check, or ``rubric_id`` for a judge rubric.
    ``extra_criteria`` are added to that rubric. ``deterministic_checks`` run
    before the judge. ``allowed_urls`` is the list ``no_fabricated_links`` allows.
    """

    id: str
    category: str
    prompt: str
    scorer: str | None = None
    rubric_id: str | None = None
    expected: Any = None
    extra_criteria: list[Criterion] = Field(default_factory=list)
    allowed_urls: list[str] = Field(default_factory=list)
    deterministic_checks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def exactly_one_grader(self) -> Task:
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
        if self.extra_criteria:
            from evalkit.rubrics import Criterion

            parsed: list[Any] = []
            for item in self.extra_criteria:
                if isinstance(item, Criterion):
                    parsed.append(item)
                else:
                    parsed.append(Criterion.model_validate(item))
            self.extra_criteria = parsed
        if self.extra_criteria and not has_rubric:
            raise ValueError("extra_criteria require rubric_id")
        if self.deterministic_checks and not has_rubric:
            raise ValueError("deterministic_checks require rubric_id")
        unknown = [name for name in self.deterministic_checks if name != "no_fabricated_links"]
        if unknown:
            raise ValueError("unknown deterministic check: " + ", ".join(unknown))
        return self


class CriterionMark(BaseModel):
    """Whether one rubric criterion was met."""

    criterion_id: str
    met: bool
    reason: str


class ScoreResult(BaseModel):
    """Score for a single model output."""

    score: float = Field(ge=0, le=1)
    passed: bool
    reason: str
    forced_failure: str | None = None
    criteria: list[CriterionMark] = Field(default_factory=list)


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
    judge: str | None = None


def _bind_extra_criteria() -> None:
    """Resolve ``Criterion`` once both models can be imported."""
    from evalkit.rubrics import Criterion

    Task.model_rebuild(_types_namespace={"Criterion": Criterion})


_bind_extra_criteria()
