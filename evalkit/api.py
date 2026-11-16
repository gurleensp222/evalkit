"""HTTP API for datasets, runs, and comparisons.

Start it with ``uvicorn evalkit.api:app``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from evalkit.analysis import by_category, compare, failures, load_runs, summary
from evalkit.pipeline import answer_run, grade_run
from evalkit.providers import ProviderError, create_judge_provider, create_provider
from evalkit.rubrics import Rubric, load_prompts, load_rubrics
from evalkit.runner import ensure_model_available
from evalkit.storage import (
    DEFAULT_RUNS_DIR,
    StoredRun,
    list_run_ids,
    load_run,
    run_progress,
    upsert_run,
)

DEFAULT_DATASET = Path("datasets/sample_tasks.jsonl")
DEFAULT_RUBRICS = Path("datasets/rubrics")
RunStatusName = Literal[
    "answering",
    "answered",
    "grading",
    "graded",
    "interrupted",
    "failed",
]


class HealthResponse(BaseModel):
    """Liveness check."""

    status: Literal["ok"] = "ok"


class TaskOut(BaseModel):
    """One dataset row."""

    id: str
    category: str
    prompt: str
    scorer: str | None = None
    rubric_id: str | None = None
    expected: Any = Field(default=None, description="Scorer-specific expected value.")


class RunCreate(BaseModel):
    """Body for POST /runs."""

    provider: str = Field(examples=["mock"], description="mock, ollama, openai, or anthropic.")
    dataset: str = Field(
        examples=["datasets/sample_tasks.jsonl"],
        description="Path to a JSONL task file.",
    )
    model: str | None = Field(
        default=None,
        description="Model id. Ollama defaults to llama3.2.",
    )
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature for Ollama.",
    )
    workers: int | None = Field(
        default=None,
        ge=1,
        description="How many tasks to run at once. Ollama uses 1 unless this is set.",
    )
    judge_provider: str | None = Field(
        default=None,
        description="Provider that grades rubric tasks. Uses temperature 0 and JSON output.",
    )
    judge_model: str | None = Field(
        default=None,
        description="Model id for the judge.",
    )
    attempts: int = Field(
        default=3,
        ge=1,
        description="How many times to run each task.",
    )


class GradeCreate(BaseModel):
    """Body for POST /runs/{run_id}/grade."""

    judge_provider: str | None = Field(
        default=None,
        description="Provider that grades rubric tasks. Uses temperature 0 and JSON output.",
    )
    judge_model: str | None = Field(default=None, description="Model id for the judge.")


class RunCreated(BaseModel):
    """Accepted work. It continues in the background."""

    run_id: str
    status: Literal["answering", "grading"]


class RunInfo(BaseModel):
    """Status of one run."""

    run_id: str
    status: RunStatusName
    provider: str | None = None
    dataset: str | None = None
    error: str | None = None


class ModelSummary(BaseModel):
    """summary() for one model."""

    model: str
    pass_rate: float = Field(description="Share of tasks that passed.")
    mean_score: float
    mean_latency_ms: float
    p95_latency_ms: float


class RunDetail(BaseModel):
    """Summary, progress counts, and per-category pass rates."""

    run_id: str
    status: RunStatusName
    provider: str | None = None
    error: str | None = None
    answers_done: int = 0
    answers_expected: int = 0
    grades_done: int = 0
    grades_expected: int = 0
    summary: list[ModelSummary]
    by_category: dict[str, dict[str, float | None]] = Field(
        description="Pass rate by category (row) and model (column). Missing cells are null.",
    )


class FailureOut(BaseModel):
    """One failed task from failures()."""

    category: str
    task_id: str
    model: str
    run_id: str
    score: float
    reason: str
    failure_type: str | None = None
    attempt: int = 1


class CompareResponse(BaseModel):
    """compare() for two runs.

    Each run is one series. ``difference`` is run ``a``'s pass rate minus run ``b``'s.
    """

    model_a: str = Field(description="Run id passed as a.")
    model_b: str = Field(description="Run id passed as b.")
    n_tasks: int
    pass_rate_a: float
    pass_rate_b: float
    difference: float
    ci_low: float
    ci_high: float
    excludes_zero: bool = Field(description="True when the 95% interval does not contain 0.")


@dataclass
class _RunState:
    run_id: str
    status: RunStatusName
    provider: str
    dataset: str
    error: str | None = None


def create_app(
    *,
    runs_directory: Path = DEFAULT_RUNS_DIR,
    default_dataset: Path = DEFAULT_DATASET,
    rubrics_directory: Path = DEFAULT_RUBRICS,
) -> FastAPI:
    """Build the API. Tests pass their own runs directory and dataset."""
    app = FastAPI(title="evalkit", version="0.1.0")
    app.state.runs_directory = runs_directory
    app.state.default_dataset = default_dataset
    app.state.rubrics_directory = rubrics_directory
    app.state.runs = {}
    app.state.lock = Lock()
    app.include_router(router)
    return app


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the process is up."""
    return HealthResponse()


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks(
    request: Request,
    dataset: Annotated[
        str | None,
        Query(description="JSONL dataset. Defaults to the server dataset."),
    ] = None,
) -> list[TaskOut]:
    """List tasks in the dataset."""
    path = _dataset_path(request, dataset)
    try:
        tasks = load_prompts(path, _rubric_catalog(request))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return [TaskOut.model_validate(task.model_dump()) for task in tasks]


@router.post("/runs", response_model=RunCreated, status_code=202)
def start_run(body: RunCreate, request: Request, background: BackgroundTasks) -> RunCreated:
    """Start an evaluation in the background and return its run id."""
    dataset = Path(body.dataset)
    if not dataset.exists():
        raise HTTPException(status_code=404, detail=f"dataset not found: {dataset}")
    try:
        catalog = _rubric_catalog(request)
        load_prompts(dataset, catalog)
        ensure_model_available(
            create_provider(body.provider, model=body.model, temperature=body.temperature)
        )
        if body.judge_provider:
            ensure_model_available(
                create_judge_provider(body.judge_provider, model=body.judge_model)
            )
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    run_id = uuid4().hex
    provider_name = body.provider.strip().lower()
    upsert_run(
        run_id,
        directory=Path(request.app.state.runs_directory),
        status="answering",
        provider=provider_name,
        dataset=str(dataset),
        attempts=body.attempts,
        model=body.model,
        temperature=body.temperature,
    )
    _put_state(
        request.app,
        _RunState(
            run_id=run_id,
            status="answering",
            provider=provider_name,
            dataset=str(dataset),
        ),
    )
    background.add_task(
        _execute_run,
        request.app,
        run_id,
        body.provider,
        dataset,
        model=body.model,
        temperature=body.temperature,
        workers=body.workers,
        judge_provider=body.judge_provider,
        judge_model=body.judge_model,
        attempts=body.attempts,
    )
    return RunCreated(run_id=run_id, status="answering")


@router.post(
    "/runs/{run_id}/grade",
    response_model=RunCreated,
    status_code=202,
    responses={404: {"description": "Unknown run"}, 409: {"description": "Run is still in progress"}},
)
def start_grade(
    run_id: str,
    body: GradeCreate,
    request: Request,
    background: BackgroundTasks,
) -> RunCreated:
    """Grade saved answers in the background. Another judge is stored beside the first."""
    directory = Path(request.app.state.runs_directory)
    stored = _stored_or_http(run_id, directory)
    if stored.status in {"answering", "grading"}:
        raise HTTPException(status_code=409, detail=f"run {run_id} is still {stored.status}")
    if body.judge_provider:
        try:
            ensure_model_available(
                create_judge_provider(body.judge_provider, model=body.judge_model)
            )
        except ProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
    upsert_run(run_id, directory=directory, status="grading", error=None)
    background.add_task(
        _execute_grade,
        request.app,
        run_id,
        body.judge_provider,
        body.judge_model,
    )
    return RunCreated(run_id=run_id, status="grading")


@router.get("/runs", response_model=list[RunInfo])
def list_runs(request: Request) -> list[RunInfo]:
    """List known run ids and the status stored for each one."""
    return [_info(state) for state in _all_states(request.app)]


@router.get("/runs/{run_id}", response_model=RunDetail, responses={404: {"description": "Unknown run"}})
def get_run(run_id: str, request: Request) -> RunDetail:
    """Return status, progress counts, summary(), and by_category()."""
    directory = Path(request.app.state.runs_directory)
    stored = _stored_or_http(run_id, directory)
    progress = run_progress(run_id, directory=directory)
    frame = _frame_for(request.app, run_id, progress.grades_done)
    status = _status_name(stored.status)
    return RunDetail(
        run_id=run_id,
        status=status,
        provider=stored.provider or stored.model,
        error=stored.error,
        answers_done=progress.answers_done,
        answers_expected=progress.answers_expected,
        grades_done=progress.grades_done,
        grades_expected=progress.grades_expected,
        summary=_summary_models(frame),
        by_category=_category_payload(frame),
    )


@router.get(
    "/runs/{run_id}/failures",
    response_model=list[FailureOut],
    responses={404: {"description": "Unknown run"}},
)
def get_failures(run_id: str, request: Request) -> list[FailureOut]:
    """Return failures() for a run."""
    directory = Path(request.app.state.runs_directory)
    _stored_or_http(run_id, directory)
    progress = run_progress(run_id, directory=directory)
    frame = _frame_for(request.app, run_id, progress.grades_done)
    rows = failures(frame)
    return [FailureOut.model_validate(row.to_dict()) for _, row in rows.iterrows()]


@router.get("/compare", response_model=CompareResponse, responses={404: {"description": "Unknown run"}})
def compare_runs(
    request: Request,
    a: Annotated[str, Query(description="First run id.")],
    b: Annotated[str, Query(description="Second run id.")],
) -> CompareResponse:
    """Return compare() for the tasks in run a and run b."""
    left = _completed_frame(request.app, a)
    right = _completed_frame(request.app, b)
    left = left.copy()
    right = right.copy()
    left["model"] = a
    right["model"] = b
    combined = pd.concat([left, right], ignore_index=True)
    try:
        result = compare(combined, a, b)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return CompareResponse.model_validate(result)


def _execute_run(
    app: FastAPI,
    run_id: str,
    provider_name: str,
    dataset: Path,
    *,
    model: str | None = None,
    temperature: float | None = None,
    workers: int | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    attempts: int = 3,
) -> None:
    directory = Path(app.state.runs_directory)
    try:
        provider = create_provider(provider_name, model=model, temperature=temperature)
        catalog = _rubric_catalog_from_app(app)
        answer_run(
            dataset,
            provider,
            run_id,
            attempts=attempts,
            workers=workers,
            directory=directory,
            rubrics=catalog,
            temperature=temperature,
            provider_name=provider_name,
        )
        judge = None
        if judge_provider:
            judge = create_judge_provider(judge_provider, model=judge_model)
        grade_run(
            run_id,
            judge=judge,
            judge_model=judge_model,
            directory=directory,
            rubrics=catalog,
        )
    except Exception as exc:  # noqa: BLE001 - record the failure instead of dropping the run
        _mark_failed(directory, run_id, str(exc))
        _update_state(app, run_id, status="failed", error=str(exc))


def _execute_grade(
    app: FastAPI,
    run_id: str,
    judge_provider: str | None,
    judge_model: str | None,
) -> None:
    directory = Path(app.state.runs_directory)
    try:
        judge = None
        if judge_provider:
            judge = create_judge_provider(judge_provider, model=judge_model)
        grade_run(
            run_id,
            judge=judge,
            judge_model=judge_model,
            directory=directory,
            rubrics=_rubric_catalog_from_app(app),
        )
    except Exception as exc:  # noqa: BLE001 - record the failure instead of dropping the run
        _mark_failed(directory, run_id, str(exc))


def _rubric_catalog(request: Request) -> dict[str, Rubric]:
    return _rubric_catalog_from_app(request.app)


def _rubric_catalog_from_app(app: FastAPI) -> dict[str, Rubric]:
    directory = Path(app.state.rubrics_directory)
    if not directory.is_dir():
        return {}
    return load_rubrics(directory)


def _dataset_path(request: Request, dataset: str | None) -> Path:
    path = Path(dataset) if dataset else Path(request.app.state.default_dataset)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"dataset not found: {path}")
    return path


def _put_state(app: FastAPI, state: _RunState) -> None:
    with app.state.lock:
        app.state.runs[state.run_id] = state


def _update_state(
    app: FastAPI,
    run_id: str,
    *,
    status: RunStatusName,
    error: str | None = None,
) -> None:
    with app.state.lock:
        state = app.state.runs[run_id]
        state.status = status
        state.error = error


def _all_states(app: FastAPI) -> list[_RunState]:
    directory = Path(app.state.runs_directory)
    with app.state.lock:
        known = {state.run_id: state for state in app.state.runs.values()}
    for run_id in list_run_ids(directory=directory):
        known[run_id] = _state_from_disk(app, run_id)
    return [known[run_id] for run_id in sorted(known)]


def _state_from_disk(app: FastAPI, run_id: str) -> _RunState:
    stored = _stored_or_http(run_id, Path(app.state.runs_directory))
    return _RunState(
        run_id=run_id,
        status=_status_name(stored.status),
        provider=stored.provider or stored.model or "",
        dataset=stored.dataset or "",
        error=stored.error,
    )


def _stored_or_http(run_id: str, directory: Path) -> StoredRun:
    try:
        stored = load_run(run_id, directory=directory)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if stored is None:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
    return stored


def _status_name(status: str | None) -> RunStatusName:
    if status in {"answering", "answered", "grading", "graded", "interrupted", "failed"}:
        return status
    return "failed"


def _frame_for(app: FastAPI, run_id: str, grades_done: int) -> pd.DataFrame:
    if grades_done == 0:
        return load_runs([], directory=app.state.runs_directory)
    try:
        return load_runs(run_id, directory=app.state.runs_directory)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def _completed_frame(app: FastAPI, run_id: str) -> pd.DataFrame:
    stored = _stored_or_http(run_id, Path(app.state.runs_directory))
    if stored.status == "failed":
        detail = stored.error or f"run {run_id} failed"
        raise HTTPException(status_code=409, detail=detail)
    if stored.status != "graded":
        raise HTTPException(status_code=409, detail=f"run {run_id} is still {stored.status}")
    return _frame_for(app, run_id, 1)


def _mark_failed(directory: Path, run_id: str, error: str) -> None:
    if load_run(run_id, directory=directory) is None:
        return
    upsert_run(run_id, directory=directory, status="failed", error=error)


def _summary_models(frame: pd.DataFrame) -> list[ModelSummary]:
    stats = summary(frame)
    models: list[ModelSummary] = []
    for model, row in stats.iterrows():
        models.append(
            ModelSummary(
                model=str(model),
                pass_rate=float(row["pass_rate"]),
                mean_score=float(row["mean_score"]),
                mean_latency_ms=float(row["mean_latency_ms"]),
                p95_latency_ms=float(row["p95_latency_ms"]),
            )
        )
    return models


def _category_payload(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    table = by_category(frame)
    payload: dict[str, dict[str, float | None]] = {}
    for category, row in table.iterrows():
        payload[str(category)] = {
            str(model): None if pd.isna(value) else float(value) for model, value in row.items()
        }
    return payload


def _info(state: _RunState) -> RunInfo:
    return RunInfo(
        run_id=state.run_id,
        status=state.status,
        provider=state.provider or None,
        dataset=state.dataset or None,
        error=state.error,
    )


app = create_app()
