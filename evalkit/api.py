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
from evalkit.providers import ProviderError, create_provider
from evalkit.runner import ensure_model_available, load_tasks, run_eval
from evalkit.storage import DEFAULT_RUNS_DIR, list_run_ids, load_results, save_results

DEFAULT_DATASET = Path("datasets/sample_tasks.jsonl")
RunStatusName = Literal["running", "completed", "failed"]


class HealthResponse(BaseModel):
    """Liveness check."""

    status: Literal["ok"] = "ok"


class TaskOut(BaseModel):
    """One dataset row."""

    id: str
    category: str
    prompt: str
    scorer: str
    expected: Any = Field(description="Scorer-specific expected value.")


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


class RunCreated(BaseModel):
    """Accepted run. The evaluation continues in the background."""

    run_id: str
    status: Literal["running"]


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
    """Summary and per-category pass rates for a finished or in-progress run."""

    run_id: str
    status: RunStatusName
    provider: str | None = None
    error: str | None = None
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
) -> FastAPI:
    """Build the API. Tests pass their own runs directory and dataset."""
    app = FastAPI(title="evalkit", version="0.1.0")
    app.state.runs_directory = runs_directory
    app.state.default_dataset = default_dataset
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
        tasks = load_tasks(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return [TaskOut.model_validate(task.model_dump()) for task in tasks]


@router.post("/runs", response_model=RunCreated, status_code=202)
def start_run(body: RunCreate, request: Request, background: BackgroundTasks) -> RunCreated:
    """Start an evaluation in the background and return its run id."""
    dataset = Path(body.dataset)
    if not dataset.is_file():
        raise HTTPException(status_code=404, detail=f"dataset not found: {dataset}")
    try:
        selected = create_provider(body.provider, model=body.model, temperature=body.temperature)
        ensure_model_available(selected)
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    run_id = uuid4().hex
    _put_state(
        request.app,
        _RunState(
            run_id=run_id,
            status="running",
            provider=body.provider.strip().lower(),
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
    )
    return RunCreated(run_id=run_id, status="running")


@router.get("/runs", response_model=list[RunInfo])
def list_runs(request: Request) -> list[RunInfo]:
    """List known run ids and whether each one is running, completed, or failed."""
    return [_info(state) for state in _all_states(request.app)]


@router.get("/runs/{run_id}", response_model=RunDetail, responses={404: {"description": "Unknown run"}})
def get_run(run_id: str, request: Request) -> RunDetail:
    """Return summary() and by_category() for a run."""
    state = _require_state(request.app, run_id)
    frame = _frame_or_empty(request.app, state)
    return RunDetail(
        run_id=state.run_id,
        status=state.status,
        provider=state.provider,
        error=state.error,
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
    state = _require_state(request.app, run_id)
    frame = _frame_or_empty(request.app, state)
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
) -> None:
    try:
        provider = create_provider(provider_name, model=model, temperature=temperature)
        results = run_eval(dataset, provider, run_id, workers=workers)
        save_results(run_id, results, directory=app.state.runs_directory)
    except Exception as exc:  # noqa: BLE001 - record the failure instead of dropping the run
        _update_state(app, run_id, status="failed", error=str(exc))
        return
    _update_state(app, run_id, status="completed")


def _dataset_path(request: Request, dataset: str | None) -> Path:
    path = Path(dataset) if dataset else Path(request.app.state.default_dataset)
    if not path.is_file():
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
    with app.state.lock:
        known = {state.run_id: state for state in app.state.runs.values()}
    for run_id in list_run_ids(directory=Path(app.state.runs_directory)):
        if run_id not in known:
            known[run_id] = _state_from_disk(app, run_id)
    return [known[run_id] for run_id in sorted(known)]


def _require_state(app: FastAPI, run_id: str) -> _RunState:
    if not run_id or run_id in {".", ".."} or any(sep in run_id for sep in ("/", "\\", "\0")):
        raise HTTPException(status_code=400, detail=f"invalid run id: {run_id}")
    with app.state.lock:
        state = app.state.runs.get(run_id)
    if state is not None:
        return state
    if run_id in list_run_ids(directory=Path(app.state.runs_directory)):
        return _state_from_disk(app, run_id)
    raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")


def _state_from_disk(app: FastAPI, run_id: str) -> _RunState:
    try:
        results = load_results(run_id, directory=app.state.runs_directory)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    provider = results[0].model if results else None
    return _RunState(
        run_id=run_id,
        status="completed",
        provider=provider or "",
        dataset="",
    )


def _frame_or_empty(app: FastAPI, state: _RunState) -> pd.DataFrame:
    if state.status != "completed":
        return load_runs([], directory=app.state.runs_directory)
    return _completed_frame(app, state.run_id)


def _completed_frame(app: FastAPI, run_id: str) -> pd.DataFrame:
    state = _require_state(app, run_id)
    if state.status == "running":
        raise HTTPException(status_code=409, detail=f"run {run_id} is still running")
    if state.status == "failed":
        detail = state.error or f"run {run_id} failed"
        raise HTTPException(status_code=409, detail=detail)
    try:
        return load_runs(run_id, directory=app.state.runs_directory)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


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
