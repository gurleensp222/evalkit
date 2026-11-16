"""Command line: ``evalkit run``, ``evalkit report``, and ``evalkit show``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import typer

from evalkit.agents import load_agents, resolve_agents
from evalkit.models import TaskResult
from evalkit.pipeline import answer_run, grade_run
from evalkit.providers import ProviderError, create_judge_provider, create_provider
from evalkit.rubrics import (
    Rubric,
    lint_extra_criteria,
    lint_rubrics,
    load_prompts,
    load_rubrics,
)
from evalkit.storage import (
    DATABASE_NAME,
    DEFAULT_RUNS_DIR,
    SQLITE_DATASET,
    StoredCriterionGrade,
    load_criterion_grades,
    load_latest_rubrics,
    load_results,
    load_run,
)

app = typer.Typer(help="Run model evaluations and report the scores.", no_args_is_help=True)

DATASET_OPTION = typer.Option(
    Path("datasets/sample_tasks.jsonl"),
    "--dataset",
    help="JSONL file of tasks.",
)
PROVIDER_OPTION = typer.Option(
    None,
    "--provider",
    help="Overrides agents.yaml. Defaults to mock.",
)
RUN_ID_OPTION = typer.Option(
    None,
    "--run-id",
    help="Id used for this run in runs/evalkit.db.",
)
WORKERS_OPTION = typer.Option(
    None,
    "--workers",
    min=1,
    help="How many tasks to run at once. Ollama uses 1 unless this is set.",
)
MODEL_OPTION = typer.Option(
    None,
    "--model",
    help="Model id. Ollama defaults to llama3.2.",
)
TEMPERATURE_OPTION = typer.Option(
    None,
    "--temperature",
    help="Sampling temperature for Ollama.",
)
JUDGE_PROVIDER_OPTION = typer.Option(
    None,
    "--judge-provider",
    help="Provider that grades rubric tasks. Uses temperature 0 and JSON output.",
)
JUDGE_MODEL_OPTION = typer.Option(
    None,
    "--judge-model",
    help="Model id for the judge.",
)
ATTEMPTS_OPTION = typer.Option(
    3,
    "--attempts",
    min=1,
    help="How many times to run each task.",
)
RUBRICS_OPTION = typer.Option(
    Path("datasets/rubrics"),
    "--directory",
    help="Folder of YAML rubrics.",
)
STRICT_OPTION = typer.Option(
    False,
    "--strict",
    help="Exit with code 1 when lint finds a warning.",
)
DEFAULT_RUBRICS = Path("datasets/rubrics")
ANSWER_DATASET_OPTION = typer.Option(
    Path("datasets/sample_tasks.jsonl"),
    "--dataset",
    help="JSONL file or directory of prompts.",
)
ANSWER_PROVIDER_OPTION = typer.Option(
    "mock",
    "--provider",
    help="mock, ollama, openai, or anthropic.",
)
ANSWER_RUN_ID_OPTION = typer.Option(
    None,
    "--run-id",
    help="Id used for this run in runs/evalkit.db.",
)
ANSWER_WORKERS_OPTION = typer.Option(
    None,
    "--workers",
    min=1,
    help="How many tasks to run at once. Ollama uses 1 unless this is set.",
)
ANSWER_MODEL_OPTION = typer.Option(
    None,
    "--model",
    help="Model id. Ollama defaults to llama3.2.",
)
ANSWER_TEMPERATURE_OPTION = typer.Option(
    None,
    "--temperature",
    help="Sampling temperature for Ollama.",
)
ANSWER_ATTEMPTS_OPTION = typer.Option(
    3,
    "--attempts",
    min=1,
    help="How many times to run each task.",
)
RESUME_OPTION = typer.Option(
    None,
    "--resume",
    help="Continue a run and skip answers that are already saved.",
)
GRADE_JUDGE_PROVIDER_OPTION = typer.Option(
    None,
    "--judge-provider",
    help="Provider that grades rubric tasks. Uses temperature 0 and JSON output.",
)
GRADE_JUDGE_MODEL_OPTION = typer.Option(
    None,
    "--judge-model",
    help="Model id for the judge.",
)
CONFIG_OPTION = typer.Option(
    None,
    "--config",
    help="YAML file with answer_agent and grader_agent.",
)
GRADER_CONTEXT_OPTION = typer.Option(
    "with_prompt",
    "--grader-context",
    help="response_only sends the response and rubric. with_prompt also sends the task.",
)
GRADE_CONTEXT_OPTION = typer.Option(
    "with_prompt",
    "--grader-context",
    help="response_only sends the response and rubric. with_prompt also sends the task.",
)

rubrics_app = typer.Typer(help="List, validate, and lint rubric files.", no_args_is_help=True)
app.add_typer(rubrics_app, name="rubrics")


@app.command()
def run(
    dataset: Path = DATASET_OPTION,
    provider: str | None = PROVIDER_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    workers: int | None = WORKERS_OPTION,
    model: str | None = MODEL_OPTION,
    temperature: float | None = TEMPERATURE_OPTION,
    judge_provider: str | None = JUDGE_PROVIDER_OPTION,
    judge_model: str | None = JUDGE_MODEL_OPTION,
    attempts: int = ATTEMPTS_OPTION,
    config: Path | None = CONFIG_OPTION,
    grader_context: str = GRADER_CONTEXT_OPTION,
) -> None:
    """Answer every task, then grade the saved answers."""
    if str(dataset) != SQLITE_DATASET and not dataset.exists():
        typer.echo(f"dataset not found: {dataset}", err=True)
        raise typer.Exit(1)
    chosen_id = run_id or _new_run_id()
    try:
        context = _grader_context(grader_context)
        agents = load_agents(config) if config else None
        resolved = resolve_agents(
            agents,
            provider=provider,
            model=model,
            temperature=temperature,
            judge_provider=judge_provider,
            judge_model=judge_model,
        )
        catalog = _catalog_or_empty(DEFAULT_RUBRICS)
        selected = create_provider(resolved.provider, **_provider_kwargs(resolved))
        answer_run(
            dataset,
            selected,
            chosen_id,
            attempts=attempts,
            workers=workers,
            rubrics=catalog,
            temperature=resolved.temperature,
            provider_name=resolved.provider,
            on_progress=typer.echo,
        )
        judge = None
        if resolved.judge_provider:
            judge = create_judge_provider(
                resolved.judge_provider,
                **_judge_kwargs(resolved),
            )
        grade_run(
            chosen_id,
            judge=judge,
            judge_model=resolved.judge_model,
            rubrics=catalog,
            on_progress=typer.echo,
            grader_context=context,
        )
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (ProviderError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    results = load_results(chosen_id)
    _print_summary(chosen_id, results)
    typer.echo(f"saved {DEFAULT_RUNS_DIR / DATABASE_NAME}")


@app.command()
def answer(
    dataset: Path = ANSWER_DATASET_OPTION,
    provider: str = ANSWER_PROVIDER_OPTION,
    run_id: str | None = ANSWER_RUN_ID_OPTION,
    workers: int | None = ANSWER_WORKERS_OPTION,
    model: str | None = ANSWER_MODEL_OPTION,
    temperature: float | None = ANSWER_TEMPERATURE_OPTION,
    attempts: int = ANSWER_ATTEMPTS_OPTION,
    resume: str | None = RESUME_OPTION,
) -> None:
    """Generate answers and save each one as soon as it finishes."""
    chosen_id = resume or run_id or _new_run_id()
    try:
        catalog = _catalog_or_empty(DEFAULT_RUBRICS)
        if resume:
            stored = load_run(resume)
            if stored is None or not stored.provider:
                raise FileNotFoundError(resume)
            selected = create_provider(
                stored.provider,
                model=stored.model,
                temperature=stored.temperature,
            )
            answer_run(
                stored.dataset,
                selected,
                resume,
                resume=True,
                workers=workers,
                rubrics=catalog,
                on_progress=typer.echo,
            )
        else:
            if str(dataset) != SQLITE_DATASET and not dataset.exists():
                typer.echo(f"dataset not found: {dataset}", err=True)
                raise typer.Exit(1)
            selected = create_provider(provider, model=model, temperature=temperature)
            answer_run(
                dataset,
                selected,
                chosen_id,
                attempts=attempts,
                workers=workers,
                rubrics=catalog,
                temperature=temperature,
                provider_name=provider,
                on_progress=typer.echo,
            )
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (ProviderError, ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    typer.echo(f"saved {DEFAULT_RUNS_DIR / DATABASE_NAME}")


@app.command()
def grade(
    run_id: str,
    judge_provider: str | None = GRADE_JUDGE_PROVIDER_OPTION,
    judge_model: str | None = GRADE_JUDGE_MODEL_OPTION,
    grader_context: str = GRADE_CONTEXT_OPTION,
) -> None:
    """Score the answers saved for a run. A new judge is stored beside older ones."""
    try:
        context = _grader_context(grader_context)
        catalog = _catalog_or_empty(DEFAULT_RUBRICS)
        judge = None
        if judge_provider:
            judge = create_judge_provider(judge_provider, model=judge_model)
        grade_run(
            run_id,
            judge=judge,
            judge_model=judge_model,
            rubrics=catalog,
            on_progress=typer.echo,
            grader_context=context,
        )
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (ProviderError, ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    results = load_results(run_id, judge=judge_model)
    _print_summary(run_id, results)
    typer.echo(f"saved {DEFAULT_RUNS_DIR / DATABASE_NAME}")


@app.command()
def report(run_id: str) -> None:
    """Print the saved scores for a run."""
    try:
        results = load_results(run_id)
    except FileNotFoundError:
        typer.echo(f"no run named {run_id}", err=True)
        raise typer.Exit(1) from None
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    _print_summary(run_id, results)
    _print_criteria(run_id, results)


@app.command()
def show(run_id: str, task_id: str) -> None:
    """Print every attempt's raw output, score, and reason for one task."""
    try:
        results = load_results(run_id)
    except FileNotFoundError:
        typer.echo(f"no run named {run_id}", err=True)
        raise typer.Exit(1) from None
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    matched = [result for result in results if result.task_id == task_id]
    if not matched:
        typer.echo(f"no task named {task_id} in run {run_id}", err=True)
        raise typer.Exit(1)
    for result in matched:
        typer.echo(f"attempt {result.attempt}")
        typer.echo(f"score: {result.score:.2f}")
        typer.echo(f"reason: {result.reason}")
        typer.echo("output:")
        typer.echo(result.output)
        typer.echo("")


@rubrics_app.command("list")
def list_rubrics(directory: Path = RUBRICS_OPTION) -> None:
    """Print each rubric id, version, pass threshold, and content hash."""
    catalog = _require_rubrics(directory)
    for rubric in catalog.values():
        typer.echo(
            f"{rubric.id}  v{rubric.version}  threshold {rubric.pass_threshold:.2f}  "
            f"{rubric.content_hash[:12]}"
        )


@rubrics_app.command("validate")
def validate_rubrics(directory: Path = RUBRICS_OPTION) -> None:
    """Check every rubric file and print how many loaded."""
    catalog = _require_rubrics(directory)
    typer.echo(f"{len(catalog)} rubrics ok")


@rubrics_app.command("lint")
def lint(directory: Path = RUBRICS_OPTION, strict: bool = STRICT_OPTION) -> None:
    """Warn about criteria that are hard for a grader to apply."""
    catalog = _require_rubrics(directory)
    warnings = lint_rubrics(catalog)
    prompts = directory.parent / "prompts"
    if prompts.is_dir():
        try:
            warnings.extend(lint_extra_criteria(load_prompts(prompts, catalog)))
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from None
    if warnings:
        for warning in warnings:
            typer.echo(warning)
        if strict:
            raise typer.Exit(1)
        return
    typer.echo(f"{len(catalog)} rubrics ok")


def _require_rubrics(directory: Path) -> dict[str, Rubric]:
    try:
        return load_rubrics(directory)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None


def _catalog_or_empty(directory: Path) -> dict[str, Rubric]:
    catalog: dict[str, Rubric] = {}
    if directory.is_dir():
        catalog.update(load_rubrics(directory))
    catalog.update(load_latest_rubrics(directory=DEFAULT_RUNS_DIR))
    return catalog


def _provider_kwargs(resolved: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": getattr(resolved, "model", None),
        "temperature": getattr(resolved, "temperature", None),
    }
    system = getattr(resolved, "system_prompt", None)
    if isinstance(system, str) and system.strip():
        kwargs["system"] = system
    return kwargs


def _judge_kwargs(resolved: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": getattr(resolved, "judge_model", None),
        "temperature": getattr(resolved, "judge_temperature", 0),
    }
    system = getattr(resolved, "judge_system_prompt", None)
    if isinstance(system, str) and system.strip():
        kwargs["system"] = system
    return kwargs


def _grader_context(value: str) -> str:
    if value not in {"response_only", "with_prompt"}:
        raise ValueError("grader context must be response_only or with_prompt")
    return value


def _print_criteria(run_id: str, results: list[TaskResult]) -> None:
    critical = [
        result
        for result in results
        if isinstance(result.failure_type, str) and result.failure_type.startswith("critical:")
    ]
    judge = ""
    if results and results[0].judge:
        judge = results[0].judge
    grades = load_criterion_grades(run_id, judge=judge) if results else []
    if not grades and not critical:
        return
    if grades:
        typer.echo("")
        typer.echo("criteria")
        typer.echo(_criterion_table(results, grades))
    typer.echo("")
    typer.echo("critical failures")
    if not critical:
        typer.echo("none")
        return
    for result in critical:
        typer.echo(
            f"{result.task_id}  attempt {result.attempt}  {result.model}  {result.failure_type}"
        )


def _criterion_table(results: list[TaskResult], grades: list[StoredCriterionGrade]) -> str:
    """Met rate for each criterion, split by the model that wrote the answer."""
    model_of = {(result.task_id, result.attempt): result.model for result in results}
    frame = pd.DataFrame(
        [
            {
                "criterion": grade.criterion_id,
                "model": model_of.get((grade.task_id, grade.attempt), ""),
                "met": 1.0 if grade.met else 0.0,
            }
            for grade in grades
        ]
    )
    rates = frame.pivot_table(index="criterion", columns="model", values="met", aggfunc="mean")
    order = list(dict.fromkeys(grade.criterion_id for grade in grades))
    rates = rates.reindex(order)
    rates.index.name = "criterion"
    with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
        return rates.to_string(float_format=lambda value: f"{float(value):.2f}")


def _new_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid4().hex[:8]}"


def _print_summary(run_id: str, results: list[TaskResult]) -> None:
    by_task: dict[str, list[bool]] = {}
    for result in results:
        by_task.setdefault(result.task_id, []).append(result.passed)
    passed_attempts = sum(1 for result in results if result.passed)
    tasks_passing = sum(1 for attempts in by_task.values() if all(attempts))
    typer.echo(
        f"run_id: {run_id}  tasks: {len(by_task)}  attempts: {len(results)}  "
        f"passed attempts: {passed_attempts}  tasks passing all attempts: {tasks_passing}"
    )
    table = _summary_table(results)
    if table:
        typer.echo(table)


def _summary_table(results: list[TaskResult]) -> str:
    """One line per attempt. Wide cells stay on that line."""
    if not results:
        return ""
    frame = pd.DataFrame(
        [
            {
                "task": result.task_id,
                "attempt": result.attempt,
                "result": "pass" if result.passed else "fail",
                "score": result.score,
                "latency_ms": result.latency_ms,
                "flags": _flags(result),
                "failure": result.failure_type or "",
                "reason": _short(result.reason),
            }
            for result in results
        ]
    )
    with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
        return frame.to_string(
            index=False,
            line_width=1_000_000,
            formatters={
                "score": lambda value: f"{float(value):.2f}",
                "latency_ms": lambda value: f"{float(value):.0f}",
            },
        )


def _flags(result: TaskResult) -> str:
    parts: list[str] = []
    if result.unstable:
        parts.append("unstable")
    if result.output_varies:
        parts.append("output varies")
    return ", ".join(parts)


def _short(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


if __name__ == "__main__":
    app()
