"""Command line: ``evalkit run``, ``evalkit report``, and ``evalkit show``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import typer

from evalkit.models import TaskResult
from evalkit.providers import ProviderError, create_provider
from evalkit.runner import run_eval
from evalkit.storage import load_results, save_results

app = typer.Typer(help="Run model evaluations and report the scores.", no_args_is_help=True)

DATASET_OPTION = typer.Option(
    Path("datasets/sample_tasks.jsonl"),
    "--dataset",
    help="JSONL file of tasks.",
)
PROVIDER_OPTION = typer.Option(
    "mock",
    "--provider",
    help="mock, ollama, openai, or anthropic.",
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


@app.command()
def run(
    dataset: Path = DATASET_OPTION,
    provider: str = PROVIDER_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    workers: int | None = WORKERS_OPTION,
    model: str | None = MODEL_OPTION,
    temperature: float | None = TEMPERATURE_OPTION,
) -> None:
    """Run every task in a dataset and save the scores."""
    if not dataset.is_file():
        typer.echo(f"dataset not found: {dataset}", err=True)
        raise typer.Exit(1)
    try:
        selected = create_provider(provider, model=model, temperature=temperature)
        chosen_id = run_id or _new_run_id()
        results = run_eval(dataset, selected, chosen_id, workers=workers)
    except ProviderError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    path = save_results(chosen_id, results)
    _print_summary(chosen_id, results)
    typer.echo(f"saved {path}")


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
