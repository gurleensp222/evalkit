"""Command line: ``evalkit run`` and ``evalkit report``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

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


@app.command()
def run(
    dataset: Path = DATASET_OPTION,
    provider: str = PROVIDER_OPTION,
    run_id: str | None = RUN_ID_OPTION,
) -> None:
    """Run every task in a dataset and save the scores."""
    if not dataset.is_file():
        typer.echo(f"dataset not found: {dataset}", err=True)
        raise typer.Exit(1)
    try:
        selected = create_provider(provider)
    except ProviderError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    chosen_id = run_id or _new_run_id()
    results = run_eval(dataset, selected, chosen_id)
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


def _new_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid4().hex[:8]}"


def _print_summary(run_id: str, results: list[TaskResult]) -> None:
    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    mean = sum(result.score for result in results) / len(results) if results else 0.0
    typer.echo(
        f"run_id: {run_id}  tasks: {len(results)}  "
        f"passed: {passed}  failed: {failed}  mean_score: {mean:.2f}"
    )
    for result in results:
        label = "pass" if result.passed else "fail"
        flags = "  unstable" if result.unstable else ""
        kind = f"  {result.failure_type}" if result.failure_type else ""
        typer.echo(
            f"{result.task_id}  attempt {result.attempt}  {label}  {result.score:.2f}  "
            f"{result.latency_ms:.0f}ms{flags}{kind}  {_short(result.reason)}"
        )


def _short(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


if __name__ == "__main__":
    app()
