"""Save and load run results as JSONL files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from evalkit.models import TaskResult

DEFAULT_RUNS_DIR = Path("runs")


def save_results(
    run_id: str,
    results: Sequence[TaskResult],
    *,
    directory: Path = DEFAULT_RUNS_DIR,
) -> Path:
    """Write ``runs/<run_id>.jsonl`` and return the path."""
    path = _results_path(run_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [result.model_dump_json() for result in results]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")
    return path


def load_results(run_id: str, *, directory: Path = DEFAULT_RUNS_DIR) -> list[TaskResult]:
    """Read ``runs/<run_id>.jsonl``."""
    path = _results_path(run_id, directory)
    text = path.read_text(encoding="utf-8")
    results: list[TaskResult] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            results.append(TaskResult.model_validate_json(stripped))
    return results


def _results_path(run_id: str, directory: Path) -> Path:
    if not run_id or run_id in {".", ".."} or any(sep in run_id for sep in ("/", "\\", "\0")):
        raise ValueError(f"invalid run id: {run_id!r}")
    path = directory / f"{run_id}.jsonl"
    if path.name != f"{run_id}.jsonl":
        raise ValueError(f"invalid run id: {run_id!r}")
    return path
