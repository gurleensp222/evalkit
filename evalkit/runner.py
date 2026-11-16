"""Run tasks through a provider and score the outputs."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from typing import Any

from evalkit.models import ScoreResult, Task, TaskResult
from evalkit.providers import Provider
from evalkit.scorers import SCORERS, llm_judge

MAX_WORKERS = 4
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 30.0


def load_tasks(path: str | Path) -> list[Task]:
    """Load a JSONL dataset. Blank lines are ignored."""
    dataset = Path(path)
    tasks: list[Task] = []
    text = dataset.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            tasks.append(Task.model_validate_json(stripped))
        except ValueError as exc:
            raise ValueError(f"{dataset}:{line_number}: invalid task") from exc
    return tasks


def run_eval(
    tasks: str | Path | Sequence[Task | Mapping[str, Any]],
    provider: Provider,
    run_id: str,
    *,
    retries: int = DEFAULT_RETRIES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    judge: Provider | None = None,
) -> list[TaskResult]:
    """Score every task with the provider and return one result per task, in order.

    ``judge`` grades tasks whose scorer is ``llm_judge``.
    """
    if retries < 1:
        raise ValueError("retries must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    loaded = _coerce_tasks(tasks)
    if not loaded:
        return []

    def evaluate(task: Task) -> TaskResult:
        return _evaluate_task(
            task,
            provider,
            run_id,
            retries=retries,
            timeout=timeout,
            judge=judge,
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(evaluate, loaded))


def _coerce_tasks(tasks: str | Path | Sequence[Task | Mapping[str, Any]]) -> list[Task]:
    if isinstance(tasks, (str, Path)):
        return load_tasks(tasks)
    loaded: list[Task] = []
    for item in tasks:
        if isinstance(item, Task):
            loaded.append(item)
        else:
            loaded.append(Task.model_validate(item))
    return loaded


def _evaluate_task(
    task: Task,
    provider: Provider,
    run_id: str,
    *,
    retries: int,
    timeout: float,
    judge: Provider | None,
) -> TaskResult:
    started = time.perf_counter()
    output = ""
    try:
        output = _generate_with_retries(
            provider,
            task.prompt,
            task.id,
            retries=retries,
            timeout=timeout,
        )
        scored = _score_output(task, output, judge)
    except Exception as exc:  # noqa: BLE001 - one task must not abort the run
        output = ""
        scored = ScoreResult(score=0.0, passed=False, reason=f"provider failed: {exc}")
    elapsed_ms = (time.perf_counter() - started) * 1000
    return TaskResult(
        run_id=run_id,
        task_id=task.id,
        category=task.category,
        model=provider.name,
        output=output,
        score=scored.score,
        passed=scored.passed,
        reason=scored.reason,
        latency_ms=elapsed_ms,
    )


def _generate_with_retries(
    provider: Provider,
    prompt: str,
    task_id: str,
    *,
    retries: int,
    timeout: float,
) -> str:
    last_error: Exception | None = None
    for _attempt in range(retries):
        try:
            return _generate_once(provider, prompt, task_id, timeout)
        except Exception as exc:  # noqa: BLE001 - retry any provider failure
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("provider failed")


def _generate_once(provider: Provider, prompt: str, task_id: str, timeout: float) -> str:
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["value"] = provider.generate(prompt, task_id=task_id)
        except Exception as exc:  # noqa: BLE001 - return the error to the caller thread
            box["error"] = exc

    worker = Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"timed out after {timeout:g} seconds")
    if "error" in box:
        error = box["error"]
        if isinstance(error, Exception):
            raise error
        raise RuntimeError(str(error))
    value = box.get("value")
    if not isinstance(value, str):
        raise TypeError("provider.generate must return str")
    return value


def _score_output(task: Task, output: str, judge: Provider | None) -> ScoreResult:
    scorer = SCORERS.get(task.scorer)
    if scorer is None:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"unknown scorer: {task.scorer}",
        )
    try:
        if scorer is llm_judge:
            if judge is None:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    reason="llm_judge requires a judge provider",
                )
            return llm_judge(output, task.expected, prompt=task.prompt, judge=judge)
        return scorer(output, task.expected)
    except Exception as exc:  # noqa: BLE001 - a scorer bug should be a failed score
        return ScoreResult(score=0.0, passed=False, reason=f"scorer failed: {exc}")
