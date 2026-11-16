"""Run tasks through a provider and score the outputs."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from typing import Any

from evalkit.grading import GRADER_CONTEXTS, grade_response
from evalkit.models import ScoreResult, Task, TaskResult
from evalkit.providers import Provider
from evalkit.rubrics import Rubric, applied_rubric, load_prompts
from evalkit.scorers import SCORERS, llm_judge, no_fabricated_links

MAX_WORKERS = 4
DEFAULT_RETRIES = 3
DEFAULT_REPEATS = 3
DEFAULT_TIMEOUT_SECONDS = 30.0
FAILURE_TIMEOUT = "timeout"
FAILURE_FORMAT = "format error"
FAILURE_REFUSAL = "refusal"
FAILURE_WRONG = "wrong answer"

_FORMAT_MARKERS = (
    "invalid json",
    "missing keys",
    "must be an object",
    "no code",
    "syntaxerror",
    "json_valid",
)
_REFUSAL_MARKERS = (
    "i don't know",
    "i dont know",
    "i can't",
    "i cannot",
    "i won't",
    "i will not",
    "as an ai",
    "i'm sorry",
    "i am sorry",
    "cannot assist",
    "i'm unable",
    "i am unable",
)


def ensure_model_available(provider: Provider) -> None:
    """Ask the provider to confirm its model is installed, when it can."""
    checker = getattr(provider, "check_model_available", None)
    if callable(checker):
        checker()


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
    timeout: float | None = None,
    judge: Provider | None = None,
    repeats: int = DEFAULT_REPEATS,
    workers: int | None = None,
    rubrics: Mapping[str, Rubric] | None = None,
    grader_context: str = "with_prompt",
) -> list[TaskResult]:
    """Score every task ``repeats`` times and return the attempts in task order.

    A task is unstable when its attempts disagree on ``passed``. ``output_varies``
    is set when the answer text differs. A single attempt skips both checks.
    ``judge`` grades ``llm_judge`` tasks and
    tasks that name a ``rubric_id``. Ollama defaults to one worker and a 120 second
    HTTP timeout. A provider with ``check_model_available`` is checked before any
    task runs. Unknown rubric ids fail while the prompts are loaded.
    """
    if timeout is None:
        timeout = float(getattr(provider, "default_timeout", DEFAULT_TIMEOUT_SECONDS))
    if workers is None:
        workers = int(getattr(provider, "default_workers", MAX_WORKERS))
    if retries < 1:
        raise ValueError("retries must be at least 1")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if grader_context not in GRADER_CONTEXTS:
        raise ValueError("grader context must be response_only or with_prompt")

    catalog = dict(rubrics or {})
    loaded = _coerce_tasks(tasks, catalog)
    if any(task.rubric_id for task in loaded) and judge is None:
        raise ValueError("rubric tasks require a judge provider")
    ensure_model_available(provider)
    if judge is not None:
        ensure_model_available(judge)
    if not loaded:
        return []

    def evaluate(task: Task) -> list[TaskResult]:
        attempts = [
            _evaluate_task(
                task,
                provider,
                run_id,
                retries=retries,
                timeout=timeout,
                judge=judge,
                attempt=attempt,
                rubrics=catalog,
                grader_context=grader_context,
            )
            for attempt in range(1, repeats + 1)
        ]
        apply_stability_flags(attempts, repeats)
        return attempts

    with ThreadPoolExecutor(max_workers=workers) as executor:
        grouped = executor.map(evaluate, loaded)
        return [result for attempts in grouped for result in attempts]


def _coerce_tasks(
    tasks: str | Path | Sequence[Task | Mapping[str, Any]],
    rubrics: Mapping[str, Rubric],
) -> list[Task]:
    if isinstance(tasks, (str, Path)):
        return load_prompts(tasks, rubrics)
    loaded: list[Task] = []
    missing: list[str] = []
    for item in tasks:
        task = item if isinstance(item, Task) else Task.model_validate(item)
        if task.rubric_id and task.rubric_id not in rubrics:
            missing.append(task.id)
        loaded.append(task)
    if missing:
        raise ValueError("unknown rubric_id for prompt ids: " + ", ".join(missing))
    return loaded


def _evaluate_task(
    task: Task,
    provider: Provider,
    run_id: str,
    *,
    retries: int,
    timeout: float,
    judge: Provider | None,
    attempt: int,
    rubrics: Mapping[str, Rubric],
    grader_context: str = "with_prompt",
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
        scored = _score_output(task, output, judge, rubrics, grader_context=grader_context)
    except Exception as exc:  # noqa: BLE001 - one task must not abort the run
        output = ""
        scored = ScoreResult(score=0.0, passed=False, reason=f"provider failed: {exc}")
    elapsed_ms = (time.perf_counter() - started) * 1000
    rubric = _applied(task, rubrics)
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
        attempt=attempt,
        failure_type=failure_for(scored, output),
        rubric_id=task.rubric_id,
        rubric_version=None if rubric is None else rubric.version,
        rubric_hash=None if rubric is None else rubric.content_hash,
    )


def apply_stability_flags(attempts: list[TaskResult], repeats: int) -> None:
    """Mark attempts that disagree. A single attempt is left unflagged."""
    if repeats <= 1:
        return
    unstable = len({result.passed for result in attempts}) > 1
    output_varies = len({result.output.strip() for result in attempts}) > 1
    for result in attempts:
        result.unstable = unstable
        result.output_varies = output_varies


def failure_for(scored: ScoreResult, output: str) -> str | None:
    """Use a grader-assigned failure when one was set, otherwise label the attempt."""
    if scored.forced_failure:
        return scored.forced_failure
    return failure_type_for(output, scored.reason, passed=scored.passed)


def failure_type_for(output: str, reason: str, *, passed: bool) -> str | None:
    """Label a failed attempt. Passed attempts have no failure type."""
    if passed:
        return None
    folded_reason = reason.casefold()
    folded_output = output.casefold()
    if "timed out" in folded_reason:
        return FAILURE_TIMEOUT
    if any(marker in folded_reason for marker in _FORMAT_MARKERS):
        return FAILURE_FORMAT
    if any(marker in folded_output or marker in folded_reason for marker in _REFUSAL_MARKERS):
        return FAILURE_REFUSAL
    return FAILURE_WRONG


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
    if getattr(provider, "uses_http_timeout", False):
        value = provider.generate(prompt, task_id=task_id, timeout=timeout)
        if not isinstance(value, str):
            raise TypeError("provider.generate must return str")
        return value

    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["value"] = provider.generate(prompt, task_id=task_id)
        except BaseException as exc:  # noqa: BLE001 - return the error, including Ctrl+C
            box["error"] = exc

    worker = Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"timed out after {timeout:g} seconds")
    if "error" in box:
        error = box["error"]
        if isinstance(error, BaseException):
            raise error
        raise RuntimeError(str(error))
    value = box.get("value")
    if not isinstance(value, str):
        raise TypeError("provider.generate must return str")
    return value


def _score_output(
    task: Task,
    output: str,
    judge: Provider | None,
    rubrics: Mapping[str, Rubric],
    *,
    grader_context: str = "with_prompt",
) -> ScoreResult:
    if task.rubric_id:
        return _score_with_rubric(task, output, judge, rubrics, grader_context=grader_context)
    scorer_name = task.scorer or ""
    scorer = SCORERS.get(scorer_name)
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
        if scorer is no_fabricated_links:
            return no_fabricated_links(output, task.allowed_urls)
        return scorer(output, task.expected)
    except Exception as exc:  # noqa: BLE001 - a scorer bug should be a failed score
        return ScoreResult(score=0.0, passed=False, reason=f"scorer failed: {exc}")


def _score_with_rubric(
    task: Task,
    output: str,
    judge: Provider | None,
    rubrics: Mapping[str, Rubric],
    *,
    grader_context: str = "with_prompt",
) -> ScoreResult:
    try:
        rubric = applied_rubric(task, rubrics)
    except ValueError as exc:
        return ScoreResult(score=0.0, passed=False, reason=str(exc))
    if rubric is None:
        return ScoreResult(score=0.0, passed=False, reason=f"unknown rubric: {task.rubric_id}")
    blocked = _failed_deterministic_check(task, output)
    if blocked is not None:
        return blocked
    if judge is None:
        return ScoreResult(score=0.0, passed=False, reason="llm_judge requires a judge provider")
    return grade_response(
        rubric,
        output,
        task.prompt,
        judge,
        context=grader_context,
    )


def _applied(task: Task, rubrics: Mapping[str, Rubric]) -> Rubric | None:
    try:
        return applied_rubric(task, rubrics)
    except ValueError:
        rubric = rubrics.get(task.rubric_id) if task.rubric_id else None
        return rubric


def _failed_deterministic_check(task: Task, output: str) -> ScoreResult | None:
    """Return a critical failure when a check listed on the task does not pass."""
    for name in task.deterministic_checks:
        if name != "no_fabricated_links":
            return ScoreResult(
                score=0.0,
                passed=False,
                reason=f"unknown deterministic check: {name}",
            )
        checked = no_fabricated_links(output, task.allowed_urls)
        if not checked.passed:
            return ScoreResult(
                score=0.0,
                passed=False,
                reason=checked.reason,
                forced_failure=f"critical:{name}",
            )
    return None
