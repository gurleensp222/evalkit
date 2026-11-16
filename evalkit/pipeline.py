"""Generate answers, save them, then grade the saved rows."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from evalkit.grading import GRADER_CONTEXTS
from evalkit.models import CriterionMark, ScoreResult, Task, TaskResult
from evalkit.providers import Provider
from evalkit.rubrics import Rubric, applied_rubric, load_prompts
from evalkit.runner import (
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_WORKERS,
    _generate_with_retries,
    _score_output,
    apply_stability_flags,
    ensure_model_available,
    failure_for,
)
from evalkit.storage import (
    DEFAULT_RUNS_DIR,
    SQLITE_DATASET,
    StoredAnswer,
    answer_keys,
    load_answers,
    load_run,
    load_stored_prompts,
    replace_criterion_grades,
    run_progress,
    save_answer,
    save_grade,
    upsert_run,
)

Progress = Callable[[str], None]


def answer_run(
    dataset: str | Path | None,
    provider: Provider,
    run_id: str,
    *,
    attempts: int = 3,
    workers: int | None = None,
    directory: Path = DEFAULT_RUNS_DIR,
    rubrics: Mapping[str, Rubric] | None = None,
    resume: bool = False,
    temperature: float | None = None,
    provider_name: str | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout: float | None = None,
    on_progress: Progress | None = None,
) -> None:
    """Generate each attempt and commit it before the next one is reported.

    ``resume`` skips task and attempt pairs that already have a row. Ctrl+C
    marks the run interrupted and is raised again.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if retries < 1:
        raise ValueError("retries must be at least 1")
    if timeout is None:
        timeout = float(getattr(provider, "default_timeout", DEFAULT_TIMEOUT_SECONDS))
    if workers is None:
        workers = int(getattr(provider, "default_workers", MAX_WORKERS))
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    catalog = dict(rubrics or {})
    stored = load_run(run_id, directory=directory)
    saved = answer_keys(run_id, directory=directory)
    if resume:
        if stored is None or not stored.dataset:
            raise FileNotFoundError(run_id)
        dataset_value: str | Path = stored.dataset
        attempts = stored.attempts or attempts
    else:
        if dataset is None:
            raise ValueError("dataset is required")
        if saved:
            raise ValueError(f"run {run_id} already has answers; pass --resume")
        dataset_value = dataset
    tasks, dataset_label = _tasks_from_dataset(dataset_value, catalog, directory)
    ensure_model_available(provider)
    pending = [
        (task, attempt)
        for task in tasks
        for attempt in range(1, attempts + 1)
        if (task.id, attempt) not in saved
    ]
    total = len(tasks) * attempts
    if not resume or pending:
        upsert_run(
            run_id,
            directory=directory,
            status="answering",
            model=provider.name,
            provider=provider_name or (stored.provider if stored else None) or provider.name,
            dataset=dataset_label,
            attempts=attempts,
            temperature=stored.temperature if resume and temperature is None else temperature,
            task_count=len(tasks),
            error=None,
        )
    if not pending:
        if not resume:
            upsert_run(run_id, directory=directory, status="answered")
        return
    try:
        _answer_pending(
            pending,
            provider,
            run_id,
            retries=retries,
            timeout=timeout,
            workers=workers,
            total=total,
            directory=directory,
            on_progress=on_progress,
        )
    except KeyboardInterrupt:
        upsert_run(run_id, directory=directory, status="interrupted")
        raise
    except Exception as exc:
        upsert_run(run_id, directory=directory, status="failed", error=str(exc))
        raise
    if pending:
        upsert_run(run_id, directory=directory, status="answered")


def grade_run(
    run_id: str,
    *,
    judge: Provider | None = None,
    judge_model: str | None = None,
    directory: Path = DEFAULT_RUNS_DIR,
    rubrics: Mapping[str, Rubric] | None = None,
    on_progress: Progress | None = None,
    grader_context: str = "with_prompt",
) -> list[TaskResult]:
    """Score saved answers and commit each result on its own.

    A second judge is stored beside the first. Deterministic scorers ignore
    ``judge`` unless the task asks for one.
    """
    if grader_context not in GRADER_CONTEXTS:
        raise ValueError("grader context must be response_only or with_prompt")
    stored = load_run(run_id, directory=directory)
    if stored is None:
        raise FileNotFoundError(run_id)
    if not stored.dataset:
        raise ValueError(f"run {run_id} has no dataset")
    catalog = dict(rubrics or {})
    tasks, _dataset_label = _tasks_from_dataset(stored.dataset, catalog, directory)
    answers = load_answers(run_id, directory=directory)
    if not answers:
        raise ValueError(f"run {run_id} has no answers")
    if _needs_judge(tasks, answers) and judge is None:
        raise ValueError("rubric tasks require a judge provider")
    if judge is not None:
        ensure_model_available(judge)

    by_id = {task.id: task for task in tasks}
    missing = sorted({answer.task_id for answer in answers if answer.task_id not in by_id})
    if missing:
        raise ValueError("unknown task ids in answers: " + ", ".join(missing))
    label = _judge_label(judge, judge_model)
    upsert_run(run_id, directory=directory, status="grading", error=None)
    repeats = stored.attempts or 1
    saved: list[TaskResult] = []
    total = len(answers)
    index = 0
    try:
        for task in tasks:
            batch = _answers_for(answers, task.id)
            if not batch:
                continue
            pairs = [
                _grade_answer(
                    task,
                    item,
                    run_id,
                    judge,
                    catalog,
                    grader_context=grader_context,
                )
                for item in batch
            ]
            scored = [result for result, _marks in pairs]
            apply_stability_flags(scored, repeats)
            for result, marks in pairs:
                save_grade(result, judge=label, directory=directory)
                _save_marks(result, label, marks, directory)
                saved.append(result)
                index += 1
                if on_progress is not None:
                    on_progress(
                        f"[{index}/{total}] {result.task_id} attempt {result.attempt}  "
                        f"score {result.score:.2f}"
                    )
    except KeyboardInterrupt:
        upsert_run(run_id, directory=directory, status="interrupted")
        raise
    except Exception as exc:
        upsert_run(run_id, directory=directory, status="failed", error=str(exc))
        raise

    _finish_grade(run_id, stored.model, saved, directory)
    return saved


def _tasks_from_dataset(
    dataset: str | Path,
    catalog: Mapping[str, Rubric],
    directory: Path,
) -> tuple[list[Task], str]:
    """Load tasks from a file or from prompts stored in SQLite.

    The second value is what a run stores so a later grade can find the same
    tasks. ``sqlite`` is kept as that word instead of a filesystem path.
    """
    if str(dataset) == SQLITE_DATASET:
        return load_stored_prompts(directory, catalog), SQLITE_DATASET
    path = Path(dataset)
    if not path.exists():
        raise ValueError(f"dataset not found: {path}")
    return load_prompts(path, catalog), str(path.resolve())


def _answer_pending(
    pending: list[tuple[Task, int]],
    provider: Provider,
    run_id: str,
    *,
    retries: int,
    timeout: float,
    workers: int,
    total: int,
    directory: Path,
    on_progress: Progress | None,
) -> None:
    if not pending:
        return
    if workers == 1:
        for task, attempt in pending:
            _store_generated(
                _generate_answer(task, attempt, provider, retries=retries, timeout=timeout),
                run_id,
                total=total,
                directory=directory,
                on_progress=on_progress,
            )
        return

    write_lock = Lock()
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [
        executor.submit(
            _generate_answer,
            task,
            attempt,
            provider,
            retries=retries,
            timeout=timeout,
        )
        for task, attempt in pending
    ]
    try:
        for future in as_completed(futures):
            generated = future.result()
            with write_lock:
                _store_generated(
                    generated,
                    run_id,
                    total=total,
                    directory=directory,
                    on_progress=on_progress,
                )
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _generate_answer(
    task: Task,
    attempt: int,
    provider: Provider,
    *,
    retries: int,
    timeout: float,
) -> tuple[str, int, str, float, str | None]:
    started = time.perf_counter()
    try:
        output = _generate_with_retries(
            provider,
            task.prompt,
            task.id,
            retries=retries,
            timeout=timeout,
        )
        error = None
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - one attempt must not abort the run
        output = ""
        error = str(exc)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return task.id, attempt, output, elapsed_ms, error


def _store_generated(
    generated: tuple[str, int, str, float, str | None],
    run_id: str,
    *,
    total: int,
    directory: Path,
    on_progress: Progress | None,
) -> None:
    task_id, attempt, output, latency_ms, error = generated
    save_answer(
        run_id,
        task_id,
        attempt,
        model=_answer_model(run_id, directory),
        output=output,
        latency_ms=latency_ms,
        error=error,
        directory=directory,
    )
    if on_progress is None:
        return
    done = len(answer_keys(run_id, directory=directory))
    seconds = latency_ms / 1000
    on_progress(f"[{done}/{total}] {task_id} attempt {attempt}  {seconds:.1f}s")


def _answer_model(run_id: str, directory: Path) -> str:
    stored = load_run(run_id, directory=directory)
    if stored is None or not stored.model:
        return ""
    return stored.model


def _needs_judge(tasks: list[Task], answers: list[StoredAnswer]) -> bool:
    answered = {answer.task_id for answer in answers}
    return any(
        (task.rubric_id or task.scorer == "llm_judge") and task.id in answered for task in tasks
    )


def _answers_for(answers: list[StoredAnswer], task_id: str) -> list[StoredAnswer]:
    return sorted(
        (answer for answer in answers if answer.task_id == task_id),
        key=lambda answer: answer.attempt,
    )


def _grade_answer(
    task: Task,
    answer: StoredAnswer,
    run_id: str,
    judge: Provider | None,
    rubrics: Mapping[str, Rubric],
    *,
    grader_context: str = "with_prompt",
) -> tuple[TaskResult, list[CriterionMark]]:
    if answer.error:
        output = answer.output
        scored = ScoreResult(score=0.0, passed=False, reason=f"provider failed: {answer.error}")
    else:
        output = answer.output
        scored = _score_output(task, output, judge, rubrics, grader_context=grader_context)
    rubric = _applied_rubric(task, rubrics)
    result = TaskResult(
        run_id=run_id,
        task_id=task.id,
        category=task.category,
        model=answer.model,
        output=output,
        score=scored.score,
        passed=scored.passed,
        reason=scored.reason,
        latency_ms=answer.latency_ms,
        attempt=answer.attempt,
        failure_type=failure_for(scored, output),
        rubric_id=task.rubric_id,
        rubric_version=None if rubric is None else rubric.version,
        rubric_hash=None if rubric is None else rubric.content_hash,
    )
    return result, list(scored.criteria)


def _applied_rubric(task: Task, rubrics: Mapping[str, Rubric]) -> Rubric | None:
    try:
        return applied_rubric(task, rubrics)
    except ValueError:
        if not task.rubric_id:
            return None
        return rubrics.get(task.rubric_id)


def _save_marks(
    result: TaskResult,
    judge: str,
    marks: list[CriterionMark],
    directory: Path,
) -> None:
    if not result.rubric_id or result.rubric_version is None or not result.rubric_hash:
        return
    replace_criterion_grades(
        result.run_id,
        result.task_id,
        result.attempt,
        judge,
        rubric_id=result.rubric_id,
        rubric_version=result.rubric_version,
        rubric_hash=result.rubric_hash,
        marks=marks,
        directory=directory,
    )


def _judge_label(judge: Provider | None, judge_model: str | None) -> str:
    if isinstance(judge_model, str) and judge_model.strip():
        return judge_model.strip()
    if judge is None:
        return ""
    model = getattr(judge, "model", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    return judge.name


def _finish_grade(
    run_id: str,
    model: str | None,
    results: list[TaskResult],
    directory: Path,
) -> None:
    used = {
        (result.rubric_id, result.rubric_version, result.rubric_hash)
        for result in results
        if result.rubric_id
    }
    rubric_id = rubric_version = rubric_hash = None
    if len(used) == 1:
        rubric_id, rubric_version, rubric_hash = next(iter(used))
    progress = run_progress(run_id, directory=directory)
    status = "graded"
    if progress.answers_expected and progress.answers_done < progress.answers_expected:
        status = "interrupted"
    upsert_run(
        run_id,
        directory=directory,
        status=status,
        model=model,
        rubric_id=rubric_id,
        rubric_version=rubric_version,
        rubric_hash=rubric_hash,
        error=None,
    )
