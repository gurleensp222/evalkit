"""Save and load run results in a SQLite database."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evalkit.models import CriterionMark, TaskResult

DEFAULT_RUNS_DIR = Path("runs")
DATABASE_NAME = "evalkit.db"
RUN_STATUSES = ("answering", "answered", "grading", "graded", "interrupted", "failed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    model TEXT,
    rubric_id TEXT,
    rubric_version INTEGER,
    rubric_hash TEXT,
    attempts INTEGER,
    status TEXT,
    dataset TEXT,
    provider TEXT,
    temperature REAL,
    error TEXT,
    task_count INTEGER
);

CREATE TABLE IF NOT EXISTS answers (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    model TEXT NOT NULL,
    output TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, task_id, attempt),
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

CREATE TABLE IF NOT EXISTS grades (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    judge_model TEXT NOT NULL,
    rubric_id TEXT NOT NULL,
    rubric_version INTEGER NOT NULL,
    rubric_hash TEXT NOT NULL,
    criterion_id TEXT NOT NULL,
    met INTEGER NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (run_id, task_id, attempt, judge_model, criterion_id),
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    category TEXT NOT NULL,
    model TEXT NOT NULL,
    output TEXT NOT NULL,
    score REAL NOT NULL,
    passed INTEGER NOT NULL,
    reason TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    failure_type TEXT,
    unstable INTEGER NOT NULL,
    output_varies INTEGER NOT NULL DEFAULT 0,
    rubric_id TEXT,
    rubric_version INTEGER,
    rubric_hash TEXT,
    judge TEXT NOT NULL DEFAULT '',
    graded_at TEXT,
    PRIMARY KEY (run_id, task_id, attempt, judge),
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);
"""

_INSERT = """
INSERT INTO results (
    run_id, task_id, attempt, category, model, output, score, passed,
    reason, latency_ms, failure_type, unstable, output_varies,
    rubric_id, rubric_version, rubric_hash, judge, graded_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_RUN_FIELDS = (
    "model",
    "rubric_id",
    "rubric_version",
    "rubric_hash",
    "attempts",
    "status",
    "dataset",
    "provider",
    "temperature",
    "error",
    "task_count",
)


def save_results(
    run_id: str,
    results: Sequence[TaskResult],
    *,
    directory: Path = DEFAULT_RUNS_DIR,
    attempts: int | None = None,
) -> Path:
    """Replace ``run_id`` in ``runs/evalkit.db`` and return the database path."""
    _validate_run_id(run_id)
    with _connect(directory) as connection:
        recorded = results[0].model if results else None
        rubric_id, rubric_version, rubric_hash = _run_rubric(results)
        recorded_attempts = attempts
        if recorded_attempts is None and results:
            recorded_attempts = max(result.attempt for result in results)
        connection.execute(
            """
            INSERT INTO runs (
                run_id, model, rubric_id, rubric_version, rubric_hash, attempts, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'graded')
            ON CONFLICT(run_id) DO UPDATE SET
                model = excluded.model,
                rubric_id = excluded.rubric_id,
                rubric_version = excluded.rubric_version,
                rubric_hash = excluded.rubric_hash,
                attempts = excluded.attempts,
                status = excluded.status
            """,
            (run_id, recorded, rubric_id, rubric_version, rubric_hash, recorded_attempts),
        )
        connection.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
        connection.executemany(_INSERT, [_row(result) for result in results])
    return directory / DATABASE_NAME


def load_results(
    run_id: str,
    *,
    directory: Path = DEFAULT_RUNS_DIR,
    judge: str | None = None,
) -> list[TaskResult]:
    """Read one grading of ``run_id``.

    ``judge`` selects that grading. When it is omitted, the newest grading is
    returned so a report still shows a single set of scores.
    """
    _validate_run_id(run_id)
    database = directory / DATABASE_NAME
    if not database.is_file():
        raise FileNotFoundError(database)
    with _connect(directory) as connection:
        found = connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if found is None:
            raise FileNotFoundError(run_id)
        selected = _stored_judge(judge) if judge is not None else _latest_judge(connection, run_id)
        if selected is None:
            return []
        rows = connection.execute(
            """
            SELECT run_id, task_id, attempt, category, model, output, score, passed,
                   reason, latency_ms, failure_type, unstable, output_varies,
                   rubric_id, rubric_version, rubric_hash, judge
            FROM results
            WHERE run_id = ? AND judge = ?
            ORDER BY rowid
            """,
            (run_id, selected),
        ).fetchall()
    return [_result(row) for row in rows]


@dataclass(frozen=True)
class StoredRun:
    """One row from the runs table."""

    run_id: str
    model: str | None
    status: str | None
    attempts: int | None
    dataset: str | None
    provider: str | None
    temperature: float | None
    rubric_id: str | None
    rubric_version: int | None
    rubric_hash: str | None
    error: str | None
    task_count: int | None


@dataclass(frozen=True)
class StoredAnswer:
    """One generated attempt, saved before grading."""

    run_id: str
    task_id: str
    attempt: int
    model: str
    output: str
    latency_ms: float
    error: str | None
    created_at: str


@dataclass(frozen=True)
class StoredCriterionGrade:
    """Whether one criterion was met for one attempt."""

    run_id: str
    task_id: str
    attempt: int
    judge_model: str
    rubric_id: str
    rubric_version: int
    rubric_hash: str
    criterion_id: str
    met: bool
    reason: str


@dataclass(frozen=True)
class RunProgress:
    """How far a run has gotten through answering and grading."""

    status: str | None
    answers_done: int
    answers_expected: int
    grades_done: int
    grades_expected: int


def load_run(run_id: str, *, directory: Path = DEFAULT_RUNS_DIR) -> StoredRun | None:
    """Return the run row, or None when it has not been created."""
    _validate_run_id(run_id)
    database = directory / DATABASE_NAME
    if not database.is_file():
        return None
    with _connect(directory) as connection:
        row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return _stored_run(row)


def upsert_run(run_id: str, *, directory: Path = DEFAULT_RUNS_DIR, **fields: object) -> None:
    """Insert a run or update the given columns."""
    _validate_run_id(run_id)
    unknown = set(fields) - set(_RUN_FIELDS)
    if unknown:
        raise ValueError(f"unknown run fields: {', '.join(sorted(unknown))}")
    with _connect(directory) as connection:
        found = connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if found is None:
            columns = ["run_id", *fields.keys()]
            marks = ", ".join("?" for _ in columns)
            connection.execute(
                f"INSERT INTO runs ({', '.join(columns)}) VALUES ({marks})",
                (run_id, *fields.values()),
            )
            return
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name in fields)
        connection.execute(
            f"UPDATE runs SET {assignments} WHERE run_id = ?",
            (*fields.values(), run_id),
        )


def save_answer(
    run_id: str,
    task_id: str,
    attempt: int,
    *,
    model: str,
    output: str,
    latency_ms: float,
    error: str | None,
    directory: Path = DEFAULT_RUNS_DIR,
) -> None:
    """Insert one answer and commit it before the next attempt starts."""
    _validate_run_id(run_id)
    created_at = _now()
    with _connect(directory) as connection:
        connection.execute(
            """
            INSERT INTO answers (
                run_id, task_id, attempt, model, output, latency_ms, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, task_id, attempt) DO UPDATE SET
                model = excluded.model,
                output = excluded.output,
                latency_ms = excluded.latency_ms,
                error = excluded.error,
                created_at = excluded.created_at
            """,
            (run_id, task_id, attempt, model, output, latency_ms, error, created_at),
        )


def load_answers(run_id: str, *, directory: Path = DEFAULT_RUNS_DIR) -> list[StoredAnswer]:
    """Return saved answers in the order they were written."""
    _validate_run_id(run_id)
    with _connect(directory) as connection:
        rows = connection.execute(
            """
            SELECT run_id, task_id, attempt, model, output, latency_ms, error, created_at
            FROM answers
            WHERE run_id = ?
            ORDER BY rowid
            """,
            (run_id,),
        ).fetchall()
    return [
        StoredAnswer(
            run_id=str(row["run_id"]),
            task_id=str(row["task_id"]),
            attempt=int(row["attempt"]),
            model=str(row["model"]),
            output=str(row["output"]),
            latency_ms=float(row["latency_ms"]),
            error=None if row["error"] is None else str(row["error"]),
            created_at=str(row["created_at"]),
        )
        for row in rows
    ]


def answer_keys(run_id: str, *, directory: Path = DEFAULT_RUNS_DIR) -> set[tuple[str, int]]:
    """Return the task id and attempt pairs already stored."""
    _validate_run_id(run_id)
    database = directory / DATABASE_NAME
    if not database.is_file():
        return set()
    with _connect(directory) as connection:
        rows = connection.execute(
            "SELECT task_id, attempt FROM answers WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    return {(str(row["task_id"]), int(row["attempt"])) for row in rows}


def save_grade(
    result: TaskResult,
    *,
    judge: str,
    directory: Path = DEFAULT_RUNS_DIR,
) -> None:
    """Insert one graded attempt without removing other judges."""
    _validate_run_id(result.run_id)
    stored = result.model_copy(update={"judge": judge or None})
    with _connect(directory) as connection:
        connection.execute(
            """
            INSERT INTO results (
                run_id, task_id, attempt, category, model, output, score, passed,
                reason, latency_ms, failure_type, unstable, output_varies,
                rubric_id, rubric_version, rubric_hash, judge, graded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, task_id, attempt, judge) DO UPDATE SET
                category = excluded.category,
                model = excluded.model,
                output = excluded.output,
                score = excluded.score,
                passed = excluded.passed,
                reason = excluded.reason,
                latency_ms = excluded.latency_ms,
                failure_type = excluded.failure_type,
                unstable = excluded.unstable,
                output_varies = excluded.output_varies,
                rubric_id = excluded.rubric_id,
                rubric_version = excluded.rubric_version,
                rubric_hash = excluded.rubric_hash,
                graded_at = excluded.graded_at
            """,
            _row(stored),
        )


def replace_criterion_grades(
    run_id: str,
    task_id: str,
    attempt: int,
    judge_model: str,
    *,
    rubric_id: str,
    rubric_version: int,
    rubric_hash: str,
    marks: Sequence[CriterionMark],
    directory: Path = DEFAULT_RUNS_DIR,
) -> None:
    """Replace one judge's criterion rows for a single attempt."""
    _validate_run_id(run_id)
    with _connect(directory) as connection:
        connection.execute(
            """
            DELETE FROM grades
            WHERE run_id = ? AND task_id = ? AND attempt = ? AND judge_model = ?
            """,
            (run_id, task_id, attempt, judge_model),
        )
        connection.executemany(
            """
            INSERT INTO grades (
                run_id, task_id, attempt, judge_model, rubric_id, rubric_version,
                rubric_hash, criterion_id, met, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    task_id,
                    attempt,
                    judge_model,
                    rubric_id,
                    rubric_version,
                    rubric_hash,
                    mark.criterion_id,
                    int(mark.met),
                    mark.reason,
                )
                for mark in marks
            ],
        )


def load_criterion_grades(
    run_id: str,
    *,
    directory: Path = DEFAULT_RUNS_DIR,
    judge: str = "",
) -> list[StoredCriterionGrade]:
    """Read criterion rows for one judge. An empty judge matches unlabeled rows."""
    _validate_run_id(run_id)
    database = directory / DATABASE_NAME
    if not database.is_file():
        raise FileNotFoundError(run_id)
    with _connect(directory) as connection:
        found = connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if found is None:
            raise FileNotFoundError(run_id)
        rows = connection.execute(
            """
            SELECT run_id, task_id, attempt, judge_model, rubric_id, rubric_version,
                   rubric_hash, criterion_id, met, reason
            FROM grades
            WHERE run_id = ? AND judge_model = ?
            ORDER BY rowid
            """,
            (run_id, judge),
        ).fetchall()
    return [
        StoredCriterionGrade(
            run_id=str(row["run_id"]),
            task_id=str(row["task_id"]),
            attempt=int(row["attempt"]),
            judge_model=str(row["judge_model"]),
            rubric_id=str(row["rubric_id"]),
            rubric_version=int(row["rubric_version"]),
            rubric_hash=str(row["rubric_hash"]),
            criterion_id=str(row["criterion_id"]),
            met=bool(row["met"]),
            reason=str(row["reason"]),
        )
        for row in rows
    ]


def list_judges(run_id: str, *, directory: Path = DEFAULT_RUNS_DIR) -> list[str]:
    """Return judge labels stored for a run, newest grading last."""
    _validate_run_id(run_id)
    database = directory / DATABASE_NAME
    if not database.is_file():
        return []
    with _connect(directory) as connection:
        rows = connection.execute(
            """
            SELECT judge
            FROM results
            WHERE run_id = ?
            GROUP BY judge
            ORDER BY MAX(COALESCE(graded_at, '')) ASC, MAX(rowid) ASC
            """,
            (run_id,),
        ).fetchall()
    return [str(row["judge"]) for row in rows]


def run_progress(run_id: str, *, directory: Path = DEFAULT_RUNS_DIR) -> RunProgress:
    """Count saved answers and the newest grading."""
    _validate_run_id(run_id)
    run = load_run(run_id, directory=directory)
    if run is None:
        raise FileNotFoundError(run_id)
    with _connect(directory) as connection:
        answers_done = _count(connection, "answers", run_id)
        judge = _latest_judge(connection, run_id)
        grades_done = 0 if judge is None else _count(connection, "results", run_id, judge)
    expected = (run.task_count or 0) * (run.attempts or 0)
    if expected == 0:
        expected = answers_done
    return RunProgress(
        status=run.status,
        answers_done=answers_done,
        answers_expected=expected,
        grades_done=grades_done,
        grades_expected=answers_done,
    )


def list_run_ids(*, directory: Path = DEFAULT_RUNS_DIR) -> list[str]:
    """Return saved run ids in alphabetical order."""
    database = directory / DATABASE_NAME
    if not database.is_file():
        return []
    with _connect(directory) as connection:
        rows = connection.execute("SELECT run_id FROM runs ORDER BY run_id").fetchall()
    return [str(row["run_id"]) for row in rows]


@contextmanager
def _connect(directory: Path) -> Iterator[sqlite3.Connection]:
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / DATABASE_NAME)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(_SCHEMA)
        _ensure_output_varies(connection)
        _ensure_runs_model(connection)
        _ensure_rubric_columns(connection)
        _ensure_runs_attempts(connection)
        _ensure_run_progress_columns(connection)
        _ensure_results_judge(connection)
        yield connection
        connection.commit()
    finally:
        connection.close()


def _row(result: TaskResult) -> tuple[object, ...]:
    return (
        result.run_id,
        result.task_id,
        result.attempt,
        result.category,
        result.model,
        result.output,
        result.score,
        int(result.passed),
        result.reason,
        result.latency_ms,
        result.failure_type,
        int(result.unstable),
        int(result.output_varies),
        result.rubric_id,
        result.rubric_version,
        result.rubric_hash,
        result.judge or "",
        _now(),
    )


def _result(row: sqlite3.Row) -> TaskResult:
    return TaskResult(
        run_id=row["run_id"],
        task_id=row["task_id"],
        attempt=row["attempt"],
        category=row["category"],
        model=row["model"],
        output=row["output"],
        score=row["score"],
        passed=bool(row["passed"]),
        reason=row["reason"],
        latency_ms=row["latency_ms"],
        failure_type=row["failure_type"],
        unstable=bool(row["unstable"]),
        output_varies=bool(row["output_varies"]),
        rubric_id=None if row["rubric_id"] is None else str(row["rubric_id"]),
        rubric_version=None if row["rubric_version"] is None else int(row["rubric_version"]),
        rubric_hash=None if row["rubric_hash"] is None else str(row["rubric_hash"]),
        judge=None if not row["judge"] else str(row["judge"]),
    )


def _ensure_output_varies(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(results)")}
    if columns and "output_varies" not in columns:
        connection.execute(
            "ALTER TABLE results ADD COLUMN output_varies INTEGER NOT NULL DEFAULT 0"
        )


def _ensure_runs_model(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(runs)")}
    if columns and "model" not in columns:
        connection.execute("ALTER TABLE runs ADD COLUMN model TEXT")
    connection.execute(
        """
        UPDATE runs
        SET model = (
            SELECT results.model FROM results
            WHERE results.run_id = runs.run_id
            LIMIT 1
        )
        WHERE model IS NULL
          AND EXISTS (SELECT 1 FROM results WHERE results.run_id = runs.run_id)
        """
    )


def _ensure_runs_attempts(connection: sqlite3.Connection) -> None:
    _add_column(connection, "runs", "attempts", "INTEGER")
    connection.execute(
        """
        UPDATE runs
        SET attempts = (
            SELECT MAX(results.attempt) FROM results
            WHERE results.run_id = runs.run_id
        )
        WHERE attempts IS NULL
          AND EXISTS (SELECT 1 FROM results WHERE results.run_id = runs.run_id)
        """
    )


def _ensure_run_progress_columns(connection: sqlite3.Connection) -> None:
    _add_column(connection, "runs", "status", "TEXT")
    _add_column(connection, "runs", "dataset", "TEXT")
    _add_column(connection, "runs", "provider", "TEXT")
    _add_column(connection, "runs", "temperature", "REAL")
    _add_column(connection, "runs", "error", "TEXT")
    _add_column(connection, "runs", "task_count", "INTEGER")
    connection.execute(
        """
        UPDATE runs
        SET status = 'graded'
        WHERE status IS NULL
          AND EXISTS (SELECT 1 FROM results WHERE results.run_id = runs.run_id)
        """
    )


def _ensure_results_judge(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(results)")}
    if not columns or "judge" in columns:
        if columns and "graded_at" not in columns:
            _add_column(connection, "results", "graded_at", "TEXT")
        return
    connection.execute("ALTER TABLE results RENAME TO results_legacy")
    connection.execute(
        """
        CREATE TABLE results (
            run_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            category TEXT NOT NULL,
            model TEXT NOT NULL,
            output TEXT NOT NULL,
            score REAL NOT NULL,
            passed INTEGER NOT NULL,
            reason TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            failure_type TEXT,
            unstable INTEGER NOT NULL,
            output_varies INTEGER NOT NULL DEFAULT 0,
            rubric_id TEXT,
            rubric_version INTEGER,
            rubric_hash TEXT,
            judge TEXT NOT NULL DEFAULT '',
            graded_at TEXT,
            PRIMARY KEY (run_id, task_id, attempt, judge),
            FOREIGN KEY (run_id) REFERENCES runs (run_id)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO results (
            run_id, task_id, attempt, category, model, output, score, passed,
            reason, latency_ms, failure_type, unstable, output_varies,
            rubric_id, rubric_version, rubric_hash, judge, graded_at
        )
        SELECT
            run_id, task_id, attempt, category, model, output, score, passed,
            reason, latency_ms, failure_type, unstable, output_varies,
            rubric_id, rubric_version, rubric_hash, '', NULL
        FROM results_legacy
        """
    )
    connection.execute("DROP TABLE results_legacy")


def _ensure_rubric_columns(connection: sqlite3.Connection) -> None:
    for table in ("results", "runs"):
        _add_column(connection, table, "rubric_id", "TEXT")
        _add_column(connection, table, "rubric_version", "INTEGER")
        _add_column(connection, table, "rubric_hash", "TEXT")


def _add_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    declaration: str,
) -> None:
    columns = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}
    if columns and column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _run_rubric(
    results: Sequence[TaskResult],
) -> tuple[str | None, int | None, str | None]:
    used = {
        (result.rubric_id, result.rubric_version, result.rubric_hash)
        for result in results
        if result.rubric_id
    }
    if len(used) == 1:
        return next(iter(used))
    return None, None, None


def _stored_run(row: sqlite3.Row) -> StoredRun:
    names = set(row.keys())

    def text(name: str) -> str | None:
        if name not in names or row[name] is None:
            return None
        return str(row[name])

    def number(name: str) -> int | None:
        if name not in names or row[name] is None:
            return None
        return int(row[name])

    temperature = None
    if "temperature" in names and row["temperature"] is not None:
        temperature = float(row["temperature"])
    return StoredRun(
        run_id=str(row["run_id"]),
        model=text("model"),
        status=text("status"),
        attempts=number("attempts"),
        dataset=text("dataset"),
        provider=text("provider"),
        temperature=temperature,
        rubric_id=text("rubric_id"),
        rubric_version=number("rubric_version"),
        rubric_hash=text("rubric_hash"),
        error=text("error"),
        task_count=number("task_count"),
    )


def _latest_judge(connection: sqlite3.Connection, run_id: str) -> str | None:
    row = connection.execute(
        """
        SELECT judge
        FROM results
        WHERE run_id = ?
        GROUP BY judge
        ORDER BY MAX(COALESCE(graded_at, '')) DESC, MAX(rowid) DESC
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return str(row["judge"])


def _stored_judge(judge: str | None) -> str:
    return judge or ""


def _count(
    connection: sqlite3.Connection,
    table: str,
    run_id: str,
    judge: str | None = None,
) -> int:
    if table not in {"answers", "results"}:
        raise ValueError(f"unknown table: {table}")
    if judge is None:
        row = connection.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    else:
        row = connection.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE run_id = ? AND judge = ?",
            (run_id, judge),
        ).fetchone()
    return int(row["n"])


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or any(sep in run_id for sep in ("/", "\\", "\0")):
        raise ValueError(f"invalid run id: {run_id!r}")
