"""Save and load run results in a SQLite database."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from evalkit.models import TaskResult

DEFAULT_RUNS_DIR = Path("runs")
DATABASE_NAME = "evalkit.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    model TEXT,
    rubric_id TEXT,
    rubric_version INTEGER,
    rubric_hash TEXT,
    attempts INTEGER
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
    PRIMARY KEY (run_id, task_id, attempt),
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);
"""

_INSERT = """
INSERT INTO results (
    run_id, task_id, attempt, category, model, output, score, passed,
    reason, latency_ms, failure_type, unstable, output_varies,
    rubric_id, rubric_version, rubric_hash
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


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
                run_id, model, rubric_id, rubric_version, rubric_hash, attempts
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                model = excluded.model,
                rubric_id = excluded.rubric_id,
                rubric_version = excluded.rubric_version,
                rubric_hash = excluded.rubric_hash,
                attempts = excluded.attempts
            """,
            (run_id, recorded, rubric_id, rubric_version, rubric_hash, recorded_attempts),
        )
        connection.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
        connection.executemany(_INSERT, [_row(result) for result in results])
    return directory / DATABASE_NAME


def load_results(run_id: str, *, directory: Path = DEFAULT_RUNS_DIR) -> list[TaskResult]:
    """Read every attempt stored for ``run_id``."""
    _validate_run_id(run_id)
    database = directory / DATABASE_NAME
    if not database.is_file():
        raise FileNotFoundError(database)
    with _connect(directory) as connection:
        found = connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if found is None:
            raise FileNotFoundError(run_id)
        rows = connection.execute(
            """
            SELECT run_id, task_id, attempt, category, model, output, score, passed,
                   reason, latency_ms, failure_type, unstable, output_varies,
                   rubric_id, rubric_version, rubric_hash
            FROM results
            WHERE run_id = ?
            ORDER BY rowid
            """,
            (run_id,),
        ).fetchall()
    return [_result(row) for row in rows]


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


def _validate_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or any(sep in run_id for sep in ("/", "\\", "\0")):
        raise ValueError(f"invalid run id: {run_id!r}")
