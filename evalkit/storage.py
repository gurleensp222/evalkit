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
    run_id TEXT PRIMARY KEY
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
    PRIMARY KEY (run_id, task_id, attempt),
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);
"""

_INSERT = """
INSERT INTO results (
    run_id, task_id, attempt, category, model, output, score, passed,
    reason, latency_ms, failure_type, unstable
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def save_results(
    run_id: str,
    results: Sequence[TaskResult],
    *,
    directory: Path = DEFAULT_RUNS_DIR,
) -> Path:
    """Replace ``run_id`` in ``runs/evalkit.db`` and return the database path."""
    _validate_run_id(run_id)
    with _connect(directory) as connection:
        connection.execute(
            "INSERT INTO runs (run_id) VALUES (?) ON CONFLICT(run_id) DO NOTHING",
            (run_id,),
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
                   reason, latency_ms, failure_type, unstable
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
    )


def _validate_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or any(sep in run_id for sep in ("/", "\\", "\0")):
        raise ValueError(f"invalid run id: {run_id!r}")
