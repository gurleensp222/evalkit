"""Aggregate saved runs with pandas."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from evalkit.storage import DEFAULT_RUNS_DIR, load_results

RESULT_COLUMNS = [
    "run_id",
    "task_id",
    "attempt",
    "category",
    "model",
    "output",
    "score",
    "passed",
    "reason",
    "latency_ms",
    "failure_type",
    "unstable",
]
FAILURE_COLUMNS = ["category", "task_id", "attempt", "model", "run_id", "score", "reason", "failure_type"]
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0


class Comparison(TypedDict):
    """Paired pass-rate difference of model A minus model B."""

    model_a: str
    model_b: str
    n_tasks: int
    pass_rate_a: float
    pass_rate_b: float
    difference: float
    ci_low: float
    ci_high: float
    excludes_zero: bool


def load_runs(
    run_ids: str | Sequence[str],
    *,
    directory: Path = DEFAULT_RUNS_DIR,
) -> pd.DataFrame:
    """Load one run id, or several, into a single DataFrame."""
    ids = [run_ids] if isinstance(run_ids, str) else list(run_ids)
    frames: list[pd.DataFrame] = []
    for run_id in ids:
        results = load_results(run_id, directory=directory)
        if not results:
            continue
        frames.append(pd.DataFrame([result.model_dump() for result in results]))
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pass rate, mean score, and latency by model."""
    columns = ["pass_rate", "mean_score", "mean_latency_ms", "p95_latency_ms"]
    if df.empty:
        empty = pd.DataFrame(columns=columns)
        empty.index.name = "model"
        return empty

    grouped = df.groupby("model", sort=True)
    frame = pd.DataFrame(
        {
            "pass_rate": grouped["passed"].mean(),
            "mean_score": grouped["score"].mean(),
            "mean_latency_ms": grouped["latency_ms"].mean(),
            "p95_latency_ms": grouped["latency_ms"].quantile(0.95),
        }
    )
    frame.index.name = "model"
    return frame


def by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Pass rate with categories as rows and models as columns."""
    if df.empty:
        empty = pd.DataFrame()
        empty.index.name = "category"
        empty.columns.name = "model"
        return empty

    table = df.pivot_table(
        values="passed",
        index="category",
        columns="model",
        aggfunc="mean",
    )
    return table.sort_index().sort_index(axis=1)


def failures(df: pd.DataFrame) -> pd.DataFrame:
    """Failed tasks and their reasons, sorted by category."""
    columns = [column for column in FAILURE_COLUMNS if column in df.columns]
    failed = df.loc[~df["passed"].astype(bool), columns]
    sort_by = [
        column
        for column in ("category", "task_id", "attempt", "model")
        if column in failed.columns
    ]
    if failed.empty or not sort_by:
        return failed.reset_index(drop=True)
    return failed.sort_values(sort_by, kind="mergesort").reset_index(drop=True)


def compare(df: pd.DataFrame, model_a: str, model_b: str) -> Comparison:
    """Bootstrap the pass-rate gap for tasks both models answered.

    The difference is model A's pass rate minus model B's. The interval is the
    2.5 and 97.5 percentiles of 1000 task resamples from a fixed seed.
    """
    paired = _paired_passes(df, model_a, model_b)
    deltas = (paired[model_a] - paired[model_b]).to_numpy(dtype=float)
    observed = float(deltas.mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(deltas), size=(BOOTSTRAP_RESAMPLES, len(deltas)))
    means = deltas[draws].mean(axis=1)
    ci_low, ci_high = (float(bound) for bound in np.quantile(means, [0.025, 0.975]))
    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_tasks": len(deltas),
        "pass_rate_a": float(paired[model_a].mean()),
        "pass_rate_b": float(paired[model_b].mean()),
        "difference": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "excludes_zero": bool(ci_low > 0 or ci_high < 0),
    }


def _paired_passes(df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """One pass rate per task for each model. Tasks missing either model are dropped."""
    subset = df.loc[df["model"].isin([model_a, model_b]), ["task_id", "model", "passed"]]
    if subset.empty:
        raise ValueError(f"no shared tasks for {model_a!r} and {model_b!r}")

    wide = subset.groupby(["task_id", "model"], sort=True)["passed"].mean().unstack("model")
    missing = [name for name in (model_a, model_b) if name not in wide.columns]
    if missing:
        raise ValueError(f"no shared tasks for {model_a!r} and {model_b!r}")

    paired = wide[[model_a, model_b]].dropna()
    if paired.empty:
        raise ValueError(f"no shared tasks for {model_a!r} and {model_b!r}")
    return paired
