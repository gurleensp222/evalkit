from pathlib import Path

import pandas as pd
import pytest

from evalkit.analysis import by_category, compare, failures, load_runs, summary
from evalkit.models import TaskResult
from evalkit.storage import save_results


def _result(run_id: str, task_id: str, model: str, passed: bool) -> TaskResult:
    return TaskResult(
        run_id=run_id,
        task_id=task_id,
        category="math",
        model=model,
        output="4" if passed else "no",
        score=1.0 if passed else 0.0,
        passed=passed,
        reason="ok" if passed else "wrong",
        latency_ms=10,
    )


def test_load_runs_stacks_saved_jsonl_files(tmp_path: Path) -> None:
    save_results("run-a", [_result("run-a", "t1", "alpha", True)], directory=tmp_path)
    save_results("run-b", [_result("run-b", "t1", "beta", False)], directory=tmp_path)

    loaded = load_runs(["run-a", "run-b"], directory=tmp_path)
    assert list(loaded["run_id"]) == ["run-a", "run-b"]
    assert list(loaded["model"]) == ["alpha", "beta"]
    assert load_runs("run-a", directory=tmp_path)["task_id"].tolist() == ["t1"]


def test_summary_pass_rate_score_and_latency() -> None:
    df = pd.DataFrame(
        [
            {"model": "alpha", "passed": True, "score": 1.0, "latency_ms": 10},
            {"model": "alpha", "passed": False, "score": 0.0, "latency_ms": 20},
            {"model": "alpha", "passed": True, "score": 1.0, "latency_ms": 30},
            {"model": "alpha", "passed": False, "score": 0.0, "latency_ms": 40},
            {"model": "beta", "passed": True, "score": 0.5, "latency_ms": 100},
            {"model": "beta", "passed": True, "score": 1.0, "latency_ms": 300},
        ]
    )
    stats = summary(df)

    assert stats.loc["alpha", "pass_rate"] == pytest.approx(0.5)
    assert stats.loc["alpha", "mean_score"] == pytest.approx(0.5)
    assert stats.loc["alpha", "mean_latency_ms"] == pytest.approx(25)
    # Linear p95 of [10, 20, 30, 40] sits 0.85 of the way from 30 to 40.
    assert stats.loc["alpha", "p95_latency_ms"] == pytest.approx(38.5)
    assert stats.loc["beta", "pass_rate"] == pytest.approx(1)
    assert stats.loc["beta", "mean_score"] == pytest.approx(0.75)
    assert stats.loc["beta", "mean_latency_ms"] == pytest.approx(200)
    # Linear p95 of [100, 300] is 100 + 0.95 * 200.
    assert stats.loc["beta", "p95_latency_ms"] == pytest.approx(290)


def test_summary_of_an_empty_frame_has_the_metric_columns() -> None:
    stats = summary(pd.DataFrame(columns=["model", "passed", "score", "latency_ms"]))
    assert list(stats.columns) == [
        "pass_rate",
        "mean_score",
        "mean_latency_ms",
        "p95_latency_ms",
    ]
    assert stats.empty


def test_by_category_pivots_pass_rate() -> None:
    df = pd.DataFrame(
        [
            {"category": "math", "model": "alpha", "passed": True},
            {"category": "math", "model": "alpha", "passed": False},
            {"category": "math", "model": "beta", "passed": True},
            {"category": "code", "model": "alpha", "passed": True},
        ]
    )
    table = by_category(df)

    assert table.loc["math", "alpha"] == pytest.approx(0.5)
    assert table.loc["math", "beta"] == pytest.approx(1)
    assert table.loc["code", "alpha"] == pytest.approx(1)
    assert pd.isna(table.loc["code", "beta"])
    assert list(table.index) == ["code", "math"]
    assert list(table.columns) == ["alpha", "beta"]


def test_failures_keeps_failed_reasons_sorted_by_category() -> None:
    df = pd.DataFrame(
        [
            {
                "category": "math",
                "task_id": "t1",
                "model": "alpha",
                "run_id": "r",
                "score": 1.0,
                "passed": True,
                "reason": "should be hidden",
            },
            {
                "category": "math",
                "task_id": "t2",
                "model": "beta",
                "run_id": "r",
                "score": 0.0,
                "passed": False,
                "reason": "wrong city",
            },
            {
                "category": "code",
                "task_id": "t3",
                "model": "alpha",
                "run_id": "r",
                "score": 0.0,
                "passed": False,
                "reason": "assert failed",
            },
        ]
    )
    failed = failures(df)

    assert failed["reason"].tolist() == ["assert failed", "wrong city"]
    assert failed["category"].tolist() == ["code", "math"]
    assert "should be hidden" not in failed["reason"].tolist()


def test_compare_interval_is_exact_when_every_gap_is_identical() -> None:
    df = pd.DataFrame(
        [
            {"task_id": "t1", "model": "alpha", "passed": True},
            {"task_id": "t1", "model": "beta", "passed": False},
            {"task_id": "t2", "model": "alpha", "passed": True},
            {"task_id": "t2", "model": "beta", "passed": False},
            {"task_id": "only-alpha", "model": "alpha", "passed": False},
        ]
    )
    result = compare(df, "alpha", "beta")

    assert result["n_tasks"] == 2
    assert result["pass_rate_a"] == pytest.approx(1)
    assert result["pass_rate_b"] == pytest.approx(0)
    assert result["difference"] == pytest.approx(1)
    assert result["ci_low"] == pytest.approx(1)
    assert result["ci_high"] == pytest.approx(1)
    assert result["excludes_zero"] is True


def test_compare_interval_includes_zero_when_models_tie() -> None:
    df = pd.DataFrame(
        [
            {"task_id": "t1", "model": "alpha", "passed": True},
            {"task_id": "t1", "model": "beta", "passed": True},
            {"task_id": "t2", "model": "alpha", "passed": False},
            {"task_id": "t2", "model": "beta", "passed": False},
        ]
    )
    result = compare(df, "alpha", "beta")

    assert result["difference"] == pytest.approx(0)
    assert result["ci_low"] == pytest.approx(0)
    assert result["ci_high"] == pytest.approx(0)
    assert result["excludes_zero"] is False


def test_compare_mixed_gaps_cover_zero() -> None:
    df = pd.DataFrame(
        [
            {"task_id": "t1", "model": "alpha", "passed": True},
            {"task_id": "t1", "model": "beta", "passed": False},
            {"task_id": "t2", "model": "alpha", "passed": False},
            {"task_id": "t2", "model": "beta", "passed": True},
        ]
    )
    result = compare(df, "alpha", "beta")

    assert result["difference"] == pytest.approx(0)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["excludes_zero"] is False


def test_compare_averages_duplicate_rows_for_the_same_task() -> None:
    df = pd.DataFrame(
        [
            {"task_id": "t1", "model": "alpha", "passed": True},
            {"task_id": "t1", "model": "alpha", "passed": False},
            {"task_id": "t1", "model": "beta", "passed": True},
        ]
    )
    result = compare(df, "alpha", "beta")

    assert result["n_tasks"] == 1
    assert result["pass_rate_a"] == pytest.approx(0.5)
    assert result["pass_rate_b"] == pytest.approx(1)
    assert result["difference"] == pytest.approx(-0.5)
    assert result["ci_low"] == pytest.approx(-0.5)
    assert result["ci_high"] == pytest.approx(-0.5)
    assert result["excludes_zero"] is True


def test_compare_requires_tasks_answered_by_both_models() -> None:
    df = pd.DataFrame(
        [
            {"task_id": "t1", "model": "alpha", "passed": True},
            {"task_id": "t2", "model": "beta", "passed": True},
        ]
    )
    with pytest.raises(ValueError, match="no shared tasks"):
        compare(df, "alpha", "beta")
