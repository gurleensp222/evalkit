from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evalkit.api import create_app

TASKS = """\
{"id": "exact-right", "category": "math", "prompt": "2+2", "scorer": "exact_match", "expected": "4"}
{"id": "exact-wrong", "category": "knowledge", "prompt": "capital", "scorer": "exact_match", "expected": "Paris"}
"""


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text(TASKS, encoding="utf-8")
    return path


@pytest.fixture
def client(tmp_path: Path, dataset: Path) -> TestClient:
    app = create_app(runs_directory=tmp_path / "runs", default_dataset=dataset)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_tasks(client: TestClient) -> None:
    response = client.get("/tasks")
    assert response.status_code == 200
    body = response.json()
    assert [task["id"] for task in body] == ["exact-right", "exact-wrong"]
    assert body[0]["scorer"] == "exact_match"
    assert body[1]["expected"] == "Paris"


def test_mock_run_summary_and_failures(client: TestClient, dataset: Path) -> None:
    created = client.post("/runs", json={"provider": "mock", "dataset": str(dataset)})
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    assert created.json()["status"] == "running"

    listed = client.get("/runs")
    assert listed.status_code == 200
    match = next(item for item in listed.json() if item["run_id"] == run_id)
    assert match["status"] == "completed"
    assert match["provider"] == "mock"

    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == "completed"
    assert body["summary"] == [
        {
            "model": "mock",
            "pass_rate": 0.5,
            "mean_score": 0.5,
            "mean_latency_ms": body["summary"][0]["mean_latency_ms"],
            "p95_latency_ms": body["summary"][0]["p95_latency_ms"],
        }
    ]
    assert body["summary"][0]["mean_latency_ms"] >= 0
    assert body["by_category"]["math"]["mock"] == 1.0
    assert body["by_category"]["knowledge"]["mock"] == 0.0

    failed = client.get(f"/runs/{run_id}/failures")
    assert failed.status_code == 200
    reasons = failed.json()
    assert [item["task_id"] for item in reasons] == ["exact-wrong"]
    assert reasons[0]["category"] == "knowledge"
    assert "Paris" in reasons[0]["reason"]


def test_compare_two_mock_runs(client: TestClient, dataset: Path) -> None:
    first = client.post("/runs", json={"provider": "mock", "dataset": str(dataset)}).json()["run_id"]
    second = client.post("/runs", json={"provider": "mock", "dataset": str(dataset)}).json()["run_id"]

    response = client.get("/compare", params={"a": first, "b": second})
    assert response.status_code == 200
    body = response.json()
    assert body["model_a"] == first
    assert body["model_b"] == second
    assert body["n_tasks"] == 2
    assert body["pass_rate_a"] == 0.5
    assert body["pass_rate_b"] == 0.5
    assert body["difference"] == 0.0
    assert body["ci_low"] == 0.0
    assert body["ci_high"] == 0.0
    assert body["excludes_zero"] is False


def test_saved_run_is_visible_to_a_new_app(tmp_path: Path, dataset: Path) -> None:
    runs = tmp_path / "runs"
    first = TestClient(create_app(runs_directory=runs, default_dataset=dataset))
    run_id = first.post("/runs", json={"provider": "mock", "dataset": str(dataset)}).json()["run_id"]

    second = TestClient(create_app(runs_directory=runs, default_dataset=dataset))
    response = second.get(f"/runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["summary"][0]["pass_rate"] == 0.5
    assert response.json()["provider"] == "mock"


def test_unknown_runs_are_404(client: TestClient) -> None:
    missing = client.get("/runs/missing")
    assert missing.status_code == 404
    assert "unknown run" in missing.json()["detail"]

    failed = client.get("/runs/missing/failures")
    assert failed.status_code == 404

    compared = client.get("/compare", params={"a": "missing", "b": "also-missing"})
    assert compared.status_code == 404


def test_reject_unknown_provider_and_missing_dataset(client: TestClient, tmp_path: Path) -> None:
    unknown = client.post(
        "/runs",
        json={"provider": "nope", "dataset": str(tmp_path / "tasks.jsonl")},
    )
    assert unknown.status_code == 400

    missing = client.post(
        "/runs",
        json={"provider": "mock", "dataset": str(tmp_path / "absent.jsonl")},
    )
    assert missing.status_code == 404


def test_openapi_names_the_response_models(client: TestClient) -> None:
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    body = schema.json()
    assert {
        "/health",
        "/tasks",
        "/runs",
        "/runs/{run_id}",
        "/runs/{run_id}/failures",
        "/compare",
    } <= set(body["paths"])
    names = set(body["components"]["schemas"])
    assert {"RunCreate", "RunDetail", "CompareResponse", "FailureOut"} <= names
