import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from evalkit.api import create_app
from evalkit.providers import ProviderError
from evalkit.storage import load_results, upsert_run

TASKS = """\
{"id": "two-plus-two", "category": "math", "prompt": "2+2", "scorer": "exact_match", "expected": "4"}
{"id": "capital-of-france", "category": "knowledge", "prompt": "capital", "scorer": "exact_match", "expected": "Paris"}
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
    assert [task["id"] for task in body] == ["two-plus-two", "capital-of-france"]
    assert body[0]["scorer"] == "exact_match"
    assert body[1]["expected"] == "Paris"


def test_mock_run_summary_and_failures(client: TestClient, dataset: Path) -> None:
    created = client.post("/runs", json={"provider": "mock", "dataset": str(dataset)})
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    assert created.json()["status"] == "answering"

    listed = client.get("/runs")
    assert listed.status_code == 200
    match = next(item for item in listed.json() if item["run_id"] == run_id)
    assert match["status"] == "graded"
    assert match["provider"] == "mock"

    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == "graded"
    assert body["answers_done"] == 6
    assert body["answers_expected"] == 6
    assert body["grades_done"] == 6
    assert body["grades_expected"] == 6
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
    assert [item["task_id"] for item in reasons] == [
        "capital-of-france",
        "capital-of-france",
        "capital-of-france",
    ]
    assert reasons[0]["category"] == "knowledge"
    assert reasons[0]["failure_type"] == "wrong answer"
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


def test_post_runs_forwards_model_temperature_and_workers(
    client: TestClient,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[object, ...]] = []

    def create(name: str, **kwargs: object) -> object:
        seen.append(("create", name, kwargs))

        class Selected:
            name = "mock"

        return Selected()

    def answer_run(*args: object, **kwargs: object) -> None:
        del args
        seen.append(("answer", kwargs))

    def grade_run(*args: object, **kwargs: object) -> None:
        del args
        seen.append(("grade", kwargs))

    monkeypatch.setattr("evalkit.api.create_provider", create)
    monkeypatch.setattr("evalkit.api.answer_run", answer_run)
    monkeypatch.setattr("evalkit.api.grade_run", grade_run)
    response = client.post(
        "/runs",
        json={
            "provider": "ollama",
            "dataset": str(dataset),
            "model": "llama3.1",
            "temperature": 0.4,
            "workers": 2,
        },
    )
    assert response.status_code == 202
    creates = [item for item in seen if item[0] == "create"]
    assert creates[0][2] == {"model": "llama3.1", "temperature": 0.4}
    answer_calls = [item for item in seen if item[0] == "answer"]
    assert answer_calls[0][1]["workers"] == 2
    assert answer_calls[0][1]["attempts"] == 3

    once = client.post(
        "/runs",
        json={"provider": "mock", "dataset": str(dataset), "attempts": 1},
    )
    assert once.status_code == 202
    answer_calls = [item for item in seen if item[0] == "answer"]
    assert answer_calls[-1][1]["attempts"] == 1

    rejected = client.post(
        "/runs",
        json={"provider": "mock", "dataset": str(dataset), "workers": 0},
    )
    assert rejected.status_code == 422
    zero_attempts = client.post(
        "/runs",
        json={"provider": "mock", "dataset": str(dataset), "attempts": 0},
    )
    assert zero_attempts.status_code == 422


def test_post_runs_rejects_a_missing_ollama_model(
    client: TestClient,
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Missing:
        def check_model_available(self) -> None:
            raise ProviderError("run: ollama pull llama3.2")

    monkeypatch.setattr("evalkit.api.create_provider", lambda *args, **kwargs: Missing())
    response = client.post(
        "/runs",
        json={"provider": "ollama", "dataset": str(dataset), "model": "llama3.2"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "run: ollama pull llama3.2"


def test_grade_keeps_a_second_judge_and_reports_progress(
    client: TestClient,
    dataset: Path,
) -> None:
    created = client.post(
        "/runs",
        json={"provider": "mock", "dataset": str(dataset), "attempts": 1},
    )
    run_id = created.json()["run_id"]
    graded = client.post(
        f"/runs/{run_id}/grade",
        json={"judge_provider": "mock", "judge_model": "beta"},
    )
    assert graded.status_code == 202
    assert graded.json()["status"] == "grading"

    directory = dataset.parent / "runs"
    original = load_results(run_id, directory=directory, judge="")
    second = load_results(run_id, directory=directory, judge="beta")
    assert len(original) == 2
    assert [(item.task_id, item.score) for item in second] == [
        (item.task_id, item.score) for item in original
    ]
    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "graded"
    assert detail.json()["grades_done"] == 2

    missing = client.post("/runs/missing/grade", json={})
    assert missing.status_code == 404
    upsert_run(
        "stuck",
        directory=directory,
        status="answering",
        provider="mock",
        dataset=str(dataset),
        attempts=1,
        task_count=1,
    )
    busy = client.post("/runs/stuck/grade", json={})
    assert busy.status_code == 409


def test_openapi_names_the_response_models(client: TestClient) -> None:
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    body = schema.json()
    assert {
        "/health",
        "/tasks",
        "/rubrics",
        "/rubrics/{rubric_id}",
        "/rubrics/{rubric_id}/versions",
        "/prompts",
        "/runs",
        "/runs/{run_id}",
        "/runs/{run_id}/grade",
        "/runs/{run_id}/failures",
        "/runs/{run_id}/criteria",
        "/compare",
    } <= set(body["paths"])
    names = set(body["components"]["schemas"])
    assert {
        "RunCreate",
        "RunDetail",
        "CompareResponse",
        "FailureOut",
        "RubricCreated",
        "CriterionRate",
        "Task",
        "Rubric",
    } <= names


def _rubric_body(description: str, *, implicit: bool) -> dict[str, object]:
    criteria: list[dict[str, object]] = [
        {
            "id": "answers",
            "weight": 0.6 if implicit else 1.0,
            "description": description,
            "necessity": "explicit",
            "kind": "subjective",
            "example": "It states the refund window.",
        }
    ]
    if implicit:
        criteria.append(
            {
                "id": "no_password",
                "weight": 0.4,
                "description": "The reply never asks for a password.",
                "necessity": "implicit",
                "kind": "objective",
                "critical": True,
                "answer_key": "Fails if the reply asks for a password.",
            }
        )
    return {
        "id": "demo",
        "version": 1,
        "description": "Support replies.",
        "pass_threshold": 0.5,
        "criteria": criteria,
    }


def test_rubric_posts_keep_each_version_and_return_lint_warnings(client: TestClient) -> None:
    first = client.post(
        "/rubrics",
        json=_rubric_body("The reply answers the question.", implicit=False),
    )
    assert first.status_code == 201
    stored = first.json()
    assert stored["rubric"]["version"] == 1
    assert stored["warnings"] == ["demo: rubric has no implicit criteria"]

    invalid = client.post(
        "/rubrics",
        json=_rubric_body("The reply answers the question.", implicit=False) | {"pass_threshold": 2},
    )
    assert invalid.status_code == 422

    second = client.post(
        "/rubrics",
        json=_rubric_body("The reply states the refund window.", implicit=True),
    )
    assert second.status_code == 201
    assert second.json()["rubric"]["version"] == 2
    assert second.json()["warnings"] == []

    latest = client.get("/rubrics/demo")
    assert latest.status_code == 200
    assert latest.json()["version"] == 2
    assert latest.json()["criteria"][0]["description"] == "The reply states the refund window."

    versions = client.get("/rubrics/demo/versions")
    assert versions.status_code == 200
    body = versions.json()
    assert [item["version"] for item in body] == [1, 2]
    assert body[0]["criteria"][0]["description"] == "The reply answers the question."
    assert body[0]["criteria"][0]["weight"] == 1.0

    missing = client.get("/rubrics/missing")
    assert missing.status_code == 404
    assert client.get("/rubrics/missing/versions").status_code == 404


def test_prompts_reject_an_unknown_rubric_and_list_saved_prompts(client: TestClient) -> None:
    unknown = client.post(
        "/prompts",
        json={
            "id": "refund",
            "category": "support",
            "prompt": "How long is the refund window?",
            "rubric_id": "demo",
        },
    )
    assert unknown.status_code == 400
    assert "unknown rubric_id" in unknown.json()["detail"]

    client.post("/rubrics", json=_rubric_body("The reply answers the question.", implicit=True))
    saved = client.post(
        "/prompts",
        json={
            "id": "refund",
            "category": "support",
            "prompt": "How long is the refund window?",
            "rubric_id": "demo",
        },
    )
    assert saved.status_code == 201
    assert saved.json()["rubric_id"] == "demo"

    scored = client.post(
        "/prompts",
        json={
            "id": "two-plus-two",
            "category": "math",
            "prompt": "2+2",
            "scorer": "exact_match",
            "expected": "4",
        },
    )
    assert scored.status_code == 201

    replaced = client.post(
        "/prompts",
        json={
            "id": "refund",
            "category": "support",
            "prompt": "What is the refund window?",
            "rubric_id": "demo",
        },
    )
    assert replaced.status_code == 201

    listed = client.get("/prompts")
    assert listed.status_code == 200
    by_id = {item["id"]: item for item in listed.json()}
    assert set(by_id) == {"refund", "two-plus-two"}
    assert by_id["refund"]["prompt"] == "What is the refund window?"


def test_run_can_use_prompts_stored_in_sqlite(client: TestClient) -> None:
    empty = client.post("/runs", json={"provider": "mock", "dataset": "sqlite", "attempts": 1})
    assert empty.status_code == 400
    assert empty.json()["detail"] == "no prompts stored"

    client.post(
        "/prompts",
        json={
            "id": "two-plus-two",
            "category": "math",
            "prompt": "2+2",
            "scorer": "exact_match",
            "expected": "4",
        },
    )
    created = client.post("/runs", json={"provider": "mock", "dataset": "sqlite", "attempts": 1})
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "graded"
    assert detail.json()["summary"][0]["pass_rate"] == 1.0

    rates = client.get(f"/runs/{run_id}/criteria")
    assert rates.status_code == 200
    assert rates.json() == []
    missing = client.get("/runs/missing/criteria")
    assert missing.status_code == 404


def test_criterion_rates_use_sqlite_and_file_sources(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client.post("/rubrics", json=_rubric_body("The reply answers the question.", implicit=True))
    client.post(
        "/prompts",
        json={
            "id": "stored-task",
            "category": "support",
            "prompt": "How long is the refund window?",
            "rubric_id": "demo",
        },
    )
    dataset = tmp_path / "file_prompts.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "file-task",
                "category": "support",
                "prompt": "Where do I send the return?",
                "rubric_id": "demo",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Judge:
        def __init__(self) -> None:
            self.calls = 0
            self.name = "judge"

        def generate(self, prompt: str, **kwargs: object) -> str:
            del kwargs
            self.calls += 1
            return json.dumps(
                {
                    "criteria": [
                        {"id": "answers", "met": True, "reason": "yes"},
                        {"id": "no_password", "met": self.calls % 2 == 1, "reason": "checked"},
                    ]
                }
            )

    monkeypatch.setattr("evalkit.api.create_judge_provider", lambda *args, **kwargs: Judge())
    stored = client.post(
        "/runs",
        json={
            "provider": "mock",
            "dataset": "sqlite",
            "attempts": 2,
            "judge_provider": "mock",
        },
    )
    assert stored.status_code == 202
    stored_id = stored.json()["run_id"]
    stored_rates = client.get(f"/runs/{stored_id}/criteria")
    assert stored_rates.status_code == 200
    assert {(row["criterion_id"], row["model"], row["met_rate"]) for row in stored_rates.json()} == {
        ("answers", "mock", 1.0),
        ("no_password", "mock", 0.5),
    }

    filed = client.post(
        "/runs",
        json={
            "provider": "mock",
            "dataset": str(dataset),
            "attempts": 1,
            "judge_provider": "mock",
        },
    )
    assert filed.status_code == 202
    filed_id = filed.json()["run_id"]
    filed_detail = client.get(f"/runs/{filed_id}")
    assert filed_detail.json()["status"] == "graded"
    filed_rates = client.get(f"/runs/{filed_id}/criteria")
    assert filed_rates.status_code == 200
    assert {row["criterion_id"] for row in filed_rates.json()} == {"answers", "no_password"}


def test_sqlite_prompt_can_use_a_file_rubric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rubrics = tmp_path / "rubrics"
    rubrics.mkdir()
    (rubrics / "filedemo.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "filedemo",
                "version": 1,
                "description": "A short reply.",
                "pass_threshold": 0.5,
                "criteria": [
                    {
                        "id": "clear",
                        "weight": 1.0,
                        "necessity": "explicit",
                        "kind": "subjective",
                        "description": "The reply is clear.",
                        "example": "It answers in one sentence.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            runs_directory=tmp_path / "runs",
            default_dataset=tmp_path / "unused.jsonl",
            rubrics_directory=rubrics,
        )
    )
    missing = client.post(
        "/prompts",
        json={
            "id": "ask",
            "category": "support",
            "prompt": "Say hello.",
            "rubric_id": "not-a-rubric",
        },
    )
    assert missing.status_code == 400

    saved = client.post(
        "/prompts",
        json={
            "id": "ask",
            "category": "support",
            "prompt": "Say hello.",
            "rubric_id": "filedemo",
        },
    )
    assert saved.status_code == 201

    class Judge:
        name = "judge"

        def generate(self, prompt: str, **kwargs: object) -> str:
            del prompt, kwargs
            return json.dumps({"criteria": [{"id": "clear", "met": True, "reason": "yes"}]})

    monkeypatch.setattr("evalkit.api.create_judge_provider", lambda *args, **kwargs: Judge())
    created = client.post(
        "/runs",
        json={"provider": "mock", "dataset": "sqlite", "attempts": 1, "judge_provider": "mock"},
    )
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    detail = client.get(f"/runs/{run_id}")
    assert detail.json()["status"] == "graded"
    assert detail.json()["summary"][0]["pass_rate"] == 1.0
    rates = client.get(f"/runs/{run_id}/criteria")
    assert rates.json() == [{"criterion_id": "clear", "model": "mock", "met_rate": 1.0}]

