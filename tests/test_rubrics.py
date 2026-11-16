import sqlite3
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from evalkit.cli import app
from evalkit.models import Task
from evalkit.providers import MockProvider, create_judge_provider
from evalkit.rubrics import Rubric, load_prompts, load_rubrics
from evalkit.runner import run_eval
from evalkit.storage import load_results, save_results

ROOT = Path(__file__).resolve().parents[1]
RUBRICS = ROOT / "datasets" / "rubrics"
PROMPTS = ROOT / "datasets" / "prompts"


def _write(directory: Path, name: str, payload: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(yaml.safe_dump(payload), encoding="utf-8")


def _rubric(rubric_id: str = "demo") -> dict[str, object]:
    return {
        "id": rubric_id,
        "version": 1,
        "description": "Demo rubric.",
        "pass_threshold": 0.5,
        "criteria": [
            {"id": "one", "weight": 0.5, "description": "First part."},
            {"id": "two", "weight": 0.5, "description": "Second part."},
        ],
    }


def test_shipped_rubrics_and_prompts_load() -> None:
    rubrics = load_rubrics(RUBRICS)
    assert set(rubrics) == {"helpful_answer_v1", "code_explanation_v1"}
    helpful = rubrics["helpful_answer_v1"]
    assert helpful.pass_threshold == 0.7
    assert [item.id for item in helpful.criteria] == [
        "accuracy",
        "instruction_following",
        "tone",
        "safety",
    ]
    prompts = load_prompts(PROMPTS, rubrics)
    assert len(prompts) == 10
    assert {prompt.rubric_id for prompt in prompts} == {
        "helpful_answer_v1",
        "code_explanation_v1",
    }
    assert "accuracy" in helpful.render()
    assert "weight 0.40" in helpful.render()


def test_content_hash_changes_when_the_rubric_changes() -> None:
    rubric = load_rubrics(RUBRICS)["helpful_answer_v1"]
    original = rubric.content_hash
    changed = rubric.model_copy(update={"description": rubric.description + " extra"})
    assert original != changed.content_hash
    assert len(original) == 64


def test_bad_rubrics_are_rejected(tmp_path: Path) -> None:
    weights = _rubric()
    weights["criteria"] = [
        {"id": "one", "weight": 0.5, "description": "First."},
        {"id": "two", "weight": 0.6, "description": "Second."},
    ]
    _write(tmp_path, "demo.yaml", weights)
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_rubrics(tmp_path)

    duplicate = _rubric()
    duplicate["criteria"] = [
        {"id": "accuracy", "weight": 0.5, "description": "First."},
        {"id": "accuracy", "weight": 0.5, "description": "Again."},
    ]
    _write(tmp_path, "demo.yaml", duplicate)
    with pytest.raises(ValueError, match="duplicate criterion ids"):
        load_rubrics(tmp_path)

    levels = _rubric()
    levels["criteria"][0]["levels"] = {1: "poor", 6: "not a level"}
    _write(tmp_path, "demo.yaml", levels)
    with pytest.raises(ValueError, match="1 to 5"):
        load_rubrics(tmp_path)

    mismatch = _rubric("other-id")
    _write(tmp_path, "demo.yaml", mismatch)
    with pytest.raises(ValueError, match="does not match filename"):
        load_rubrics(tmp_path)

    _write(tmp_path, "demo.yaml", _rubric("demo"))
    _write(tmp_path, "demo.yml", _rubric("demo"))
    with pytest.raises(ValueError, match="duplicate rubric id"):
        load_rubrics(tmp_path)


def test_unknown_rubric_id_fails_before_a_model_call(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"id": "refund-window", "category": "support", "prompt": "hi", "rubric_id": "missing"}\n'
        '{"id": "late-delivery", "category": "support", "prompt": "yo", "rubric_id": "also-missing"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refund-window") as raised:
        load_prompts(path, {})
    assert "late-delivery" in str(raised.value)


def test_task_needs_exactly_one_of_scorer_or_rubric() -> None:
    with pytest.raises(ValidationError, match="exactly one of scorer or rubric_id"):
        Task(id="bare", category="support", prompt="hi")
    with pytest.raises(ValidationError, match="exactly one of scorer or rubric_id"):
        Task(
            id="both",
            category="support",
            prompt="hi",
            scorer="exact_match",
            rubric_id="helpful_answer_v1",
            expected="4",
        )
    kept = Task(id="two-plus-two", category="math", prompt="2+2", scorer="exact_match", expected="4")
    assert kept.scorer == "exact_match"
    assert kept.rubric_id is None


def test_rubric_task_is_scored_by_a_mock_judge_and_saved(tmp_path: Path) -> None:
    rubrics = load_rubrics(RUBRICS)
    task = next(item for item in load_prompts(PROMPTS, rubrics) if item.id == "refund-window")

    class Judge:
        name = "judge"

        def generate(
            self,
            prompt: str,
            *,
            task_id: str | None = None,
            timeout: float | None = None,
        ) -> str:
            del task_id, timeout
            assert "accuracy" in prompt
            assert "30 days" in prompt
            return '{"score": 5, "reason": "meets the rubric"}'

    results = run_eval(
        [task],
        MockProvider({"refund-window": "Refunds are open for 30 days after delivery."}),
        "rubric-run",
        judge=Judge(),
        rubrics=rubrics,
        repeats=1,
        retries=1,
    )
    rubric = rubrics["helpful_answer_v1"]
    assert results[0].passed is True
    assert results[0].score == 1.0
    assert results[0].reason == "meets the rubric"
    assert results[0].rubric_id == "helpful_answer_v1"
    assert results[0].rubric_version == rubric.version
    assert results[0].rubric_hash == rubric.content_hash

    database = save_results("rubric-run", results, directory=tmp_path)
    assert load_results("rubric-run", directory=tmp_path) == results
    row = sqlite3.connect(database).execute(
        "SELECT rubric_id, rubric_version, rubric_hash FROM runs WHERE run_id = 'rubric-run'"
    ).fetchone()
    assert row == ("helpful_answer_v1", 1, rubric.content_hash)


def test_rubric_pass_threshold_overrides_the_judge_default() -> None:
    rubrics = load_rubrics(RUBRICS)
    task = Task(
        id="refund-window",
        category="support",
        prompt="How long is the refund window?",
        rubric_id="helpful_answer_v1",
    )

    class Judge:
        name = "judge"

        def generate(
            self,
            prompt: str,
            *,
            task_id: str | None = None,
            timeout: float | None = None,
        ) -> str:
            del prompt, task_id, timeout
            return '{"score": 3, "reason": "partial"}'

    results = run_eval(
        [task],
        MockProvider({"refund-window": "Thirty days."}),
        "threshold",
        judge=Judge(),
        rubrics=rubrics,
        repeats=1,
        retries=1,
    )
    assert results[0].score == 0.5
    assert results[0].passed is False


def test_missing_judge_fails_before_the_answer_model_runs() -> None:
    rubrics = load_rubrics(RUBRICS)
    task = Task(
        id="refund-window",
        category="support",
        prompt="How long is the refund window?",
        rubric_id="helpful_answer_v1",
    )

    class Answer(MockProvider):
        def generate(
            self,
            prompt: str,
            *,
            task_id: str | None = None,
            timeout: float | None = None,
        ) -> str:
            del prompt, task_id, timeout
            raise AssertionError("answer model was called")

    with pytest.raises(ValueError, match="judge"):
        run_eval([task], Answer(), "no-judge", rubrics=rubrics, repeats=1)


def test_create_judge_provider_uses_temperature_zero_and_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake(name: str, **kwargs: object) -> MockProvider:
        seen["name"] = name
        seen["kwargs"] = kwargs
        return MockProvider()

    monkeypatch.setattr("evalkit.providers.create_provider", fake)
    create_judge_provider("ollama", model="llama3.2")
    assert seen["name"] == "ollama"
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["model"] == "llama3.2"
    assert kwargs["temperature"] == 0
    assert kwargs["json_format"] is True


def test_cli_lists_and_validates_rubrics(tmp_path: Path) -> None:
    runner = CliRunner()
    listed = runner.invoke(app, ["rubrics", "list", "--directory", str(RUBRICS)])
    assert listed.exit_code == 0
    assert "helpful_answer_v1" in listed.stdout
    assert "code_explanation_v1" in listed.stdout

    validated = runner.invoke(app, ["rubrics", "validate", "--directory", str(RUBRICS)])
    assert validated.exit_code == 0
    assert validated.stdout.strip() == "2 rubrics ok"

    missing = runner.invoke(app, ["rubrics", "validate", "--directory", str(tmp_path / "absent")])
    assert missing.exit_code == 1
    assert "rubric directory not found" in missing.stdout + (missing.stderr or "")


def test_rubric_model_rejects_a_direct_bad_level_key() -> None:
    payload = _rubric()
    payload["criteria"][0]["levels"] = {"0": "too low", "5": "high"}
    with pytest.raises(ValidationError, match="1 to 5"):
        Rubric.model_validate(payload)
