import json
import sqlite3
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from evalkit.cli import app
from evalkit.models import Task
from evalkit.providers import MockProvider, create_judge_provider
from evalkit.rubrics import (
    Rubric,
    applied_rubric,
    lint_rubrics,
    load_prompts,
    load_rubrics,
    merge_rubric,
)
from evalkit.runner import run_eval
from evalkit.storage import load_results, save_results

ROOT = Path(__file__).resolve().parents[1]
RUBRICS = ROOT / "datasets" / "rubrics"
PROMPTS = ROOT / "datasets" / "prompts"


def _write(directory: Path, name: str, payload: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(yaml.safe_dump(payload), encoding="utf-8")


def _criterion(
    criterion_id: str,
    weight: float,
    description: str,
    **overrides: object,
) -> dict[str, object]:
    criterion: dict[str, object] = {
        "id": criterion_id,
        "weight": weight,
        "description": description,
        "necessity": "explicit",
        "kind": "objective",
        "answer_key": "A stated fact.",
    }
    criterion.update(overrides)
    return criterion


def _implicit(weight: float = 0.5) -> dict[str, object]:
    return _criterion(
        "goes-further",
        weight,
        "The reply adds a useful detail.",
        necessity="implicit",
        kind="subjective",
        example="A reply that adds one concrete next step.",
        answer_key=None,
    )


def _rubric(rubric_id: str = "demo") -> dict[str, object]:
    return {
        "id": rubric_id,
        "version": 1,
        "description": "Demo rubric.",
        "pass_threshold": 0.5,
        "criteria": [
            _criterion("one", 0.5, "First part."),
            _implicit(),
        ],
    }


def _lint(criteria: list[dict[str, object]], rubric_id: str = "demo") -> list[str]:
    payload = _rubric(rubric_id)
    payload["criteria"] = criteria
    rubric = Rubric.model_validate(payload)
    return lint_rubrics({rubric.id: rubric})


def test_shipped_rubrics_and_prompts_load() -> None:
    rubrics = load_rubrics(RUBRICS)
    assert set(rubrics) == {"helpful_answer_v1", "code_explanation_v1"}
    helpful = rubrics["helpful_answer_v1"]
    assert helpful.version == 2
    assert helpful.pass_threshold == 0.7
    assert [item.id for item in helpful.criteria] == [
        "answers_question",
        "no_invented_policy",
        "actionable_next_step",
        "plain_language",
        "no_sensitive_data_request",
    ]
    prompts = load_prompts(PROMPTS, rubrics)
    assert len(prompts) == 10
    assert {prompt.rubric_id for prompt in prompts} == {
        "helpful_answer_v1",
        "code_explanation_v1",
    }
    answers = helpful.criteria[0]
    assert answers.necessity == "explicit"
    assert answers.kind == "subjective"
    assert answers.example
    assert answers.critical is False
    invented = next(item for item in helpful.criteria if item.id == "no_invented_policy")
    assert invented.necessity == "implicit"
    assert invented.answer_key
    sensitive = next(item for item in helpful.criteria if item.id == "no_sensitive_data_request")
    assert sensitive.kind == "objective"
    assert sensitive.critical is True
    code = rubrics["code_explanation_v1"]
    assert code.criteria[0].kind == "objective"
    assert code.criteria[0].answer_key
    assert any(item.necessity == "implicit" for item in code.criteria)
    assert "answers_question" in helpful.render()
    assert "weight 0.30" in helpful.render()
    assert "answer key:" in helpful.render()


def test_content_hash_changes_when_the_rubric_changes() -> None:
    rubric = load_rubrics(RUBRICS)["helpful_answer_v1"]
    original = rubric.content_hash
    changed = rubric.model_copy(update={"description": rubric.description + " extra"})
    assert original != changed.content_hash
    assert len(original) == 64


def test_bad_rubrics_are_rejected(tmp_path: Path) -> None:
    weights = _rubric()
    weights["criteria"] = [
        _criterion("one", 0.5, "First."),
        _criterion("two", 0.6, "Second."),
    ]
    _write(tmp_path, "demo.yaml", weights)
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_rubrics(tmp_path)

    duplicate = _rubric()
    duplicate["criteria"] = [
        _criterion("accuracy", 0.5, "First."),
        _criterion("accuracy", 0.5, "Again."),
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
            assert "answers_question" in prompt
            assert "30 days" in prompt
            return json.dumps(
                {
                    "criteria": [
                        {"id": criterion.id, "met": True, "reason": "met"}
                        for criterion in rubrics["helpful_answer_v1"].criteria
                    ]
                }
            )

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
    assert results[0].reason == "all criteria met"
    assert results[0].rubric_id == "helpful_answer_v1"
    assert results[0].rubric_version == rubric.version
    assert results[0].rubric_hash == rubric.content_hash

    database = save_results("rubric-run", results, directory=tmp_path)
    assert load_results("rubric-run", directory=tmp_path) == results
    row = sqlite3.connect(database).execute(
        "SELECT rubric_id, rubric_version, rubric_hash FROM runs WHERE run_id = 'rubric-run'"
    ).fetchone()
    assert row == ("helpful_answer_v1", rubric.version, rubric.content_hash)


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
            met = {
                "answers_question": False,
                "no_invented_policy": True,
                "actionable_next_step": False,
                "plain_language": True,
                "no_sensitive_data_request": True,
            }
            return json.dumps(
                {
                    "criteria": [
                        {"id": criterion_id, "met": flag, "reason": "checked"}
                        for criterion_id, flag in met.items()
                    ]
                }
            )

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


def test_objective_criteria_require_an_answer_key() -> None:
    missing = _criterion("one", 1.0, "The reply states 4.", answer_key=None)
    with pytest.raises(ValidationError, match="answer_key"):
        Rubric.model_validate({**_rubric(), "criteria": [missing]})
    blank = _criterion("one", 1.0, "The reply states 4.", answer_key="  ")
    with pytest.raises(ValidationError, match="answer_key"):
        Rubric.model_validate({**_rubric(), "criteria": [blank]})
    kept = Rubric.model_validate(
        {**_rubric(), "criteria": [_criterion("one", 1.0, "The reply states 4.")]}
    )
    assert kept.criteria[0].answer_key == "A stated fact."
    assert kept.criteria[0].critical is False


def test_subjective_criteria_require_an_example() -> None:
    missing = _criterion(
        "one",
        1.0,
        "The reply sounds clear.",
        kind="subjective",
        answer_key=None,
    )
    with pytest.raises(ValidationError, match="example"):
        Rubric.model_validate({**_rubric(), "criteria": [missing]})
    blank = _criterion(
        "one",
        1.0,
        "The reply sounds clear.",
        kind="subjective",
        example="  ",
        answer_key=None,
    )
    with pytest.raises(ValidationError, match="example"):
        Rubric.model_validate({**_rubric(), "criteria": [blank]})
    kept = Rubric.model_validate(
        {
            **_rubric(),
            "criteria": [
                _criterion(
                    "one",
                    1.0,
                    "The reply sounds clear.",
                    kind="subjective",
                    example="A plain reply.",
                    answer_key=None,
                )
            ],
        }
    )
    assert kept.criteria[0].example == "A plain reply."
    assert kept.criteria[0].necessity == "explicit"


def test_necessity_and_kind_are_closed_sets() -> None:
    bad_necessity = _criterion("one", 1.0, "The reply states 4.", necessity="optional")
    with pytest.raises(ValidationError):
        Rubric.model_validate({**_rubric(), "criteria": [bad_necessity]})
    bad_kind = _criterion("one", 1.0, "The reply states 4.", kind="mixed")
    with pytest.raises(ValidationError):
        Rubric.model_validate({**_rubric(), "criteria": [bad_kind]})


def test_lint_warns_when_and_joins_two_verbs() -> None:
    warnings = _lint(
        [
            _criterion("packed", 0.5, "The reply mentions bees and explains honey."),
            _criterion(
                "short",
                0.5,
                "The reply is short and clear.",
                necessity="implicit",
                kind="subjective",
                example="A short reply.",
                answer_key=None,
            ),
        ]
    )
    assert warnings == ["demo packed: description likely holds two requirements"]


def test_lint_warns_on_a_comma_list_of_demands() -> None:
    warnings = _lint(
        [
            _criterion("items", 0.5, "The reply covers price, date, and place."),
            _criterion(
                "aside",
                0.5,
                "The reply states the price, with the date in the next sentence.",
                necessity="implicit",
                kind="subjective",
                example="A reply that states the price.",
                answer_key=None,
            ),
        ]
    )
    assert warnings == ["demo items: description likely holds two requirements"]


def test_lint_warns_when_the_grader_cannot_see_the_fact() -> None:
    hidden = [
        ("correct-total", "The reply states the correct total."),
        ("quoted-figure", "The reply quotes a figure from the text."),
        ("prompt-date", "The reply uses the date in the prompt."),
    ]
    criteria = [
        _criterion(
            criterion_id,
            0.25,
            description,
            kind="subjective",
            example="A reply that states the figure.",
            answer_key=None,
        )
        for criterion_id, description in hidden
    ]
    criteria.append(_implicit(0.25))
    warnings = _lint(criteria)
    assert warnings == [
        "demo correct-total: refers to information the grader cannot see",
        "demo quoted-figure: refers to information the grader cannot see",
        "demo prompt-date: refers to information the grader cannot see",
    ]
    grounded = _lint(
        [
            _criterion("correct-total", 0.5, "The reply states the correct total."),
            _implicit(),
        ]
    )
    assert grounded == []


def test_lint_warns_on_generic_criteria() -> None:
    generic = [
        ("spelling", "The reply uses correct spelling."),
        ("courtesy", "The reply shows politeness."),
        ("obedience", "The reply follows instructions."),
    ]
    criteria = [_criterion(criterion_id, 0.25, description) for criterion_id, description in generic]
    criteria.append(_implicit(0.25))
    warnings = _lint(criteria)
    assert warnings == [
        "demo spelling: criterion is generic",
        "demo courtesy: criterion is generic",
        "demo obedience: criterion is generic",
    ]


def test_lint_warns_when_a_rubric_has_no_implicit_criteria() -> None:
    warnings = _lint(
        [
            _criterion("one", 0.5, "The reply states 4."),
            _criterion(
                "two",
                0.5,
                "The reply sounds clear.",
                kind="subjective",
                example="A plain reply.",
                answer_key=None,
            ),
        ]
    )
    assert warnings == ["demo: rubric has no implicit criteria"]


def test_cli_lint_prints_ids_and_does_not_fail(tmp_path: Path) -> None:
    runner = CliRunner()
    clean = tmp_path / "clean"
    _write(clean, "demo.yaml", _rubric("demo"))
    quiet = runner.invoke(app, ["rubrics", "lint", "--directory", str(clean)])
    assert quiet.exit_code == 0
    assert quiet.stdout.strip() == "1 rubrics ok"

    noisy = tmp_path / "noisy"
    payload = _rubric("demo")
    payload["criteria"] = [_criterion("spelling", 1.0, "The reply uses correct spelling.")]
    _write(noisy, "demo.yaml", payload)
    warned = runner.invoke(app, ["rubrics", "lint", "--directory", str(noisy)])
    assert warned.exit_code == 0
    assert "demo spelling: criterion is generic" in warned.stdout
    assert "demo: rubric has no implicit criteria" in warned.stdout

    strict = runner.invoke(app, ["rubrics", "lint", "--strict", "--directory", str(noisy)])
    assert strict.exit_code == 1
    assert "demo spelling: criterion is generic" in strict.stdout

    strict_clean = runner.invoke(app, ["rubrics", "lint", "--strict", "--directory", str(clean)])
    assert strict_clean.exit_code == 0
    assert strict_clean.stdout.strip() == "1 rubrics ok"

    shipped = runner.invoke(app, ["rubrics", "lint", "--strict", "--directory", str(RUBRICS)])
    assert shipped.exit_code == 0
    assert shipped.stdout.strip() == "2 rubrics ok"

    missing = runner.invoke(app, ["rubrics", "lint", "--directory", str(tmp_path / "absent")])
    assert missing.exit_code == 1
    assert "rubric directory not found" in missing.stdout + (missing.stderr or "")


def test_extra_criteria_keep_relative_weights_and_then_sum_to_one() -> None:
    payload = _rubric()
    payload["criteria"] = [
        _criterion("one", 0.6, "First part."),
        _implicit(0.4),
    ]
    rubric = Rubric.model_validate(payload)
    extra = Rubric.model_validate(
        {
            "id": "extra",
            "version": 1,
            "description": "holder",
            "pass_threshold": 1,
            "criteria": [_criterion("cite", 1.0, "The reply names the source page.")],
        }
    ).criteria[0]
    extra = extra.model_copy(update={"weight": 0.5})
    merged = merge_rubric(rubric, [extra])
    assert [item.weight for item in rubric.criteria] == [0.6, 0.4]
    assert merged.criteria[0].weight / merged.criteria[1].weight == pytest.approx(0.6 / 0.4)
    assert merged.criteria[2].id == "cite"
    assert merged.criteria[2].weight == pytest.approx(0.5 / 1.5)
    assert sum(item.weight for item in merged.criteria) == pytest.approx(1)
    assert merged.content_hash != rubric.content_hash
    assert merge_rubric(rubric, []).content_hash == rubric.content_hash


def test_duplicate_extra_criterion_ids_are_rejected(tmp_path: Path) -> None:
    rubric = Rubric.model_validate(_rubric())
    extra = Rubric.model_validate(
        {
            "id": "extra",
            "version": 1,
            "description": "holder",
            "pass_threshold": 1,
            "criteria": [_criterion("one", 1.0, "Again.")],
        }
    ).criteria[0]
    with pytest.raises(ValueError, match="duplicate criterion ids: one"):
        merge_rubric(rubric, [extra])

    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "ask",
                "category": "support",
                "prompt": "Hello",
                "rubric_id": "demo",
                "extra_criteria": [_criterion("one", 0.2, "Again.")],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ask: duplicate criterion ids: one"):
        load_prompts(path, {"demo": rubric})


def test_grader_prompt_includes_extra_criteria_and_stores_the_merged_hash() -> None:
    rubric = Rubric.model_validate(_rubric())
    task = Task(
        id="ask",
        category="support",
        prompt="Where is the policy?",
        rubric_id="demo",
        extra_criteria=[_criterion("cite", 0.25, "The reply names the source page.")],
    )
    merged = applied_rubric(task, {"demo": rubric})
    assert merged is not None

    class Judge:
        name = "judge"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str, **kwargs: object) -> str:
            del kwargs
            self.prompts.append(prompt)
            return json.dumps(
                {
                    "criteria": [
                        {"id": item.id, "met": True, "reason": "yes"} for item in merged.criteria
                    ]
                }
            )

    judge = Judge()
    results = run_eval(
        [task],
        MockProvider({"ask": "The policy is on the account page."}),
        "merged",
        judge=judge,
        rubrics={"demo": rubric},
        repeats=1,
        retries=1,
    )
    assert "cite" in judge.prompts[0]
    assert results[0].passed is True
    assert results[0].rubric_hash == merged.content_hash
    assert results[0].rubric_hash != rubric.content_hash


def test_fabricated_link_overrides_the_grader() -> None:
    rubric = Rubric.model_validate(_rubric())
    task = Task(
        id="ask",
        category="support",
        prompt="Where is the policy?",
        rubric_id="demo",
        allowed_urls=["https://example.com/policy"],
        deterministic_checks=["no_fabricated_links"],
    )

    class Judge:
        name = "judge"

        def generate(self, prompt: str, **kwargs: object) -> str:
            del prompt, kwargs
            raise AssertionError("grader ran")

    results = run_eval(
        [task],
        MockProvider({"ask": "See https://evil.example/phish"}),
        "links",
        judge=Judge(),
        rubrics={"demo": rubric},
        repeats=1,
        retries=1,
    )
    assert results[0].passed is False
    assert results[0].failure_type == "critical:no_fabricated_links"
    assert "https://evil.example/phish" in results[0].reason
    assert results[0].rubric_hash == rubric.content_hash


def test_lint_warns_on_extra_criteria(tmp_path: Path) -> None:
    rubrics = tmp_path / "rubrics"
    prompts = tmp_path / "prompts"
    _write(rubrics, "demo.yaml", _rubric("demo"))
    prompts.mkdir()
    (prompts / "ask.jsonl").write_text(
        json.dumps(
            {
                "id": "ask",
                "category": "support",
                "prompt": "Hello",
                "rubric_id": "demo",
                "extra_criteria": [
                    _criterion(
                        "tone",
                        0.2,
                        "The reply follows instructions.",
                        necessity="explicit",
                        kind="subjective",
                        example="It restates the request.",
                        answer_key=None,
                    )
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    warned = runner.invoke(app, ["rubrics", "lint", "--directory", str(rubrics)])
    assert warned.exit_code == 0
    assert "ask tone: criterion is generic" in warned.stdout
    strict = runner.invoke(app, ["rubrics", "lint", "--strict", "--directory", str(rubrics)])
    assert strict.exit_code == 1
