"""Per-criterion rubric grading, agent config, and the criterion report."""

import json
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from evalkit.agents import resolve_agents
from evalkit.cli import app
from evalkit.grading import grade_response
from evalkit.models import CriterionMark, TaskResult
from evalkit.pipeline import answer_run, grade_run
from evalkit.providers import MockProvider
from evalkit.rubrics import Criterion, Rubric, load_rubrics
from evalkit.runner import failure_for
from evalkit.storage import (
    load_criterion_grades,
    load_results,
    replace_criterion_grades,
    save_results,
)

CLI = CliRunner()


def _rubric(*criteria: Criterion, threshold: float = 0.6) -> Rubric:
    return Rubric(
        id="demo",
        version=1,
        description="A short reply.",
        pass_threshold=threshold,
        criteria=list(criteria),
    )


def _subjective(criterion_id: str, weight: float, *, critical: bool = False) -> Criterion:
    return Criterion(
        id=criterion_id,
        weight=weight,
        description=f"The reply does {criterion_id}.",
        necessity="explicit",
        kind="subjective",
        critical=critical,
        example=f"An example of {criterion_id}.",
    )


def _objective(criterion_id: str, weight: float, *, critical: bool = False) -> Criterion:
    return Criterion(
        id=criterion_id,
        weight=weight,
        description=f"The reply includes {criterion_id}.",
        necessity="implicit",
        kind="objective",
        critical=critical,
        answer_key=f"The reply must include {criterion_id}.",
    )


def _marks(flags: dict[str, bool]) -> str:
    return json.dumps(
        {
            "criteria": [
                {"id": criterion_id, "met": met, "reason": "checked"}
                for criterion_id, met in flags.items()
            ]
        }
    )


class _Judge:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.name = "judge"

    def generate(self, prompt: str, **kwargs: object) -> str:
        del kwargs
        self.prompts.append(prompt)
        return self.reply


def test_score_is_the_sum_of_met_weights() -> None:
    rubric = _rubric(
        _subjective("a", 0.5),
        _subjective("b", 0.3),
        _objective("c", 0.2),
    )
    scored = grade_response(
        rubric,
        "hello",
        "question",
        _Judge(_marks({"a": True, "b": False, "c": True})),
    )
    assert scored.score == 0.7
    assert scored.passed is True
    assert scored.reason == "score meets the threshold"
    assert scored.forced_failure is None
    assert [mark.criterion_id for mark in scored.criteria] == ["a", "b", "c"]
    assert [mark.met for mark in scored.criteria] == [True, False, True]

    low = grade_response(
        rubric,
        "hello",
        "question",
        _Judge(_marks({"a": False, "b": False, "c": True})),
    )
    assert low.score == 0.2
    assert low.passed is False
    assert low.reason == "unmet criteria: a, b"
    assert failure_for(low, "hello") == "wrong answer"


def test_critical_failure_overrides_a_high_score() -> None:
    rubric = _rubric(
        _subjective("covered", 0.75),
        _objective("must_cite", 0.25, critical=True),
        threshold=0.5,
    )
    scored = grade_response(
        rubric,
        "a long correct reply",
        "question",
        _Judge(_marks({"covered": True, "must_cite": False})),
    )
    assert scored.score == 0.75
    assert scored.passed is False
    assert scored.forced_failure == "critical:must_cite"
    assert failure_for(scored, "a long correct reply") == "critical:must_cite"

    both = _rubric(
        _subjective("covered", 0.5),
        _objective("first_gap", 0.25, critical=True),
        _objective("second_gap", 0.25, critical=True),
        threshold=0.5,
    )
    failed = grade_response(
        both,
        "reply",
        "question",
        _Judge(_marks({"covered": True, "first_gap": False, "second_gap": False})),
    )
    assert failed.score == 0.5
    assert failed.forced_failure == "critical:first_gap"


def test_missing_or_duplicate_criterion_is_a_parse_error() -> None:
    rubric = _rubric(_subjective("a", 0.5), _objective("b", 0.5))
    missing = grade_response(rubric, "reply", "question", _Judge(_marks({"a": True})))
    assert missing.score == 0.0
    assert missing.passed is False
    assert missing.forced_failure == "grader_parse_error"
    assert missing.criteria == []
    assert "missing criterion ids: b" in missing.reason

    duplicated = json.dumps(
        {
            "criteria": [
                {"id": "a", "met": True, "reason": "once"},
                {"id": "b", "met": True, "reason": "once"},
                {"id": "a", "met": False, "reason": "again"},
            ]
        }
    )
    duplicate = grade_response(rubric, "reply", "question", _Judge(duplicated))
    assert duplicate.forced_failure == "grader_parse_error"
    assert "duplicate criterion ids: a" in duplicate.reason

    unknown = grade_response(
        rubric,
        "reply",
        "question",
        _Judge(_marks({"a": True, "b": True, "extra": False})),
    )
    assert unknown.forced_failure == "grader_parse_error"
    assert "unknown criterion ids: extra" in unknown.reason

    invalid = grade_response(rubric, "reply", "question", _Judge("not json"))
    assert invalid.forced_failure == "grader_parse_error"
    assert "invalid JSON" in invalid.reason


def test_response_only_never_includes_the_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "SECRET_PROMPT_XYZ"
    rubric = _rubric(_subjective("a", 0.6), _objective("b", 0.4))
    judge = _Judge(_marks({"a": True, "b": True}))
    grade_response(rubric, "Hello there", secret, judge, context="response_only")
    assert secret not in judge.prompts[0]
    assert "Hello there" in judge.prompts[0]
    assert "answer key:" in judge.prompts[0]
    assert "example:" in judge.prompts[0]
    grade_response(rubric, "Hello there", secret, judge, context="with_prompt")
    assert secret in judge.prompts[1]

    monkeypatch.chdir(tmp_path)
    rubric_dir, dataset = _write_dataset(tmp_path, prompt=secret)
    catalog = load_rubrics(rubric_dir)

    class Recording:
        name = "judge"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str, **kwargs: object) -> str:
            del kwargs
            self.prompts.append(prompt)
            return _marks({"clear": True, "safe": True})

    recording = Recording()
    answer_run(
        dataset,
        MockProvider({"only": "Hello there"}),
        "context-run",
        attempts=1,
        rubrics=catalog,
        provider_name="mock",
    )
    grade_run("context-run", judge=recording, rubrics=catalog, grader_context="response_only")
    assert recording.prompts
    assert secret not in recording.prompts[0]
    assert "Hello there" in recording.prompts[0]


def test_parse_error_drops_grade_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    rubric_dir, dataset = _write_dataset(tmp_path, prompt="How long is the window?")
    catalog = load_rubrics(rubric_dir)
    answer_run(
        dataset,
        MockProvider({"only": "Thirty days."}),
        "parse-run",
        attempts=1,
        rubrics=catalog,
        provider_name="mock",
    )
    saved = grade_run(
        "parse-run",
        judge=_Judge(_marks({"clear": True, "safe": True})),
        rubrics=catalog,
    )
    assert saved[0].passed is True
    grades = load_criterion_grades("parse-run", judge="judge")
    assert [grade.criterion_id for grade in grades] == ["clear", "safe"]
    assert all(grade.met for grade in grades)
    assert grades[0].rubric_id == "demo"
    assert grades[0].rubric_version == 1
    assert grades[0].rubric_hash == catalog["demo"].content_hash

    other = grade_run(
        "parse-run",
        judge=_Judge(_marks({"clear": True})),
        judge_model="other",
        rubrics=catalog,
    )
    assert other[0].failure_type == "grader_parse_error"
    assert [grade.criterion_id for grade in load_criterion_grades("parse-run", judge="judge")] == [
        "clear",
        "safe",
    ]
    assert load_criterion_grades("parse-run", judge="other") == []

    failed = grade_run(
        "parse-run",
        judge=_Judge(_marks({"clear": True})),
        rubrics=catalog,
    )
    assert failed[0].failure_type == "grader_parse_error"
    assert failed[0].passed is False
    assert load_criterion_grades("parse-run", judge="judge") == []


def test_report_shows_met_rates_and_critical_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    rubric_dir, dataset = _write_dataset(
        tmp_path,
        prompts=(
            ("good", "ALPHA_REPLY"),
            ("bad", "BETA_REPLY"),
        ),
    )
    catalog = load_rubrics(rubric_dir)

    class Judge:
        name = "judge"

        def generate(self, prompt: str, **kwargs: object) -> str:
            del kwargs
            return _marks({"clear": True, "safe": "ALPHA_REPLY" in prompt})

    answer_run(
        dataset,
        MockProvider({"good": "ALPHA_REPLY", "bad": "BETA_REPLY"}),
        "report-run",
        attempts=1,
        rubrics=catalog,
        provider_name="mock",
    )
    graded = grade_run("report-run", judge=Judge(), rubrics=catalog)
    by_task = {result.task_id: result for result in graded}
    assert by_task["good"].passed is True
    assert by_task["bad"].score == 0.75
    assert by_task["bad"].failure_type == "critical:safe"

    reported = CLI.invoke(app, ["report", "report-run"])
    assert reported.exit_code == 0
    assert "criteria" in reported.stdout
    assert "clear" in reported.stdout
    assert "safe" in reported.stdout
    assert "1.00" in reported.stdout
    assert "0.50" in reported.stdout
    assert "critical failures" in reported.stdout
    assert "bad  attempt 1  mock  critical:safe" in reported.stdout
    assert load_results("report-run")[0].model == "mock"


def test_show_prints_criterion_grades_for_each_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = Path("runs")
    ask = _stored_attempt("ask", 1, "WINDOW_REPLY", score=0.8, reason="first judge summary")
    other_attempt = _stored_attempt(
        "ask",
        2,
        "OTHER_REPLY",
        score=0.8,
        reason="first judge summary",
    )
    latest_ask = ask.model_copy(
        update={"judge": "judge-b", "score": 0.4, "reason": "second judge summary"}
    )
    latest_other = other_attempt.model_copy(
        update={"judge": "judge-b", "score": 0.4, "reason": "second judge summary"}
    )
    save_results(
        "fake-run",
        [ask, other_attempt, latest_ask, latest_other],
        directory=directory,
    )
    _store_grades(
        "ask",
        1,
        "judge-a",
        [("clear", True, "Clear enough."), ("safe", False, "Gave a secret.")],
        directory,
    )
    _store_grades(
        "ask",
        1,
        "judge-b",
        [("clear", False, "Too vague."), ("safe", True, "Kept it safe.")],
        directory,
    )
    _store_grades(
        "ask",
        2,
        "judge-a",
        [("clear", True, "Second attempt is clear.")],
        directory,
    )
    _store_grades(
        "ask",
        2,
        "judge-b",
        [("clear", False, "Second attempt is vague.")],
        directory,
    )
    _store_grades(
        "other-task",
        1,
        "judge-a",
        [("hidden", True, "NOT_IN_SHOW")],
        directory,
    )

    shown = CLI.invoke(app, ["show", "fake-run", "ask"])
    assert shown.exit_code == 0, shown.output
    assert shown.stdout.count("attempt 1\n") == 1
    assert shown.stdout.count("attempt 2\n") == 1
    assert "NOT_IN_SHOW" not in shown.stdout
    assert "first judge summary" not in shown.stdout

    blocks = _attempt_blocks(shown.stdout)
    assert "score: 0.40" in blocks[1]
    assert "second judge summary" in blocks[1]
    assert blocks[1].index("WINDOW_REPLY") < blocks[1].index("judge: judge-a")
    assert blocks[1].index("judge: judge-a") < blocks[1].index("judge: judge-b")
    assert _grade_rows(blocks[1], "judge-a") == [
        ("clear", "yes", "Clear enough."),
        ("safe", "no", "Gave a secret."),
    ]
    assert _grade_rows(blocks[1], "judge-b") == [
        ("clear", "no", "Too vague."),
        ("safe", "yes", "Kept it safe."),
    ]
    assert blocks[2].index("OTHER_REPLY") < blocks[2].index("judge: judge-a")
    assert _grade_rows(blocks[2], "judge-a") == [("clear", "yes", "Second attempt is clear.")]
    assert _grade_rows(blocks[2], "judge-b") == [("clear", "no", "Second attempt is vague.")]


def _stored_attempt(
    task_id: str,
    attempt: int,
    output: str,
    *,
    score: float,
    reason: str,
) -> TaskResult:
    return TaskResult(
        run_id="fake-run",
        task_id=task_id,
        attempt=attempt,
        category="support",
        model="mock",
        output=output,
        score=score,
        passed=score >= 0.6,
        reason=reason,
        latency_ms=12,
        judge="judge-a",
        rubric_id="demo",
        rubric_version=1,
        rubric_hash="hash",
    )


def _store_grades(
    task_id: str,
    attempt: int,
    judge: str,
    rows: list[tuple[str, bool, str]],
    directory: Path,
) -> None:
    replace_criterion_grades(
        "fake-run",
        task_id,
        attempt,
        judge,
        rubric_id="demo",
        rubric_version=1,
        rubric_hash="hash",
        marks=[
            CriterionMark(criterion_id=criterion_id, met=met, reason=reason)
            for criterion_id, met, reason in rows
        ],
        directory=directory,
    )


def _attempt_blocks(text: str) -> dict[int, str]:
    matches = list(re.finditer(r"^attempt (\d+)$", text, re.MULTILINE))
    blocks: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks[int(match.group(1))] = text[match.end() : end]
    return blocks


def _grade_rows(block: str, judge: str) -> list[tuple[str, str, str]]:
    body = block.split(f"judge: {judge}", 1)[1].split("\njudge: ", 1)[0]
    lines = [line for line in body.splitlines() if line.strip()]
    assert lines[0].split()[:3] == ["criterion_id", "met", "reason"]
    rows: list[tuple[str, str, str]] = []
    for line in lines[1:]:
        criterion_id, met, reason = line.split(None, 2)
        rows.append((criterion_id, met, reason))
    return rows


def test_config_loads_and_cli_flags_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = resolve_agents(
        None,
        provider=None,
        model=None,
        temperature=None,
        judge_provider=None,
        judge_model=None,
    )
    assert resolved.provider == "mock"
    assert resolved.judge_provider is None
    assert resolved.judge_temperature == 0

    config = tmp_path / "agents.yaml"
    config.write_text(
        """
answer_agent:
  provider: ollama
  model: llama3.2
  temperature: 0.2
  system_prompt: answer as a clerk
grader_agent:
  provider: ollama
  model: qwen2.5
  temperature: 0.1
  system_prompt: be strict
""",
        encoding="utf-8",
    )
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        '{"id": "t", "category": "c", "prompt": "hi", "scorer": "exact_match", "expected": "hi"}\n',
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def create(name: str, **kwargs: object) -> MockProvider:
        seen.setdefault("create", []).append((name, kwargs))
        return MockProvider()

    def judge(name: str, **kwargs: object) -> MockProvider:
        seen.setdefault("judge", []).append((name, kwargs))
        return MockProvider(name="judge")

    def answer(dataset_path: object, provider: object, run_id: str, **kwargs: object) -> None:
        del dataset_path, provider, kwargs
        from evalkit.storage import upsert_run

        upsert_run(run_id, status="answered", model="mock", attempts=1)

    def grade(run_id: str, **kwargs: object) -> list[object]:
        del run_id
        seen["grade"] = kwargs
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalkit.cli.create_provider", create)
    monkeypatch.setattr("evalkit.cli.create_judge_provider", judge)
    monkeypatch.setattr("evalkit.cli.answer_run", answer)
    monkeypatch.setattr("evalkit.cli.grade_run", grade)
    loaded = CLI.invoke(
        app,
        ["run", "--dataset", str(dataset), "--config", str(config), "--run-id", "cfg", "--attempts", "1"],
    )
    assert loaded.exit_code == 0, loaded.output
    assert seen["create"] == [
        (
            "ollama",
            {"model": "llama3.2", "temperature": 0.2, "system": "answer as a clerk"},
        )
    ]
    assert seen["judge"] == [
        (
            "ollama",
            {"model": "qwen2.5", "temperature": 0.1, "system": "be strict"},
        )
    ]
    grade_kwargs = seen["grade"]
    assert isinstance(grade_kwargs, dict)
    assert grade_kwargs["grader_context"] == "with_prompt"

    overridden = CLI.invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--config",
            str(config),
            "--provider",
            "mock",
            "--model",
            "override",
            "--temperature",
            "0.9",
            "--judge-provider",
            "openai",
            "--judge-model",
            "gpt-4o-mini",
            "--grader-context",
            "response_only",
            "--run-id",
            "over",
            "--attempts",
            "1",
        ],
    )
    assert overridden.exit_code == 0, overridden.output
    assert seen["create"][-1] == (
        "mock",
        {"model": "override", "temperature": 0.9, "system": "answer as a clerk"},
    )
    assert seen["judge"][-1] == (
        "openai",
        {"model": "gpt-4o-mini", "temperature": 0.1, "system": "be strict"},
    )
    grade_kwargs = seen["grade"]
    assert isinstance(grade_kwargs, dict)
    assert grade_kwargs["grader_context"] == "response_only"

    missing = CLI.invoke(
        app,
        ["run", "--dataset", str(dataset), "--config", str(tmp_path / "missing.yaml"), "--run-id", "nope"],
    )
    assert missing.exit_code == 1
    assert "config not found" in missing.output

    rejected = CLI.invoke(
        app,
        ["run", "--dataset", str(dataset), "--grader-context", "both", "--run-id", "bad"],
    )
    assert rejected.exit_code == 1
    assert "grader context" in rejected.output


def _write_dataset(
    tmp_path: Path,
    *,
    prompt: str = "How long is the window?",
    prompts: tuple[tuple[str, str], ...] | None = None,
) -> tuple[Path, Path]:
    rubric_dir = tmp_path / "rubrics"
    rubric_dir.mkdir()
    payload = {
        "id": "demo",
        "version": 1,
        "description": "A short reply.",
        "pass_threshold": 0.5,
        "criteria": [
            {
                "id": "clear",
                "weight": 0.75,
                "necessity": "explicit",
                "kind": "subjective",
                "description": "The reply is clear.",
                "example": "It answers in one sentence.",
            },
            {
                "id": "safe",
                "weight": 0.25,
                "necessity": "implicit",
                "kind": "objective",
                "critical": True,
                "description": "The reply does not ask for a password.",
                "answer_key": "Fails if the reply asks for a password.",
            },
        ],
    }
    (rubric_dir / "demo.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    rows = prompts or (("only", prompt),)
    lines = [
        json.dumps(
            {
                "id": task_id,
                "category": "support",
                "prompt": text,
                "rubric_id": "demo",
            }
        )
        for task_id, text in rows
    ]
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rubric_dir, dataset
