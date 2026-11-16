import re
from pathlib import Path

from typer.testing import CliRunner

from evalkit.cli import app
from evalkit.models import Task
from evalkit.pipeline import answer_run, grade_run
from evalkit.providers import MockProvider
from evalkit.runner import run_eval
from evalkit.storage import answer_keys, load_answers, load_results, load_run

TASKS = """\
{"id": "two-plus-two", "category": "math", "prompt": "2+2", "scorer": "exact_match", "expected": "4"}
{"id": "capital-of-france", "category": "knowledge", "prompt": "capital", "scorer": "exact_match", "expected": "Paris"}
"""


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text(TASKS, encoding="utf-8")
    return path


def _signature(results: list) -> list[tuple[object, ...]]:
    return sorted(
        (
            result.task_id,
            result.attempt,
            result.output,
            result.score,
            result.passed,
            result.reason,
            result.failure_type,
            result.unstable,
            result.output_varies,
        )
        for result in results
    )


def test_answers_are_saved_before_the_next_attempt(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    directory = tmp_path / "runs"
    seen: list[int] = []

    class Watch(MockProvider):
        def generate(
            self,
            prompt: str,
            *,
            task_id: str | None = None,
            timeout: float | None = None,
        ) -> str:
            seen.append(len(answer_keys("incremental", directory=directory)))
            return super().generate(prompt, task_id=task_id, timeout=timeout)

    lines: list[str] = []
    answer_run(
        dataset,
        Watch(),
        "incremental",
        attempts=2,
        workers=1,
        directory=directory,
        provider_name="mock",
        on_progress=lines.append,
    )
    assert seen == [0, 1, 2, 3]
    assert load_run("incremental", directory=directory).status == "answered"
    assert len(load_answers("incremental", directory=directory)) == 4
    assert lines[0] == "[1/4] two-plus-two attempt 1  " + lines[0].rsplit("  ", 1)[1]
    assert re.fullmatch(r"\[\d/4\] \S+ attempt \d  \d+\.\d+s", lines[0])


def test_resume_skips_answers_that_are_already_saved(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    directory = tmp_path / "runs"

    class Stop(MockProvider):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def generate(
            self,
            prompt: str,
            *,
            task_id: str | None = None,
            timeout: float | None = None,
        ) -> str:
            self.calls += 1
            if self.calls > 1:
                raise KeyboardInterrupt
            return super().generate(prompt, task_id=task_id, timeout=timeout)

    with_interrupt = Stop()
    try:
        answer_run(
            dataset,
            with_interrupt,
            "partial",
            attempts=1,
            workers=1,
            directory=directory,
            provider_name="mock",
        )
    except KeyboardInterrupt:
        interrupted = True
    else:
        interrupted = False
    assert interrupted
    assert load_run("partial", directory=directory).status == "interrupted"
    assert answer_keys("partial", directory=directory) == {("two-plus-two", 1)}

    resumed = MockProvider()
    calls = {"n": 0}

    def generate(
        prompt: str,
        *,
        task_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        calls["n"] += 1
        return MockProvider.generate(resumed, prompt, task_id=task_id, timeout=timeout)

    resumed.generate = generate  # type: ignore[method-assign]
    answer_run(
        dataset,
        resumed,
        "partial",
        resume=True,
        workers=1,
        directory=directory,
    )
    assert calls["n"] == 1
    assert answer_keys("partial", directory=directory) == {
        ("two-plus-two", 1),
        ("capital-of-france", 1),
    }
    assert load_run("partial", directory=directory).status == "answered"


def test_regrading_keeps_both_judges(tmp_path: Path) -> None:
    dataset = tmp_path / "judge.jsonl"
    dataset.write_text(
        '{"id": "tone", "category": "writing", "prompt": "Say hello", '
        '"scorer": "llm_judge", "expected": "Be brief"}\n',
        encoding="utf-8",
    )
    directory = tmp_path / "runs"
    answer_run(
        dataset,
        MockProvider({"tone": "Hello"}),
        "judged",
        attempts=1,
        workers=1,
        directory=directory,
        provider_name="mock",
    )

    class Judge:
        def __init__(self, name: str, score: int) -> None:
            self.name = name
            self.score = score

        def generate(
            self,
            prompt: str,
            *,
            task_id: str | None = None,
            timeout: float | None = None,
        ) -> str:
            del prompt, task_id, timeout
            return f'{{"score": {self.score}, "reason": "{self.name}"}}'

    grade_run("judged", judge=Judge("judge-a", 5), directory=directory)
    grade_run("judged", judge=Judge("judge-b", 1), directory=directory)
    first = load_results("judged", directory=directory, judge="judge-a")
    second = load_results("judged", directory=directory, judge="judge-b")
    assert first[0].score == 1.0
    assert first[0].passed is True
    assert first[0].judge == "judge-a"
    assert second[0].score == 0.0
    assert second[0].passed is False
    assert second[0].judge == "judge-b"
    assert load_run("judged", directory=directory).status == "graded"


def test_cli_answer_then_grade(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _dataset(tmp_path)
    runner = CliRunner()
    answered = runner.invoke(
        app,
        [
            "answer",
            "--dataset",
            str(dataset),
            "--provider",
            "mock",
            "--attempts",
            "1",
            "--run-id",
            "cli-answer",
        ],
    )
    assert answered.exit_code == 0, answered.stdout + answered.stderr
    assert "[1/2] two-plus-two attempt 1" in answered.stdout
    assert "[2/2] capital-of-france attempt 1" in answered.stdout

    graded = runner.invoke(app, ["grade", "cli-answer"])
    assert graded.exit_code == 0, graded.stdout + graded.stderr
    assert "tasks: 2  attempts: 2  passed attempts: 1" in graded.stdout
    assert load_run("cli-answer").status == "graded"


def test_answer_then_grade_matches_run_eval(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    tasks = [
        Task.model_validate_json(line)
        for line in TASKS.splitlines()
    ]
    direct = run_eval(tasks, MockProvider(), "memory", repeats=2, retries=1)
    directory = tmp_path / "runs"
    answer_run(
        dataset,
        MockProvider(),
        "split",
        attempts=2,
        workers=2,
        retries=1,
        directory=directory,
        provider_name="mock",
    )
    grade_run("split", directory=directory)
    stored = load_results("split", directory=directory)
    assert _signature(stored) == _signature(direct)
    assert load_run("split", directory=directory).status == "graded"
