import threading
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalkit.cli import app
from evalkit.models import Task
from evalkit.providers import (
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    create_anthropic_provider,
    create_openai_provider,
    create_provider,
)
from evalkit.runner import MAX_WORKERS, load_tasks, run_eval
from evalkit.storage import load_results, save_results

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "sample_tasks.jsonl"
CLI = CliRunner()


def _task(task_id: str, scorer: str = "exact_match", expected: object = "4") -> Task:
    return Task(
        id=task_id,
        category="math",
        prompt=f"Answer {task_id}",
        scorer=scorer,
        expected=expected,
    )


def test_run_eval_scores_right_and_wrong_canned_answers() -> None:
    tasks = [
        _task("exact-right"),
        _task("exact-wrong", expected="Paris"),
        _task("missing-id"),
    ]
    results = run_eval(tasks, MockProvider(), "run-scores", repeats=1)

    assert [result.task_id for result in results] == [
        "exact-right",
        "exact-wrong",
        "missing-id",
    ]
    assert results[0].passed is True
    assert results[0].score == 1.0
    assert results[0].output == "4"
    assert results[1].passed is False
    assert results[1].output == "Lyon"
    assert results[2].passed is False
    assert results[2].output == "I don't know"
    assert {result.run_id for result in results} == {"run-scores"}
    assert {result.model for result in results} == {"mock"}
    assert all(result.latency_ms >= 0 for result in results)


def test_run_eval_loads_the_sample_dataset() -> None:
    results = run_eval(DATASET, MockProvider(), "sample")
    passed = {result.task_id for result in results if result.passed}
    failed = {result.task_id for result in results if not result.passed}

    assert passed == {"exact-right", "contains-right", "json-right", "python-right"}
    assert failed == {"exact-wrong", "contains-wrong", "json-wrong", "python-wrong"}


def test_load_tasks_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "\n"
        '{"id": "a", "category": "math", "prompt": "2+2", '
        '"scorer": "exact_match", "expected": "4"}\n'
        "\n",
        encoding="utf-8",
    )
    loaded = load_tasks(path)
    assert [task.id for task in loaded] == ["a"]


def test_run_eval_preserves_order_when_the_first_task_is_slower() -> None:
    class SlowFirst(MockProvider):
        def generate(self, prompt: str, *, task_id: str | None = None) -> str:
            if task_id == "first":
                time.sleep(0.2)
            return super().generate(prompt, task_id=task_id)

    tasks = [_task("first"), _task("second")]
    provider = SlowFirst({"first": "4", "second": "4"})
    results = run_eval(tasks, provider, "order", retries=1, timeout=5, repeats=1)
    assert [result.task_id for result in results] == ["first", "second"]
    assert all(result.passed for result in results)


def test_run_eval_retries_then_scores_the_successful_output() -> None:
    class Flaky(MockProvider):
        def __init__(self) -> None:
            super().__init__({"t1": "  4!\n"})
            self.calls = 0

        def generate(self, prompt: str, *, task_id: str | None = None) -> str:
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporary")
            return super().generate(prompt, task_id=task_id)

    provider = Flaky()
    results = run_eval([_task("t1")], provider, "retry", retries=3, timeout=2, repeats=1)
    assert provider.calls == 3
    assert results[0].passed is True
    assert results[0].output == "  4!\n"


def test_run_eval_records_provider_errors() -> None:
    class Down(MockProvider):
        def __init__(self) -> None:
            super().__init__({})
            self.calls = 0

        def generate(self, prompt: str, *, task_id: str | None = None) -> str:
            self.calls += 1
            raise RuntimeError("down")

    provider = Down()
    results = run_eval([_task("t1")], provider, "down", retries=2, timeout=2, repeats=1)
    assert provider.calls == 2
    assert results[0].passed is False
    assert results[0].score == 0.0
    assert "down" in results[0].reason


def test_run_eval_records_timeouts() -> None:
    class Hung(MockProvider):
        def __init__(self) -> None:
            super().__init__({})
            self.calls = 0

        def generate(self, prompt: str, *, task_id: str | None = None) -> str:
            self.calls += 1
            time.sleep(5)
            return "4"

    provider = Hung()
    results = run_eval([_task("t1")], provider, "timeout", retries=2, timeout=0.2, repeats=1)
    assert provider.calls == 2
    assert results[0].passed is False
    assert results[0].failure_type == "timeout"
    assert "timed out" in results[0].reason
    assert results[0].latency_ms < 3000


def test_run_eval_uses_at_most_four_workers() -> None:
    class Tracking(MockProvider):
        def __init__(self) -> None:
            super().__init__({f"t{index}": "4" for index in range(8)})
            self._lock = threading.Lock()
            self.current = 0
            self.peak = 0

        def generate(self, prompt: str, *, task_id: str | None = None) -> str:
            with self._lock:
                self.current += 1
                self.peak = max(self.peak, self.current)
            try:
                time.sleep(0.3)
                return super().generate(prompt, task_id=task_id)
            finally:
                with self._lock:
                    self.current -= 1

    provider = Tracking()
    tasks = [_task(f"t{index}") for index in range(8)]
    results = run_eval(tasks, provider, "pool", retries=1, timeout=5, repeats=1)
    assert len(results) == 8
    assert all(result.passed for result in results)
    assert 2 <= provider.peak <= MAX_WORKERS


class _Judge:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.name = "judge"

    def generate(self, prompt: str, *, task_id: str | None = None) -> str:
        del task_id
        self.prompts.append(prompt)
        return self.reply


def test_run_eval_asks_the_judge_and_stores_the_scaled_score() -> None:
    judge = _Judge('{"score": 4, "reason": "meets the rubric"}')
    task = _task("judged", scorer="llm_judge", expected="digits only")
    results = run_eval([task], MockProvider(), "judged-run", judge=judge, repeats=1)
    assert results[0].output == "I don't know"
    assert results[0].score == 0.75
    assert results[0].passed is True
    assert results[0].reason == "meets the rubric"
    assert "Answer judged" in judge.prompts[0]
    assert "Do not reward length." in judge.prompts[0]


def test_run_eval_records_a_missing_judge() -> None:
    results = run_eval(
        [_task("judged", scorer="llm_judge", expected="digits only")],
        MockProvider({"judged": "4"}),
        "no-judge",
        repeats=1,
    )
    assert results[0].output == "4"
    assert results[0].passed is False
    assert results[0].reason == "llm_judge requires a judge provider"


def test_run_eval_reports_an_unknown_scorer() -> None:
    results = run_eval(
        [_task("exact-right", scorer="not-a-scorer")],
        MockProvider(),
        "bad-scorer",
        repeats=1,
    )
    assert results[0].output == "4"
    assert results[0].passed is False
    assert results[0].reason == "unknown scorer: not-a-scorer"


def test_run_eval_flags_answers_that_change_across_attempts() -> None:
    class Vary(MockProvider):
        def __init__(self) -> None:
            super().__init__({})
            self.calls = 0

        def generate(self, prompt: str, *, task_id: str | None = None) -> str:
            self.calls += 1
            return "4" if self.calls == 1 else "5"

    results = run_eval([_task("t1")], Vary(), "unstable", retries=1, repeats=3)
    assert [result.attempt for result in results] == [1, 2, 3]
    assert all(result.unstable for result in results)
    assert results[0].passed is True
    assert results[0].failure_type is None
    assert results[1].failure_type == "wrong answer"
    assert results[2].failure_type == "wrong answer"


def test_run_eval_keeps_stable_answers_unflagged() -> None:
    results = run_eval([_task("exact-right")], MockProvider(), "stable", repeats=3)
    assert len(results) == 3
    assert all(result.passed for result in results)
    assert all(not result.unstable for result in results)
    assert all(result.failure_type is None for result in results)


def test_run_eval_labels_refusal_and_format_failures() -> None:
    refused = run_eval([_task("missing")], MockProvider({}), "refuse", repeats=1)
    assert refused[0].output == "I don't know"
    assert refused[0].failure_type == "refusal"

    malformed = run_eval(
        [_task("json", scorer="json_valid", expected={"required_keys": ["a"]})],
        MockProvider({"json": "not json"}),
        "format",
        repeats=1,
    )
    assert malformed[0].failure_type == "format error"


def test_save_and_load_results_roundtrip(tmp_path: Path) -> None:
    results = run_eval([_task("exact-right"), _task("exact-wrong", expected="Paris")], MockProvider(), "persist")
    path = save_results("persist", results, directory=tmp_path)
    assert path == tmp_path / "evalkit.db"
    assert load_results("persist", directory=tmp_path) == results


def test_storage_rejects_a_run_id_that_escapes_the_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_results("../outside", [], directory=tmp_path)


def test_remote_providers_are_not_created_without_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert create_openai_provider() is None
    assert create_anthropic_provider() is None
    with pytest.raises(ProviderError):
        create_provider("openai")
    with pytest.raises(ProviderError):
        create_provider("anthropic")


def test_create_provider_passes_a_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    openai = create_provider("openai", model="gpt-4o")
    anthropic = create_provider("anthropic", model="claude-3-5-haiku-latest")
    ollama = create_provider("ollama", model="llama3.1")
    assert isinstance(openai, OpenAIProvider)
    assert openai.model == "gpt-4o"
    assert isinstance(anthropic.model, str)
    assert anthropic.model == "claude-3-5-haiku-latest"
    assert isinstance(ollama, OllamaProvider)
    assert ollama.model == "llama3.1"


def test_cli_run_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    completed = CLI.invoke(
        app,
        ["run", "--dataset", str(DATASET), "--provider", "mock", "--run-id", "cli-run"],
    )
    assert completed.exit_code == 0
    assert "run_id: cli-run" in completed.stdout
    assert "passed: 12" in completed.stdout
    assert (tmp_path / "runs" / "evalkit.db").is_file()

    reported = CLI.invoke(app, ["report", "cli-run"])
    assert reported.exit_code == 0
    assert "run_id: cli-run" in reported.stdout
    assert "python-wrong" in reported.stdout
