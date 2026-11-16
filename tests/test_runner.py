import sqlite3
import threading
import time
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from evalkit.cli import _summary_table, app
from evalkit.models import Task, TaskResult
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
        _task("two-plus-two"),
        _task("capital-of-france", expected="Paris"),
        _task("missing-id"),
    ]
    results = run_eval(tasks, MockProvider(), "run-scores", repeats=1)

    assert [result.task_id for result in results] == [
        "two-plus-two",
        "capital-of-france",
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

    assert passed == {"two-plus-two", "primary-colors", "name-and-city", "add-function"}
    assert failed == {"capital-of-france", "closest-planets", "keys-a-and-b", "add-function-arguments"}


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
        [_task("two-plus-two", scorer="not-a-scorer")],
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
    assert all(result.output_varies for result in results)
    assert results[1].failure_type == "wrong answer"
    assert results[2].failure_type == "wrong answer"


def test_run_eval_keeps_stable_answers_unflagged() -> None:
    results = run_eval([_task("two-plus-two")], MockProvider(), "stable", repeats=3)
    assert len(results) == 3
    assert all(result.passed for result in results)
    assert all(not result.unstable for result in results)
    assert all(not result.output_varies for result in results)
    assert all(result.failure_type is None for result in results)


def test_output_text_can_vary_without_marking_the_task_unstable() -> None:
    class Wording(MockProvider):
        def __init__(self) -> None:
            super().__init__({})
            self.calls = 0

        def generate(self, prompt: str, *, task_id: str | None = None, timeout: float | None = None) -> str:
            del prompt, task_id, timeout
            self.calls += 1
            return "4" if self.calls == 1 else "4."

    results = run_eval([_task("wording")], Wording(), "wording", retries=1, repeats=2)
    assert all(result.passed for result in results)
    assert all(not result.unstable for result in results)
    assert all(result.output_varies for result in results)


def test_one_attempt_skips_stability_flags(tmp_path: Path) -> None:
    class Vary(MockProvider):
        def __init__(self) -> None:
            super().__init__({})
            self.calls = 0

        def generate(self, prompt: str, *, task_id: str | None = None) -> str:
            del prompt, task_id
            self.calls += 1
            return "4" if self.calls == 1 else "5"

    provider = Vary()
    results = run_eval([_task("two-plus-two")], provider, "once", retries=1, repeats=1)
    assert provider.calls == 1
    assert len(results) == 1
    assert results[0].attempt == 1
    assert results[0].unstable is False
    assert results[0].output_varies is False
    path = save_results("once", results, directory=tmp_path, attempts=1)
    row = sqlite3.connect(path).execute("SELECT attempts FROM runs WHERE run_id = 'once'").fetchone()
    assert row == (1,)


def test_run_eval_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_eval([_task("two-plus-two")], MockProvider(), "zero", repeats=0)


def test_ollama_uses_one_worker_and_cancels_through_http_timeout() -> None:
    class LocalOllama:
        name = "ollama"
        uses_http_timeout = True
        default_timeout = 120.0
        default_workers = 1

        def __init__(self) -> None:
            self.timeouts: list[float | None] = []
            self.peak = 0
            self.current = 0
            self._lock = threading.Lock()

        def generate(
            self,
            prompt: str,
            *,
            task_id: str | None = None,
            timeout: float | None = None,
        ) -> str:
            del prompt, task_id
            self.timeouts.append(timeout)
            with self._lock:
                self.current += 1
                self.peak = max(self.peak, self.current)
            try:
                time.sleep(0.2)
            finally:
                with self._lock:
                    self.current -= 1
            return "4"

    provider = LocalOllama()
    results = run_eval(
        [_task("two-plus-two"), _task("capital-of-france", expected="4")],
        provider,
        "serial",
        retries=1,
        repeats=1,
    )
    assert provider.peak == 1
    assert provider.timeouts == [120.0, 120.0]
    assert all(result.passed for result in results)

    widened = LocalOllama()
    run_eval(
        [_task("two-plus-two")],
        widened,
        "wider",
        retries=1,
        repeats=1,
        workers=2,
        timeout=15,
    )
    assert widened.timeouts == [15]


def test_ollama_chat_uses_options_format_and_the_env_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"message": {"role": "assistant", "content": "4"}}

    class FakeClient:
        def __init__(self, timeout: float | None = None) -> None:
            captured["client_timeout"] = timeout

        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(
            self,
            url: str,
            json: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeResponse:
            del headers
            captured["url"] = url
            captured["payload"] = json
            captured["post_timeout"] = timeout
            return FakeResponse()

    monkeypatch.setenv("OLLAMA_URL", "http://ollama.test:11434/")
    monkeypatch.setattr(httpx, "Client", FakeClient)
    provider = OllamaProvider(system="Be brief.", temperature=0.0, json_format=True)
    assert provider.base_url == "http://ollama.test:11434"
    assert provider.name == "llama3.2"
    answer = provider.generate("What is 2 + 2?", timeout=120)
    payload = captured["payload"]
    assert answer == "4"
    assert captured["url"] == "http://ollama.test:11434/api/chat"
    assert captured["client_timeout"] == 120
    assert isinstance(payload, dict)
    assert payload["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    assert payload["options"] == {"num_predict": 512, "temperature": 0.0}
    assert payload["format"] == "json"
    assert "prompt" not in payload


def test_ollama_check_model_available_reads_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, body: dict[str, object]) -> None:
            self._body = body

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._body

    class FakeClient:
        def __init__(self, timeout: float | None = None) -> None:
            self.timeout = timeout
            self.urls: list[str] = []

        def get(self, url: str, timeout: float | None = None) -> FakeResponse:
            del timeout
            self.urls.append(url)
            if "missing" in url:
                return FakeResponse({"models": [{"name": "mistral:latest"}]})
            return FakeResponse({"models": [{"name": "llama3.2:latest"}]})

    client = FakeClient()
    OllamaProvider(client=client).check_model_available()
    assert client.urls == ["http://localhost:11434/api/tags"]

    missing = OllamaProvider(base_url="http://missing", client=FakeClient())
    with pytest.raises(ProviderError, match="run: ollama pull llama3.2"):
        missing.check_model_available()

    class Down(FakeClient):
        def get(self, url: str, timeout: float | None = None) -> FakeResponse:
            del url, timeout
            raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError, match="could not reach Ollama"):
        OllamaProvider(client=Down()).check_model_available()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    created = create_provider(
        "ollama",
        model="llama3.1",
        temperature=0.2,
        system="Be brief.",
        json_format=True,
    )
    assert isinstance(created, OllamaProvider)
    assert created.model == "llama3.1"
    assert created.name == "llama3.1"
    assert created.temperature == 0.2
    assert created.json_format is True


def test_run_eval_records_the_ollama_model_name(tmp_path: Path) -> None:
    class FakeResponse:
        def __init__(self, body: dict[str, object]) -> None:
            self._body = body

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._body

    class FakeClient:
        def __init__(self) -> None:
            self.order: list[str] = []

        def get(self, url: str, timeout: float | None = None) -> FakeResponse:
            del url, timeout
            self.order.append("tags")
            return FakeResponse({"models": [{"model": "llama3.2:latest"}]})

        def post(
            self,
            url: str,
            json: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeResponse:
            del url, headers, timeout
            self.order.append("chat")
            assert json is not None
            assert json["model"] == "llama3.2"
            assert json["options"] == {"num_predict": 512}
            assert "format" not in json
            return FakeResponse({"message": {"content": "4"}})

    client = FakeClient()
    results = run_eval(
        [_task("two-plus-two")],
        OllamaProvider(client=client),
        "ollama-run",
        retries=1,
        repeats=1,
    )
    assert client.order == ["tags", "chat"]
    assert results[0].model == "llama3.2"
    assert results[0].output == "4"
    path = save_results("ollama-run", results, directory=tmp_path)
    row = sqlite3.connect(path).execute(
        "SELECT model FROM runs WHERE run_id = 'ollama-run'"
    ).fetchone()
    assert row is not None
    assert row[0] == "llama3.2"


def test_ollama_http_timeout_is_reported() -> None:
    class SlowClient:
        def post(
            self,
            url: str,
            json: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            timeout: float | None = None,
        ) -> object:
            del url, json, headers, timeout
            raise httpx.ReadTimeout("read timed out")

    with pytest.raises(TimeoutError, match="timed out after 120"):
        OllamaProvider(client=SlowClient()).generate("hi", timeout=120)


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
    results = run_eval(
        [_task("two-plus-two"), _task("capital-of-france", expected="Paris")],
        MockProvider(),
        "persist",
    )
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
    assert "tasks: 8  attempts: 24  passed attempts: 12  tasks passing all attempts: 4" in completed.stdout
    assert (tmp_path / "runs" / "evalkit.db").is_file()

    reported = CLI.invoke(app, ["report", "cli-run"])
    assert reported.exit_code == 0
    assert "run_id: cli-run" in reported.stdout
    assert "add-function-arguments" in reported.stdout
    header = next(
        line
        for line in reported.stdout.splitlines()
        if line.split()[:4] == ["task", "attempt", "result", "score"]
    )
    assert "flags" in header.split()
    assert "failure" in header.split()

    shown = CLI.invoke(app, ["show", "cli-run", "two-plus-two"])
    assert shown.exit_code == 0
    assert "attempt 1" in shown.stdout
    assert "attempt 3" in shown.stdout
    assert "score: 1.00" in shown.stdout
    assert "output:\n4" in shown.stdout

    missing = CLI.invoke(app, ["show", "cli-run", "not-a-task"])
    assert missing.exit_code == 1


def test_summary_table_keeps_each_attempt_on_one_line() -> None:
    reason = "because " * 40
    table = _summary_table(
        [
            TaskResult(
                run_id="r",
                task_id="two-plus-two",
                category="math",
                model="llama3.2",
                output="4\nstill 4",
                score=1,
                passed=True,
                reason=reason,
                latency_ms=12.4,
                attempt=1,
                output_varies=True,
            )
        ]
    )
    lines = [line for line in table.splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[1].count("two-plus-two") == 1
    assert "output varies" in lines[1]
    assert "because" in lines[1]


def test_cli_forwards_model_temperature_and_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def create(name: str, **kwargs: object) -> MockProvider:
        seen["create"] = (name, kwargs)
        return MockProvider()

    def run(tasks: object, provider: object, run_id: str, **kwargs: object) -> list[TaskResult]:
        del tasks, provider
        seen["run"] = kwargs
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evalkit.cli.create_provider", create)
    monkeypatch.setattr("evalkit.cli.run_eval", run)
    completed = CLI.invoke(
        app,
        [
            "run",
            "--dataset",
            str(DATASET),
            "--provider",
            "ollama",
            "--model",
            "llama3.1",
            "--temperature",
            "0.2",
            "--workers",
            "2",
            "--run-id",
            "opts",
        ],
    )
    assert completed.exit_code == 0
    assert seen["create"] == (
        "ollama",
        {"model": "llama3.1", "temperature": 0.2},
    )
    assert seen["run"]["workers"] == 2
    assert seen["run"]["judge"] is None
    assert seen["run"]["repeats"] == 3

    once = CLI.invoke(
        app,
        [
            "run",
            "--dataset",
            str(DATASET),
            "--provider",
            "mock",
            "--attempts",
            "1",
            "--run-id",
            "once",
        ],
    )
    assert once.exit_code == 0
    assert seen["run"]["repeats"] == 1

    rejected = CLI.invoke(
        app,
        ["run", "--dataset", str(DATASET), "--provider", "mock", "--attempts", "0"],
    )
    assert rejected.exit_code != 0


def test_cli_stops_when_the_ollama_model_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"models": []}

    class FakeClient:
        def __init__(self, timeout: float | None = None) -> None:
            del timeout

        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(self, url: str, timeout: float | None = None) -> FakeResponse:
            del url, timeout
            return FakeResponse()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(httpx, "Client", FakeClient)
    completed = CLI.invoke(
        app,
        ["run", "--dataset", str(DATASET), "--provider", "ollama", "--run-id", "missing-model"],
    )
    assert completed.exit_code == 1
    assert "run: ollama pull llama3.2" in completed.stdout + completed.stderr
    assert not (tmp_path / "runs" / "evalkit.db").is_file()


def test_old_database_gains_a_model_column(tmp_path: Path) -> None:
    database = tmp_path / "evalkit.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY)")
    connection.execute(
        """
        CREATE TABLE results (
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
            PRIMARY KEY (run_id, task_id, attempt)
        )
        """
    )
    connection.execute("INSERT INTO runs (run_id) VALUES ('old')")
    connection.execute(
        """
        INSERT INTO results (
            run_id, task_id, attempt, category, model, output, score, passed,
            reason, latency_ms, failure_type, unstable
        ) VALUES ('old', 'two-plus-two', 1, 'math', 'llama3.2', '4', 1, 1, 'ok', 3, NULL, 0)
        """
    )
    connection.commit()
    connection.close()

    loaded = load_results("old", directory=tmp_path)
    assert loaded[0].model == "llama3.2"
    assert loaded[0].output_varies is False
    row = sqlite3.connect(database).execute("SELECT model FROM runs WHERE run_id = 'old'").fetchone()
    assert row == ("llama3.2",)
