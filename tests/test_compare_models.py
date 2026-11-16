import importlib.util
from pathlib import Path

import pytest

from evalkit.providers import MockProvider

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_models.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("compare_models", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset(path: Path, scorer: str = "exact_match") -> None:
    path.write_text(
        '{"id": "t", "category": "math", "prompt": "What is 2 + 2?", '
        f'"scorer": "{scorer}", "expected": "4"}}\n',
        encoding="utf-8",
    )


def test_script_saves_both_runs_and_compares_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = tmp_path / "tasks.jsonl"
    _dataset(dataset)
    script = _load_script()

    def fake_create(name: str, *, answers: dict[str, str] | None = None, model: str | None = None) -> MockProvider:
        del answers, model
        canned = {"t": "4"} if name == "openai" else {"t": "0"}
        return MockProvider(canned, name=name)

    monkeypatch.setattr(script, "create_provider", fake_create)
    code = script.main(
        ["--dataset", str(dataset), "--a", "openai", "--b", "anthropic", "--runs-dir", str(tmp_path / "runs")]
    )
    assert code == 0
    saved = list((tmp_path / "runs").glob("*.jsonl"))
    assert len(saved) == 2
    output = capsys.readouterr().out
    assert "openai pass_rate 1.00" in output
    assert "anthropic pass_rate 0.00" in output
    assert "difference 1.00" in output
    assert "excludes_zero True" in output


def test_script_requires_a_judge_before_calling_the_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = tmp_path / "tasks.jsonl"
    _dataset(dataset, scorer="llm_judge")
    script = _load_script()

    def fail_create(*args: object, **kwargs: object) -> MockProvider:
        del args, kwargs
        raise AssertionError("providers should not be built without a judge")

    monkeypatch.setattr(script, "create_provider", fail_create)
    code = script.main(["--dataset", str(dataset), "--runs-dir", str(tmp_path / "runs")])
    assert code == 1
    assert "pass --judge" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_script_labels_two_models_from_the_same_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = tmp_path / "tasks.jsonl"
    _dataset(dataset)
    script = _load_script()
    calls: list[tuple[str, str | None]] = []

    def fake_create(name: str, *, answers: dict[str, str] | None = None, model: str | None = None) -> MockProvider:
        del answers
        calls.append((name, model))
        reply = "4" if model == "gpt-4o" else "0"
        return MockProvider({"t": reply}, name=name)

    monkeypatch.setattr(script, "create_provider", fake_create)
    code = script.main(
        [
            "--dataset",
            str(dataset),
            "--a",
            "openai",
            "--b",
            "openai",
            "--model-a",
            "gpt-4o-mini",
            "--model-b",
            "gpt-4o",
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert code == 0
    assert calls == [("openai", "gpt-4o-mini"), ("openai", "gpt-4o")]
    output = capsys.readouterr().out
    assert "gpt-4o-mini pass_rate 0.00" in output
    assert "gpt-4o pass_rate 1.00" in output
    assert "difference -1.00" in output
