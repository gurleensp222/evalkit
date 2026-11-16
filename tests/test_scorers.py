import pytest
from pydantic import ValidationError

from evalkit.models import ScoreResult, Task, TaskResult
from evalkit.scorers import (
    SCORERS,
    contains_all,
    exact_match,
    json_valid,
    llm_judge,
    no_fabricated_links,
    python_unit_tests,
)


def assert_passed(result: ScoreResult) -> None:
    assert result.passed is True
    assert result.score == 1.0
    assert result.reason


def assert_failed(result: ScoreResult) -> None:
    assert result.passed is False
    assert result.score == 0.0
    assert result.reason


def test_score_result_rejects_scores_outside_zero_to_one() -> None:
    ScoreResult(score=0, passed=False, reason="low")
    ScoreResult(score=1, passed=True, reason="high")
    with pytest.raises(ValidationError):
        ScoreResult(score=-0.1, passed=False, reason="below")
    with pytest.raises(ValidationError):
        ScoreResult(score=1.1, passed=True, reason="above")


def test_task_models_keep_the_specified_fields() -> None:
    task = Task(
        id="t1",
        category="math",
        prompt="What is 2 + 2?",
        scorer="exact_match",
        expected="4",
    )
    result = TaskResult(
        run_id="run-1",
        task_id=task.id,
        category=task.category,
        model="mock",
        output="4",
        score=1,
        passed=True,
        reason="exact match",
        latency_ms=12.5,
    )
    assert result.task_id == "t1"
    assert result.model == "mock"
    assert result.latency_ms == 12.5


def test_scorers_are_registered_by_name() -> None:
    assert set(SCORERS) == {
        "exact_match",
        "contains_all",
        "json_valid",
        "python_unit_tests",
        "llm_judge",
        "no_fabricated_links",
    }
    assert SCORERS["exact_match"] is exact_match
    assert SCORERS["contains_all"] is contains_all
    assert SCORERS["json_valid"] is json_valid
    assert SCORERS["python_unit_tests"] is python_unit_tests
    assert SCORERS["llm_judge"] is llm_judge
    assert SCORERS["no_fabricated_links"] is no_fabricated_links


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("hello", "hello"),
        ("  hello world  ", "hello world"),
        ("\n hello! \n", "hello"),
        ("hello...", "hello"),
        ("  hello! !  ", "hello"),
        ("answer?", "  answer.  "),
        ("   ", ""),
        ("???", "..."),
    ],
)
def test_exact_match_passes(output: str, expected: str) -> None:
    assert_passed(exact_match(output, expected))


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("hello", "Hello"),
        ("hello", "hello world"),
        ("a.b", "ab"),
        ("", "x"),
        ("(hello)", "hello"),
    ],
)
def test_exact_match_fails(output: str, expected: str) -> None:
    result = exact_match(output, expected)
    assert_failed(result)
    assert "expected" in result.reason


def test_exact_match_rejects_non_strings() -> None:
    assert_failed(exact_match("3", 3))


def test_exact_match_preserves_internal_whitespace() -> None:
    result = exact_match("hello  world", "hello world")
    assert_failed(result)
    assert "hello  world" in result.reason


def test_exact_match_strips_tabs_and_carriage_returns() -> None:
    assert_passed(exact_match("\thello\r\n", "hello"))


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("The Quick Brown Fox", ["quick", "FOX"]),
        ("  ABC def  ", ["abc", "DEF"]),
        ("say hello!", "HELLO"),
        ("anything at all", []),
    ],
)
def test_contains_all_passes(output: str, expected: str | list[str]) -> None:
    assert_passed(contains_all(output, expected))


def test_contains_all_reports_only_missing_strings() -> None:
    result = contains_all("Hello World", ["hello", "world", "moon"])
    assert_failed(result)
    assert "moon" in result.reason
    assert "hello" not in result.reason


def test_contains_all_fails_when_output_is_empty() -> None:
    assert_failed(contains_all("", ["needed"]))


def test_contains_all_rejects_a_mapping() -> None:
    result = contains_all("hello", {"values": ["hello"]})
    assert_failed(result)
    assert "list" in result.reason


def test_contains_all_searches_casefolded_substrings_not_patterns() -> None:
    assert_passed(contains_all("The STRASSE is long", ["straße"]))
    assert_passed(contains_all("see a.b and 420", ["a.b", 42]))

    missed = contains_all("axb and 4", ["a.b", 42])
    assert_failed(missed)
    assert "a.b" in missed.reason
    assert "42" in missed.reason


def test_json_valid_accepts_an_object_with_required_keys() -> None:
    output = '{"name": "ada", "city": "London"}'
    assert_passed(json_valid(output, {"required_keys": ["name", "city"]}))


def test_json_valid_accepts_fenced_json_with_extra_whitespace() -> None:
    output = """Here is the object:
```json

  {
    "name": "ada",
    "city": "London"
  }

```
Thanks.
"""
    assert_passed(json_valid(output, {"required_keys": ["name"]}))


def test_json_valid_accepts_a_json_fence_without_a_language_tag() -> None:
    output = """```
{"ok": true}
```"""
    assert_passed(json_valid(output, ["ok"]))


def test_json_valid_accepts_uppercase_fence_label() -> None:
    output = """```JSON
{"ok": true}
```"""
    assert_passed(json_valid(output, "ok"))


def test_json_valid_extracts_json_from_surrounding_prose() -> None:
    output = 'Result: {"ok": true, "n": 1}'
    assert_passed(json_valid(output, {"required_keys": ["ok", "n"]}))


def test_json_valid_allows_extra_keys_and_whitespace() -> None:
    output = '\n  { "name" : "ada" , "extra" : 1 }  \n'
    assert_passed(json_valid(output, {"required_keys": ["name"]}))


def test_json_valid_passes_any_json_when_no_keys_are_required() -> None:
    assert_passed(json_valid("[1, 2, 3]", {"required_keys": []}))


def test_json_valid_fails_on_invalid_json() -> None:
    result = json_valid("```json\n{nope}\n```", {"required_keys": ["name"]})
    assert_failed(result)
    assert "invalid JSON" in result.reason


def test_json_valid_fails_when_a_required_key_is_missing() -> None:
    result = json_valid('{"name": "ada"}', {"required_keys": ["name", "age"]})
    assert_failed(result)
    assert "age" in result.reason
    assert "name" not in result.reason


def test_json_valid_fails_when_required_keys_need_an_object() -> None:
    result = json_valid("[1, 2]", ["name"])
    assert_failed(result)
    assert "object" in result.reason


def test_json_valid_fails_on_empty_output() -> None:
    assert_failed(json_valid("   ", {"required_keys": ["name"]}))


def test_json_valid_counts_null_false_and_zero_as_present() -> None:
    output = '{"ok": false, "count": 0, "label": null}'
    assert_passed(json_valid(output, ["ok", "count", "label"]))


def test_json_valid_does_not_confuse_similar_or_nested_keys() -> None:
    similar = json_valid('{"username": "ada"}', ["name"])
    assert_failed(similar)
    assert "'name'" in similar.reason

    nested = json_valid('{"user": {"name": "ada"}}', ["name"])
    assert_failed(nested)
    assert "'name'" in nested.reason


def test_json_valid_skips_invalid_fences_and_stray_braces() -> None:
    fenced = """```json
{broken
```
```json
{"ok": true}
```"""
    assert_passed(json_valid(fenced, ["ok"]))
    assert_passed(json_valid('Use {braces} in prose. {"ok": true}', ["ok"]))


def test_json_valid_rejects_a_trailing_comma() -> None:
    result = json_valid('{"ok": true,}', ["ok"])
    assert_failed(result)
    assert "invalid JSON" in result.reason


ADD_FUNCTION = """```python
def add(a, b):
    return a + b
```
"""


def test_python_unit_tests_passes_fenced_code() -> None:
    result = python_unit_tests(ADD_FUNCTION, "assert add(1, 2) == 3\nassert add(0, 0) == 0")
    assert_passed(result)


def test_python_unit_tests_ignores_prose_around_the_fence() -> None:
    output = f"The solution is below.\n{ADD_FUNCTION}\nThat should work."
    assert_passed(python_unit_tests(output, "assert add(2, 2) == 4"))


def test_python_unit_tests_runs_unfenced_code() -> None:
    output = "def add(a, b):\n    return a + b\n"
    assert_passed(python_unit_tests(output, ["assert add(1, 1) == 2"]))


def test_python_unit_tests_fails_when_an_assert_fails() -> None:
    result = python_unit_tests(
        ADD_FUNCTION,
        "assert add(1, 1) == 2\nassert add(1, 1) == 3",
    )
    assert_failed(result)
    assert "AssertionError" in result.reason


def test_python_unit_tests_fails_when_code_raises() -> None:
    output = "```python\ndef boom():\n    raise RuntimeError('nope')\n```"
    result = python_unit_tests(output, "assert boom() is None")
    assert_failed(result)
    assert "RuntimeError" in result.reason
    assert "nope" in result.reason


def test_python_unit_tests_fails_on_syntax_error() -> None:
    result = python_unit_tests("def add(:\n    return 1\n", "assert True")
    assert_failed(result)
    assert "SyntaxError" in result.reason


def test_python_unit_tests_fails_when_no_tests_are_provided() -> None:
    assert_failed(python_unit_tests("x = 1\n", "  "))


def test_python_unit_tests_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evalkit.scorers.SUBPROCESS_TIMEOUT_SECONDS", 1)
    result = python_unit_tests("import time\ntime.sleep(30)\n", "assert True")
    assert_failed(result)
    assert "timed out" in result.reason.lower()


def test_python_unit_tests_ignores_non_python_fences() -> None:
    output = """```text
this is not python
```
```python
def add(a, b):
    return a + b
```
"""
    assert_passed(python_unit_tests(output, {"tests": "assert add(1, 2) == 3"}))


def test_python_unit_tests_concatenates_blocks_and_keeps_indentation() -> None:
    output = """```py
def outer():
    def inner():
        return 1
    return inner
```
```python
VALUE = 5
```
"""
    tests = "assert outer()() == 1\nassert VALUE == 5"
    assert_passed(python_unit_tests(output, tests))


def test_python_unit_tests_fails_when_code_reads_stdin_or_is_blank() -> None:
    eof = python_unit_tests("input()\n", "assert True")
    assert_failed(eof)
    assert "EOFError" in eof.reason

    blank = python_unit_tests("  \n", "assert True")
    assert_failed(blank)
    assert "no code" in blank.reason


class _Judge:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.name = "judge"

    def generate(self, prompt: str, *, task_id: str | None = None) -> str:
        del task_id
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_llm_judge_scales_one_to_five_onto_zero_to_one() -> None:
    expected = {
        1: (0.0, False),
        2: (0.25, False),
        3: (0.5, True),
        4: (0.75, True),
        5: (1.0, True),
    }
    for raw, (score, passed) in expected.items():
        judge = _Judge(f'{{"score": {raw}, "reason": "grade {raw}"}}')
        result = llm_judge("4", "digits only", prompt="What is 2 + 2?", judge=judge)
        assert result.score == score
        assert result.passed is passed
        assert result.reason == f"grade {raw}"


def test_llm_judge_sends_the_prompt_answer_and_rubric_and_rejects_length() -> None:
    judge = _Judge('{"score": 5, "reason": "meets the rubric"}')
    result = llm_judge(
        "4",
        {"rubric": "Reply with the digit 4 and nothing else."},
        prompt="What is 2 + 2?",
        judge=judge,
    )
    assert result.score == 1.0
    assert result.passed is True
    message = judge.prompts[0]
    assert "Do not reward length." in message
    assert "What is 2 + 2?" in message
    assert "Model answer:\n4" in message
    assert "Reply with the digit 4 and nothing else." in message


def test_llm_judge_accepts_fenced_json_and_a_numeric_string() -> None:
    judge = _Judge('```json\n{"score": "4", "reason": "close enough"}\n```')
    result = llm_judge("almost", "be correct", prompt="Solve it", judge=judge)
    assert result.score == 0.75
    assert result.passed is True
    assert result.reason == "close enough"


def test_llm_judge_rejects_bad_responses_and_a_missing_rubric() -> None:
    invalid = llm_judge("4", "digits", prompt="Q", judge=_Judge("not json"))
    assert_failed(invalid)
    assert "invalid JSON" in invalid.reason

    low = llm_judge("4", "digits", prompt="Q", judge=_Judge('{"score": 0, "reason": "no"}'))
    assert_failed(low)
    assert "1 to 5" in low.reason

    missing = llm_judge("4", {"note": "no rubric"}, prompt="Q", judge=_Judge("{}"))
    assert_failed(missing)
    assert "rubric" in missing.reason
    assert missing.reason != "judge returned invalid JSON"


def test_llm_judge_reports_a_judge_failure() -> None:
    result = llm_judge("4", "digits", prompt="Q", judge=_Judge(RuntimeError("offline")))
    assert_failed(result)
    assert result.reason == "judge failed: offline"


def test_no_fabricated_links_checks_bare_and_markdown_urls() -> None:
    assert_passed(no_fabricated_links("No address here.", []))
    assert_passed(no_fabricated_links("See https://example.com/docs", ["https://example.com/docs"]))
    invented = no_fabricated_links("See https://evil.example/phish", ["https://example.com/docs"])
    assert_failed(invented)
    assert "https://evil.example/phish" in invented.reason
    markdown = no_fabricated_links(
        "Read the [guide](https://example.com/docs) today.",
        [],
    )
    assert_failed(markdown)
    assert "https://example.com/docs" in markdown.reason
    assert_passed(
        no_fabricated_links(
            "Read the [guide](https://example.com/docs) today.",
            ["https://example.com/docs"],
        )
    )
