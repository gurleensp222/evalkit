"""Scorers for model outputs.

Each scorer takes ``(output, expected)`` and returns a ``ScoreResult``.
``expected`` depends on the scorer:

- ``exact_match``: a string
- ``contains_all``: a list of strings, or one string
- ``json_valid``: a list of keys, or ``{"required_keys": [...]}``
- ``python_unit_tests``: test source, a list of test lines, or ``{"tests": ...}``
- ``llm_judge``: a rubric string, or ``{"rubric": "..."}``. The runner also
  passes the task prompt and a judge provider.
- ``no_fabricated_links``: the task's ``allowed_urls`` list. An empty list
  allows no links. The runner passes that list, not ``expected``.
"""

from __future__ import annotations

import json
import re
import string
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from evalkit.models import ScoreResult
from evalkit.providers import Provider

SUBPROCESS_TIMEOUT_SECONDS = 5
_MISSING = object()

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_PYTHON_FENCE_RE = re.compile(
    r"```(?:python|py)\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_ANY_FENCE_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)", re.IGNORECASE)
_BARE_URL_RE = re.compile(r"https?://[^\s<>\[\]()]+", re.IGNORECASE)
_URL_TRAILING = ".,;:!?\"'`>"

Scorer = Callable[[str, Any], ScoreResult]


def _passed(reason: str) -> ScoreResult:
    return ScoreResult(score=1.0, passed=True, reason=reason)


def _failed(reason: str) -> ScoreResult:
    return ScoreResult(score=0.0, passed=False, reason=reason)


def _normalize_exact(text: str) -> str:
    """Strip surrounding whitespace and trailing punctuation."""
    normalized = text.strip()
    while True:
        stripped = normalized.rstrip(string.punctuation).strip()
        if stripped == normalized:
            return normalized
        normalized = stripped


def exact_match(output: str, expected: Any) -> ScoreResult:
    """Pass when both sides match after whitespace and trailing punctuation are stripped."""
    if not isinstance(output, str) or not isinstance(expected, str):
        return _failed("exact_match requires string output and expected text")

    got = _normalize_exact(output)
    want = _normalize_exact(expected)
    if got == want:
        return _passed("exact match")
    return _failed(f"expected {want!r}, got {got!r}")


def _expected_strings(expected: Any) -> list[str] | ScoreResult:
    if isinstance(expected, str):
        return [expected]
    if isinstance(expected, list):
        return [str(item) for item in expected]
    return _failed("contains_all requires a string or a list of strings")


def contains_all(output: str, expected: Any) -> ScoreResult:
    """Pass when every expected string appears, ignoring case."""
    if not isinstance(output, str):
        return _failed("contains_all requires string output")

    needles = _expected_strings(expected)
    if isinstance(needles, ScoreResult):
        return needles

    haystack = output.casefold()
    missing = [needle for needle in needles if needle.casefold() not in haystack]
    if missing:
        listed = ", ".join(repr(needle) for needle in missing)
        return _failed(f"missing: {listed}")
    return _passed("all expected strings found")


def _required_keys(expected: Any) -> list[str] | ScoreResult:
    if expected is None:
        return []
    if isinstance(expected, str):
        return [expected]
    if isinstance(expected, list):
        return [str(key) for key in expected]
    if isinstance(expected, dict):
        if "required_keys" not in expected:
            return _failed("json_valid expected a list of keys or required_keys")
        keys = expected["required_keys"]
        if isinstance(keys, str):
            return [keys]
        if isinstance(keys, list):
            return [str(key) for key in keys]
        return _failed("required_keys must be a string or a list of keys")
    return _failed("json_valid expected a list of keys or required_keys")


def _json_candidates(output: str) -> list[str]:
    fenced = [piece.strip() for piece in _JSON_FENCE_RE.findall(output) if piece.strip()]
    if fenced:
        return fenced
    stripped = output.strip()
    return [stripped] if stripped else []


def _decode_embedded_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    return _MISSING


def _parse_json_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        value = _decode_embedded_json(text)
        if value is not _MISSING:
            return value
        raise


def _load_json(output: str) -> Any:
    candidates = _json_candidates(output)
    if not candidates:
        raise json.JSONDecodeError("Expecting value", output, 0)

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return _parse_json_text(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def json_valid(output: str, expected: Any) -> ScoreResult:
    """Pass when output contains JSON with every required key."""
    if not isinstance(output, str):
        return _failed("json_valid requires string output")

    required = _required_keys(expected)
    if isinstance(required, ScoreResult):
        return required

    try:
        value = _load_json(output)
    except json.JSONDecodeError as exc:
        return _failed(f"invalid JSON: {exc.msg}")

    if not required:
        return _passed("valid JSON")

    if not isinstance(value, dict):
        return _failed("JSON value must be an object to check required keys")

    missing = [key for key in required if key not in value]
    if missing:
        listed = ", ".join(repr(key) for key in missing)
        return _failed(f"missing keys: {listed}")
    return _passed("valid JSON with required keys")


def _extract_code(output: str) -> str:
    python_blocks = [
        block.strip("\n") for block in _PYTHON_FENCE_RE.findall(output) if block.strip()
    ]
    if python_blocks:
        return "\n\n".join(python_blocks)

    blocks = [block.strip("\n") for block in _ANY_FENCE_RE.findall(output) if block.strip()]
    if blocks:
        return "\n\n".join(blocks)
    return output.strip()


def _test_source(expected: Any) -> str | ScoreResult:
    if isinstance(expected, str):
        source = expected.strip()
    elif isinstance(expected, list):
        source = "\n".join(str(line) for line in expected).strip()
    elif isinstance(expected, dict) and isinstance(expected.get("tests"), (str, list)):
        return _test_source(expected["tests"])
    else:
        return _failed("python_unit_tests requires test source in expected")

    if not source:
        return _failed("no tests provided")
    return source


def _failure_detail(result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "tests failed").strip()
    if len(detail) > 2000:
        detail = detail[-2000:]
    return detail or "tests failed"


def _run_python(source: str, timeout: float) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        script_path = Path(directory) / "submission.py"
        script_path.write_text(source, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script_path)],
            cwd=directory,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )


def python_unit_tests(output: str, expected: Any) -> ScoreResult:
    """Run extracted code plus tests in a subprocess. Pass only if every assert succeeds."""
    if not isinstance(output, str):
        return _failed("python_unit_tests requires string output")

    tests = _test_source(expected)
    if isinstance(tests, ScoreResult):
        return tests

    code = _extract_code(output)
    if not code:
        return _failed("no code to execute")

    program = f"{code}\n\n{tests}\n"
    timeout = SUBPROCESS_TIMEOUT_SECONDS
    try:
        completed = _run_python(program, timeout)
    except subprocess.TimeoutExpired:
        return _failed(f"timed out after {timeout:g} seconds")

    if completed.returncode == 0:
        return _passed("all asserts passed")
    return _failed(_failure_detail(completed))


def _rubric(expected: Any) -> str | ScoreResult:
    if isinstance(expected, str) and expected.strip():
        return expected.strip()
    if isinstance(expected, dict):
        rubric = expected.get("rubric")
        if isinstance(rubric, str) and rubric.strip():
            return rubric.strip()
    return _failed("llm_judge requires a rubric in expected")


def _judge_message(prompt: str, output: str, rubric: str) -> str:
    return (
        "You are grading one model answer.\n"
        "Do not reward length. Score a short answer that meets the rubric "
        "the same as a long one. Ignore extra detail the rubric does not require.\n\n"
        f"Task:\n{prompt}\n\n"
        f"Model answer:\n{output}\n\n"
        f"Rubric:\n{rubric}\n\n"
        "Reply with JSON only, in this shape: "
        '{"score": <integer from 1 to 5>, "reason": "<brief reason>"}'
    )


def _coerce_judge_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        score = value
    elif isinstance(value, float) and value.is_integer():
        score = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        score = int(value.strip())
    else:
        return None
    if 1 <= score <= 5:
        return score
    return None


def _parse_judge_payload(text: str) -> tuple[int, str] | ScoreResult:
    try:
        value = _load_json(text)
    except json.JSONDecodeError as exc:
        return _failed(f"judge returned invalid JSON: {exc.msg}")
    if not isinstance(value, dict):
        return _failed("judge response must be a JSON object")

    score = _coerce_judge_score(value.get("score"))
    if score is None:
        return _failed("judge score must be an integer from 1 to 5")

    reason = value.get("reason", "")
    if isinstance(reason, str) and reason.strip():
        return score, reason.strip()
    return score, f"judge score {score}"


def llm_judge(output: str, expected: Any, *, prompt: str, judge: Provider) -> ScoreResult:
    """Ask a judge model for ``{"score": 1-5, "reason"}`` and scale it to 0-1.

    A raw score of 1 becomes 0 and 5 becomes 1, using ``(score - 1) / 4``.
    Scores of 3, 4, and 5 pass. The judge is told not to reward length.
    """
    if not isinstance(output, str):
        return _failed("llm_judge requires string output")
    if not isinstance(prompt, str):
        return _failed("llm_judge requires the task prompt")

    rubric = _rubric(expected)
    if isinstance(rubric, ScoreResult):
        return rubric

    try:
        reply = judge.generate(_judge_message(prompt, output, rubric))
    except Exception as exc:  # noqa: BLE001 - a judge failure is a failed score
        return _failed(f"judge failed: {exc}")
    if not isinstance(reply, str):
        return _failed("judge returned a non-string response")

    parsed = _parse_judge_payload(reply)
    if isinstance(parsed, ScoreResult):
        return parsed
    score, reason = parsed
    return ScoreResult(score=(score - 1) / 4, passed=score >= 3, reason=reason)


def extract_urls(text: str) -> list[str]:
    """Return http and https URLs, including the target of a markdown link."""
    if not isinstance(text, str) or not text:
        return []
    found: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        url = raw.rstrip(_URL_TRAILING)
        if url and url not in seen:
            seen.add(url)
            found.append(url)

    for match in _MARKDOWN_LINK_RE.finditer(text):
        add(match.group(1))
    without_markdown = _MARKDOWN_LINK_RE.sub(" ", text)
    for match in _BARE_URL_RE.finditer(without_markdown):
        add(match.group(0))
    return found


def _normalize_url(url: str) -> str:
    """Compare URLs after scheme, host, default port, slash, and empty fragment.

    The path, query, and a non-empty fragment stay case-sensitive. ``:80`` is
    dropped for http and ``:443`` for https. A trailing slash on the path and a
    trailing ``#`` are dropped.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    try:
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        return url
    host = (hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    auth = ""
    if parts.username is not None:
        auth = parts.username
        if parts.password is not None:
            auth += f":{parts.password}"
        auth += "@"
    netloc = auth + host
    if port is not None:
        netloc += f":{port}"
    path = parts.path.removesuffix("/")
    return urlunsplit((scheme, netloc, path, parts.query, parts.fragment))


def no_fabricated_links(output: str, allowed_urls: Any) -> ScoreResult:
    """Fail when the answer contains an http(s) URL that is not allowed.

    ``allowed_urls`` is a list. An empty list allows no links. An answer with
    no links passes. Comparison lowercases the scheme and host, drops a
    trailing slash, drops default ports, and drops a trailing ``#``.
    """
    if not isinstance(output, str):
        return _failed("no_fabricated_links requires string output")
    if allowed_urls is None:
        allowed: set[str] = set()
    elif isinstance(allowed_urls, list) and all(isinstance(item, str) for item in allowed_urls):
        allowed = {_normalize_url(url) for url in allowed_urls}
    else:
        return _failed("no_fabricated_links requires a list of URLs")
    invented = [url for url in extract_urls(output) if _normalize_url(url) not in allowed]
    if invented:
        listed = ", ".join(invented)
        return _failed(f"fabricated link: {listed}")
    return _passed("no fabricated links")


SCORERS: dict[str, Scorer] = {
    "exact_match": exact_match,
    "contains_all": contains_all,
    "json_valid": json_valid,
    "python_unit_tests": python_unit_tests,
    "llm_judge": llm_judge,
    "no_fabricated_links": no_fabricated_links,
}
