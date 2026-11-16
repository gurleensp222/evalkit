"""Grade a response against each rubric criterion."""

from __future__ import annotations

import json
from collections import Counter

from evalkit.models import CriterionMark, ScoreResult
from evalkit.providers import Provider
from evalkit.rubrics import Rubric
from evalkit.scorers import _load_json

GRADER_PARSE_ERROR = "grader_parse_error"
GRADER_CONTEXTS = ("response_only", "with_prompt")


def grade_response(
    rubric: Rubric,
    output: str,
    task_prompt: str,
    judge: Provider,
    *,
    context: str = "with_prompt",
) -> ScoreResult:
    """Ask the judge which criteria the response meets and score the weights."""
    prompt = build_grader_prompt(rubric, output, task_prompt, context=context)
    try:
        raw = judge.generate(prompt)
    except Exception as exc:  # noqa: BLE001 - a judge failure is a failed score
        return ScoreResult(score=0.0, passed=False, reason=f"judge failed: {exc}")
    parsed = parse_criterion_marks(raw, rubric)
    if isinstance(parsed, str):
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=parsed,
            forced_failure=GRADER_PARSE_ERROR,
        )
    return score_marks(rubric, parsed)


def build_grader_prompt(
    rubric: Rubric,
    output: str,
    task_prompt: str,
    *,
    context: str = "with_prompt",
) -> str:
    """List every criterion. ``response_only`` leaves the task prompt out."""
    if context not in GRADER_CONTEXTS:
        raise ValueError("grader context must be response_only or with_prompt")
    lines = [
        "You are grading one model response against each rubric criterion.",
        "Mark a criterion met only when the response satisfies it.",
        "Use the answer key or example when one is listed.",
        "",
    ]
    if context == "with_prompt":
        lines.extend(["Task:", task_prompt, ""])
    lines.extend(
        [
            "Response:",
            output,
            "",
            f"Rubric {rubric.id} (version {rubric.version})",
            rubric.description.strip(),
            "",
            "Criteria:",
        ]
    )
    for criterion in rubric.criteria:
        flag = "critical" if criterion.critical else "not critical"
        lines.append(
            f"- {criterion.id} (weight {criterion.weight:g}, {flag}): "
            f"{criterion.description.strip()}"
        )
        answer_key = criterion.answer_key
        if isinstance(answer_key, str) and answer_key.strip():
            lines.append(f"  answer key: {answer_key.strip()}")
        example = criterion.example
        if isinstance(example, str) and example.strip():
            lines.append(f"  example: {example.strip()}")
    lines.extend(
        [
            "",
            (
                "Reply with JSON only, in this shape: "
                '{"criteria": [{"id": "<criterion id>", "met": true, "reason": "<brief reason>"}]}'
            ),
            "Include every criterion id exactly once.",
        ]
    )
    return "\n".join(lines)


def parse_criterion_marks(text: str, rubric: Rubric) -> list[CriterionMark] | str:
    """Return one mark per criterion, or a reason the judge output cannot be used."""
    try:
        payload = _load_json(text)
    except json.JSONDecodeError as exc:
        return f"judge returned invalid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return "judge response must be a JSON object"
    items = payload.get("criteria")
    if not isinstance(items, list):
        return "judge response must include a criteria list"

    expected = [criterion.id for criterion in rubric.criteria]
    expected_ids = set(expected)
    seen: list[str] = []
    marks: dict[str, CriterionMark] = {}
    problems: list[str] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            problems.append(f"criterion entry {index} must be an object")
            continue
        raw_id = item.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            problems.append(f"criterion entry {index} is missing id")
            continue
        criterion_id = raw_id.strip()
        seen.append(criterion_id)
        met = item.get("met")
        if not isinstance(met, bool):
            problems.append(f"criterion {criterion_id} met must be true or false")
            continue
        reason = item.get("reason", "")
        if reason is None:
            reason = ""
        if not isinstance(reason, str):
            problems.append(f"criterion {criterion_id} reason must be a string")
            continue
        if criterion_id not in marks:
            marks[criterion_id] = CriterionMark(
                criterion_id=criterion_id,
                met=met,
                reason=reason,
            )

    counts = Counter(seen)
    missing = [criterion_id for criterion_id in expected if counts[criterion_id] == 0]
    duplicate = [criterion_id for criterion_id in expected if counts[criterion_id] > 1]
    unknown = [criterion_id for criterion_id in counts if criterion_id not in expected_ids]
    if missing:
        problems.append("missing criterion ids: " + ", ".join(missing))
    if duplicate:
        problems.append("duplicate criterion ids: " + ", ".join(duplicate))
    if unknown:
        problems.append("unknown criterion ids: " + ", ".join(unknown))
    if problems:
        return "; ".join(problems)
    return [marks[criterion_id] for criterion_id in expected]


def score_marks(rubric: Rubric, marks: list[CriterionMark]) -> ScoreResult:
    """Sum the weights of met criteria. A failed critical criterion fails the attempt."""
    by_id = {mark.criterion_id: mark for mark in marks}
    score = sum(criterion.weight for criterion in rubric.criteria if by_id[criterion.id].met)
    score = min(1.0, max(0.0, round(score, 6)))
    failed = [
        criterion.id
        for criterion in rubric.criteria
        if criterion.critical and not by_id[criterion.id].met
    ]
    passed = score + 1e-9 >= rubric.pass_threshold and not failed
    if failed:
        first = failed[0]
        return ScoreResult(
            score=score,
            passed=False,
            reason=f"critical criterion failed: {first}",
            forced_failure=f"critical:{first}",
            criteria=marks,
        )
    unmet = [criterion.id for criterion in rubric.criteria if not by_id[criterion.id].met]
    if not passed:
        reason = (
            "unmet criteria: " + ", ".join(unmet)
            if unmet
            else "score is below the pass threshold"
        )
        return ScoreResult(score=score, passed=False, reason=reason, criteria=marks)
    reason = "all criteria met" if not unmet else "score meets the threshold"
    return ScoreResult(score=score, passed=True, reason=reason, criteria=marks)
