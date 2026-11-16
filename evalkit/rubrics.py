"""Load rubric files and the prompts that point at them."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from evalkit.models import Task

_LEVELS = {1, 2, 3, 4, 5}
_WEIGHT_TOLERANCE = 1e-6
_AND = re.compile(r"\s+and\s+", re.IGNORECASE)
_COMMA = re.compile(r",\s+")
_LEADING_CONJUNCTION = re.compile(r"(?i)(?:and|or)\b")
_WORD = re.compile(r"[A-Za-z']+")
_UNSEEN = re.compile(r"\b(?:the correct|from the text|in the prompt)\b", re.IGNORECASE)
_GENERIC = re.compile(
    r"\b(?:spelling|politeness|follow(?:s|ing)?(?:\s+the)?\s+instructions)\b",
    re.IGNORECASE,
)
_TWO_REQUIREMENTS = "description likely holds two requirements"
_UNSEEN_FACT = "refers to information the grader cannot see"
_GENERIC_CRITERION = "criterion is generic"
_NO_IMPLICIT = "rubric has no implicit criteria"
# Stems, plus auxiliaries. Surface forms such as "mentions" reduce to these.
_VERBS = frozenset(
    {
        "accept",
        "add",
        "address",
        "am",
        "answer",
        "are",
        "ask",
        "avoid",
        "be",
        "been",
        "calculate",
        "can",
        "cite",
        "claim",
        "compare",
        "contain",
        "could",
        "cover",
        "define",
        "describe",
        "did",
        "do",
        "does",
        "explain",
        "follow",
        "format",
        "give",
        "had",
        "handle",
        "has",
        "have",
        "identify",
        "include",
        "invent",
        "is",
        "keep",
        "leak",
        "list",
        "match",
        "may",
        "mention",
        "might",
        "must",
        "name",
        "omit",
        "provide",
        "quote",
        "raise",
        "recommend",
        "refuse",
        "remain",
        "return",
        "shall",
        "should",
        "show",
        "skip",
        "sound",
        "state",
        "stay",
        "suggest",
        "summarize",
        "tell",
        "use",
        "warn",
        "was",
        "were",
        "will",
        "would",
        "write",
    }
)


class Criterion(BaseModel):
    """One weighted part of a rubric."""

    id: str
    weight: float = Field(gt=0)
    description: str
    necessity: Literal["explicit", "implicit"]
    kind: Literal["objective", "subjective"]
    critical: bool = False
    answer_key: str | None = None
    example: str | None = None
    levels: dict[int, str] | None = None

    @field_validator("levels", mode="before")
    @classmethod
    def coerce_level_keys(cls, value: object) -> object:
        if value is None or not isinstance(value, dict):
            return value
        coerced: dict[object, object] = {}
        for key, text in value.items():
            if isinstance(key, str) and key.strip().lstrip("-").isdigit():
                coerced[int(key.strip())] = text
            else:
                coerced[key] = text
        return coerced

    @field_validator("levels")
    @classmethod
    def levels_are_one_to_five(cls, levels: dict[int, str] | None) -> dict[int, str] | None:
        if levels is None:
            return None
        bad = sorted(key for key in levels if key not in _LEVELS)
        if bad:
            raise ValueError(f"level keys must be integers from 1 to 5, got {bad}")
        return levels

    @model_validator(mode="after")
    def grounding_matches_kind(self) -> Criterion:
        if self.kind == "objective" and not _present(self.answer_key):
            raise ValueError("objective criteria require answer_key")
        if self.kind == "subjective" and not _present(self.example):
            raise ValueError("subjective criteria require example")
        return self


class Rubric(BaseModel):
    """A versioned set of criteria a judge can apply."""

    id: str
    version: int = Field(ge=1)
    description: str
    pass_threshold: float = Field(ge=0, le=1)
    criteria: list[Criterion] = Field(min_length=1)

    @model_validator(mode="after")
    def criteria_are_consistent(self) -> Rubric:
        seen: list[str] = []
        duplicates: list[str] = []
        for criterion in self.criteria:
            if criterion.id in seen and criterion.id not in duplicates:
                duplicates.append(criterion.id)
            seen.append(criterion.id)
        if duplicates:
            raise ValueError(f"duplicate criterion ids: {', '.join(duplicates)}")
        total = sum(criterion.weight for criterion in self.criteria)
        if abs(total - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(f"criterion weights must sum to 1.0, got {total:.6f}")
        return self

    @property
    def content_hash(self) -> str:
        """SHA-256 of this rubric's canonical JSON."""
        payload = json.loads(self.model_dump_json())
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def render(self) -> str:
        """Return this rubric as text a judge can read."""
        lines = [
            f"Rubric {self.id} (version {self.version})",
            self.description.strip(),
            f"Pass threshold: {self.pass_threshold:.2f}",
            "",
            "Criteria:",
        ]
        for criterion in self.criteria:
            details = f"weight {criterion.weight:.2f}, {criterion.necessity}, {criterion.kind}"
            if criterion.critical:
                details += ", critical"
            lines.append(f"- {criterion.id} ({details}): {criterion.description.strip()}")
            answer_key = criterion.answer_key
            if answer_key and answer_key.strip():
                lines.append(f"  answer key: {answer_key.strip()}")
            example = criterion.example
            if example and example.strip():
                lines.append(f"  example: {example.strip()}")
            if criterion.levels:
                for score in sorted(criterion.levels):
                    lines.append(f"  {score}: {criterion.levels[score].strip()}")
        return "\n".join(lines)


def lint_rubrics(rubrics: Mapping[str, Rubric]) -> list[str]:
    """Return warnings for criteria that are hard for a grader to apply.

    Warnings name the rubric id and, when the problem is one criterion, that
    criterion's id. A rubric that fails to load is not a warning.
    """
    warnings: list[str] = []
    for rubric in rubrics.values():
        for criterion in rubric.criteria:
            if _holds_two_requirements(criterion.description):
                warnings.append(_warning(rubric.id, criterion.id, _TWO_REQUIREMENTS))
            if _grader_cannot_see(criterion):
                warnings.append(_warning(rubric.id, criterion.id, _UNSEEN_FACT))
            if _is_generic(criterion):
                warnings.append(_warning(rubric.id, criterion.id, _GENERIC_CRITERION))
        if not any(criterion.necessity == "implicit" for criterion in rubric.criteria):
            warnings.append(_warning(rubric.id, None, _NO_IMPLICIT))
    return warnings


def load_rubrics(directory: str | Path) -> dict[str, Rubric]:
    """Read every YAML rubric in ``directory``.

    The rubric id must match the file stem. Duplicate ids are an error.
    """
    folder = Path(directory)
    if not folder.is_dir():
        raise ValueError(f"rubric directory not found: {folder}")
    files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    loaded: dict[str, Rubric] = {}
    origins: dict[str, str] = {}
    for path in files:
        rubric = _read_rubric(path)
        if path.stem != rubric.id:
            raise ValueError(
                f"{path.name}: rubric id {rubric.id!r} does not match filename {path.stem!r}"
            )
        previous = origins.get(rubric.id)
        if previous is not None:
            raise ValueError(f"duplicate rubric id {rubric.id!r} in {previous} and {path.name}")
        loaded[rubric.id] = rubric
        origins[rubric.id] = path.name
    return loaded


def load_prompts(path: str | Path, rubrics: Mapping[str, Rubric]) -> list[Task]:
    """Load a JSONL file, or every JSONL file in a directory.

    Raises one error listing every prompt id whose ``rubric_id`` is not loaded.
    """
    files = _prompt_files(Path(path))
    tasks: list[Task] = []
    problems: list[str] = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                tasks.append(Task.model_validate_json(stripped))
            except ValidationError as exc:
                problems.append(f"{file}:{line_number}: invalid prompt: {exc}")
    missing = [task.id for task in tasks if task.rubric_id and task.rubric_id not in rubrics]
    if missing:
        problems.append("unknown rubric_id for prompt ids: " + ", ".join(missing))
    if problems:
        raise ValueError("\n".join(problems))
    return tasks


def _present(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def _warning(rubric_id: str, criterion_id: str | None, message: str) -> str:
    if criterion_id is None:
        return f"{rubric_id}: {message}"
    return f"{rubric_id} {criterion_id}: {message}"


def _holds_two_requirements(description: str) -> bool:
    return _and_joins_two_verbs(description) or _comma_list_of_demands(description)


def _and_joins_two_verbs(description: str) -> bool:
    parts = _AND.split(description)
    pairs = itertools.pairwise(parts)
    return any(_contains_verb(left) and _contains_verb(right) for left, right in pairs)


def _contains_verb(text: str) -> bool:
    return any(_is_verb(word) for word in _WORD.findall(text))


def _is_verb(word: str) -> bool:
    token = word.lower().strip("'")
    stems = {token}
    if len(token) > 4 and token.endswith("ies"):
        stems.add(token[:-3] + "y")
    if len(token) > 3 and token.endswith("s"):
        stems.add(token[:-1])
    if len(token) > 4 and token.endswith("es"):
        stems.add(token[:-2])
    if len(token) > 4 and token.endswith("ed"):
        stems.add(token[:-2])
        stems.add(token[:-1])
    if len(token) > 5 and token.endswith("ing"):
        stems.add(token[:-3])
        stems.add(f"{token[:-3]}e")
    return bool(stems & _VERBS)


def _comma_list_of_demands(description: str) -> bool:
    """True when commas separate a list of demands, not a single aside."""
    chunks = [chunk.strip() for chunk in _COMMA.split(description) if chunk.strip()]
    if len(chunks) >= 3:
        return True
    return len(chunks) == 2 and _LEADING_CONJUNCTION.match(chunks[1]) is not None


def _grader_cannot_see(criterion: Criterion) -> bool:
    if _present(criterion.answer_key):
        return False
    return _UNSEEN.search(criterion.description) is not None


def _is_generic(criterion: Criterion) -> bool:
    criterion_id = criterion.id.replace("_", " ")
    described = _GENERIC.search(criterion.description) is not None
    named = _GENERIC.search(criterion_id) is not None
    return described or named


def _read_rubric(path: Path) -> Rubric:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: rubric must be a mapping")  # noqa: TRY004
    try:
        return Rubric.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def _prompt_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix == ".jsonl")
        if not files:
            raise ValueError(f"no JSONL prompts in {path}")
        return files
    if path.is_file():
        return [path]
    raise ValueError(f"prompts not found: {path}")
