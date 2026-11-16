"""Load rubric files and the prompts that point at them."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from evalkit.models import Task

_LEVELS = {1, 2, 3, 4, 5}
_WEIGHT_TOLERANCE = 1e-6


class Criterion(BaseModel):
    """One weighted part of a rubric."""

    id: str
    weight: float = Field(gt=0)
    description: str
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
            lines.append(
                f"- {criterion.id} (weight {criterion.weight:.2f}): {criterion.description.strip()}"
            )
            if criterion.levels:
                for score in sorted(criterion.levels):
                    lines.append(f"  {score}: {criterion.levels[score].strip()}")
        return "\n".join(lines)


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
