from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SUPPORTED_M2_TYPES = ("prompt", "tool", "environment", "skill")
ALL_KNOWN_FIELDS = {
    "type",
    "turn_id",
    "mode",
    "content",
    "injected_tool",
    "injection_mcp_tool",
    "kwargs",
    "skill_name",
    "row",
}
TYPE_FIELDS = {
    "prompt": {"type", "turn_id", "mode", "content"},
    "tool": {"type", "mode", "content", "injected_tool"},
    "environment": {"type", "turn_id", "injection_mcp_tool", "kwargs"},
    "skill": {"type", "mode", "content", "skill_name", "row"},
}
REQUIRED_FIELDS = {key: set(value) for key, value in TYPE_FIELDS.items()}


@dataclass(frozen=True)
class ValidationErrorDetail:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatedAttackStep:
    type: str
    turn_id: int | None = None
    mode: str | None = None
    content: str | None = None
    injected_tool: str | None = None
    injection_mcp_tool: str | None = None
    kwargs: dict[str, Any] | None = None
    skill_name: str | None = None
    row: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}

    @property
    def dtap_turn_id(self) -> int:
        """Global DTAP injections are materialized into turn 1 at M3."""
        return self.turn_id if self.turn_id is not None else 1


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[ValidationErrorDetail, ...] = ()
    step: ValidatedAttackStep | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "valid": self.valid,
            "errors": [error.to_dict() for error in self.errors],
        }
        if self.step is not None:
            result["step"] = self.step.to_dict()
        return result


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    errors: tuple[ValidationErrorDetail, ...] = ()
    steps: tuple[ValidatedAttackStep, ...] = ()


def candidate_attack_step_schema() -> dict[str, Any]:
    """Policy-visible structural contract; task-specific allowlists live in AttackSurface."""
    string_content = {"type": "string", "minLength": 1}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "turn_id", "mode", "content"],
                "properties": {
                    "type": {"const": "prompt"},
                    "turn_id": {"type": "integer", "minimum": 1},
                    "mode": {"enum": ["suffix", "override", "jailbreak"]},
                    "content": string_content,
                },
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "mode", "content", "injected_tool"],
                "properties": {
                    "type": {"const": "tool"},
                    "mode": {"enum": ["suffix", "override"]},
                    "content": string_content,
                    "injected_tool": {"type": "string", "minLength": 1},
                },
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "turn_id", "injection_mcp_tool", "kwargs"],
                "properties": {
                    "type": {"const": "environment"},
                    "turn_id": {"type": "integer", "minimum": 1},
                    "injection_mcp_tool": {"type": "string", "minLength": 1},
                    "kwargs": {"type": "object"},
                },
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "mode", "content", "skill_name", "row"],
                "properties": {
                    "type": {"const": "skill"},
                    "mode": {"enum": ["insert", "append", "create"]},
                    "content": string_content,
                    "skill_name": {"type": "string", "minLength": 1},
                    "row": {"type": "integer"},
                },
            },
        ],
    }
