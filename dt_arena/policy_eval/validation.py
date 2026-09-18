from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jsonschema.validators import validator_for

from .actions import (
    ALL_KNOWN_FIELDS,
    REQUIRED_FIELDS,
    SUPPORTED_M2_TYPES,
    TYPE_FIELDS,
    PlanValidationResult,
    ValidatedAttackStep,
    ValidationErrorDetail,
    ValidationResult,
)

_SAFE_SKILL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class ValidationContext:
    threat_model: str | None
    instruction_count: int
    attack_surface: Any

    @classmethod
    def from_view(cls, view: Any) -> ValidationContext:
        instruction = view.task.task_instruction
        if isinstance(instruction, list):
            count = len(instruction)
        elif isinstance(instruction, str) and instruction != "":
            count = 1
        else:
            count = 0
        return cls(
            threat_model=view.task.threat_model,
            instruction_count=count,
            attack_surface=view.attack_surface,
        )


def _error(code: str, path: str, message: str) -> ValidationResult:
    return ValidationResult(False, (ValidationErrorDetail(code, path, message),))


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_shape(raw: Any) -> ValidationResult | None:
    if not isinstance(raw, Mapping):
        return _error("INVALID_SHAPE", "$", "step must be an object")
    step_type = raw.get("type")
    if step_type == "a2a":
        return _error("UNSUPPORTED_IN_M2", "type", "a2a injection is not supported in M2")
    if step_type not in SUPPORTED_M2_TYPES:
        return _error("INJECTION_TYPE_NOT_ALLOWED", "type", "unsupported attack step type")

    allowed = TYPE_FIELDS[step_type]
    unknown = set(raw) - ALL_KNOWN_FIELDS
    if unknown:
        field = sorted(unknown)[0]
        return _error("UNKNOWN_FIELD", field, "unknown field")
    irrelevant = set(raw) - allowed
    if irrelevant:
        field = sorted(irrelevant)[0]
        return _error("FIELD_NOT_ALLOWED", field, f"field is not valid for {step_type} steps")
    missing = REQUIRED_FIELDS[step_type] - set(raw)
    if missing:
        field = sorted(missing)[0]
        return _error("MISSING_FIELD", field, "required field is missing")
    return None


def _validate_turn(raw: Mapping[str, Any], ctx: ValidationContext) -> ValidationResult | None:
    turn = raw.get("turn_id")
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
        return _error("TURN_OUT_OF_RANGE", "turn_id", "turn_id must be a positive integer")
    if ctx.threat_model == "indirect" and (ctx.instruction_count < 1 or turn > ctx.instruction_count):
        return _error("TURN_OUT_OF_RANGE", "turn_id", "turn_id exceeds the task instruction count")
    return None


def _tool_by_name(tools: Iterable[Any], qualified_name: str) -> Any | None:
    return next((tool for tool in tools if tool.qualified_name == qualified_name), None)


def _validate_json_schema(instance: Any, schema: Mapping[str, Any]) -> ValidationResult | None:
    try:
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
        errors = sorted(validator_cls(schema).iter_errors(instance), key=lambda e: list(e.absolute_path))
    except Exception:
        return _error("INVALID_TOOL_SCHEMA", "kwargs", "tool schema could not be validated")
    if not errors:
        return None
    err = errors[0]
    suffix = ".".join(str(part) for part in err.absolute_path)
    path = "kwargs" + (f".{suffix}" if suffix else "")
    # Keep messages stable and implementation-light; do not pass jsonschema reprs through.
    if err.validator == "required":
        msg = "required field is missing"
    elif err.validator == "additionalProperties":
        msg = "unexpected field"
    elif err.validator == "type":
        msg = "field has the wrong type"
    elif err.validator == "enum":
        msg = "field is not an allowed value"
    else:
        msg = "arguments do not match the tool schema"
    return _error("SCHEMA_MISMATCH", path, msg)


def validate_attack_step(raw: Any, ctx: ValidationContext) -> ValidationResult:
    shape_error = _validate_shape(raw)
    if shape_error is not None:
        return shape_error
    assert isinstance(raw, Mapping)
    step_type = str(raw["type"])
    surface = ctx.attack_surface

    if step_type == "prompt":
        if not surface.prompt_enabled:
            return _error("INJECTION_TYPE_NOT_ALLOWED", "type", "prompt injection is disabled")
        turn_error = _validate_turn(raw, ctx)
        if turn_error:
            return turn_error
        mode = raw["mode"]
        allowed_modes = set(getattr(surface, "prompt_modes", ()) or ())
        if mode not in allowed_modes:
            return _error("MODE_NOT_ALLOWED", "mode", "prompt mode is not allowed for this task")
        if not _nonempty_string(raw["content"]):
            return _error("EMPTY_CONTENT", "content", "content must be a non-empty string")
        return ValidationResult(
            True,
            step=ValidatedAttackStep(type="prompt", turn_id=raw["turn_id"], mode=mode, content=raw["content"].strip()),
        )

    if step_type == "tool":
        if not surface.tool_enabled:
            return _error("INJECTION_TYPE_NOT_ALLOWED", "type", "tool injection is disabled")
        allowed_modes = set(getattr(surface, "tool_modes", ()) or ())
        if raw["mode"] not in allowed_modes:
            return _error("MODE_NOT_ALLOWED", "mode", "tool mode is not enabled for this task")
        if not _nonempty_string(raw["content"]):
            return _error("EMPTY_CONTENT", "content", "content must be a non-empty string")
        target = str(raw["injected_tool"])
        if _tool_by_name(surface.victim_tools, target) is None:
            return _error("TARGET_NOT_ALLOWED", "injected_tool", "tool target is not in this task's attack surface")
        return ValidationResult(
            True,
            step=ValidatedAttackStep(
                type="tool", mode=raw["mode"], content=raw["content"].strip(), injected_tool=target
            ),
        )

    if step_type == "environment":
        if not surface.environment_enabled:
            return _error("INJECTION_TYPE_NOT_ALLOWED", "type", "environment injection is disabled")
        turn_error = _validate_turn(raw, ctx)
        if turn_error:
            return turn_error
        target = str(raw["injection_mcp_tool"])
        tool = _tool_by_name(surface.environment_tools, target)
        if tool is None:
            return _error("TARGET_NOT_ALLOWED", "injection_mcp_tool", "environment target is not allowed")
        kwargs = raw["kwargs"]
        if not isinstance(kwargs, Mapping):
            return _error("INVALID_SHAPE", "kwargs", "kwargs must be an object")
        schema_error = _validate_json_schema(dict(kwargs), tool.input_schema)
        if schema_error:
            return schema_error
        return ValidationResult(
            True,
            step=ValidatedAttackStep(
                type="environment", turn_id=raw["turn_id"], injection_mcp_tool=target, kwargs=dict(kwargs)
            ),
        )

    if step_type == "skill":
        if not surface.skill_enabled:
            return _error("INJECTION_TYPE_NOT_ALLOWED", "type", "skill injection is disabled")
        mode = raw["mode"]
        allowed_modes = set(surface.skill_modes or ())
        if mode not in allowed_modes:
            return _error("MODE_NOT_ALLOWED", "mode", "skill mode is not enabled for this task")
        if not _nonempty_string(raw["content"]):
            return _error("EMPTY_CONTENT", "content", "content must be a non-empty string")
        name = raw["skill_name"]
        if not isinstance(name, str) or not _SAFE_SKILL_NAME.fullmatch(name) or ".." in name:
            return _error("INVALID_SKILL_NAME", "skill_name", "skill_name is not a safe identifier")
        existing = set(getattr(surface, "skill_targets", ()) or ())
        if mode in {"insert", "append"} and name not in existing:
            return _error("TARGET_NOT_ALLOWED", "skill_name", "existing skill target is not available")
        if mode == "create" and name in existing:
            return _error("TARGET_NOT_ALLOWED", "skill_name", "create requires a new skill name")
        row = raw["row"]
        if isinstance(row, bool) or not isinstance(row, int):
            return _error("INVALID_SHAPE", "row", "row must be an integer")
        if mode == "insert" and row != -1 and row < 1:
            return _error("FIELD_NOT_ALLOWED", "row", "insert row must be >= 1 or -1")
        if mode in {"append", "create"} and row != -1:
            return _error("FIELD_NOT_ALLOWED", "row", f"row must be -1 for {mode}")
        return ValidationResult(
            True,
            step=ValidatedAttackStep(
                type="skill", mode=mode, content=raw["content"].strip(), skill_name=name, row=row
            ),
        )

    raise AssertionError("unreachable")


def validate_attack_plan(raw_steps: Sequence[Any], ctx: ValidationContext) -> PlanValidationResult:
    validated: list[ValidatedAttackStep] = []
    errors: list[ValidationErrorDetail] = []
    for index, raw in enumerate(raw_steps):
        result = validate_attack_step(raw, ctx)
        if not result.valid:
            for err in result.errors:
                errors.append(ValidationErrorDetail(err.code, f"steps[{index}].{err.path}", err.message))
        elif result.step is not None:
            validated.append(result.step)
    if errors:
        return PlanValidationResult(False, tuple(errors), ())

    seen_prompt_turns: set[int] = set()
    seen_tools: set[str] = set()
    skill_ops: dict[str, set[str]] = {}
    for index, step in enumerate(validated):
        if step.type == "prompt":
            assert step.turn_id is not None
            if step.turn_id in seen_prompt_turns:
                errors.append(
                    ValidationErrorDetail(
                        "PLAN_CONFLICT", f"steps[{index}].turn_id", "only one prompt injection is allowed per turn"
                    )
                )
            seen_prompt_turns.add(step.turn_id)
        elif step.type == "tool":
            assert step.injected_tool is not None
            if step.injected_tool in seen_tools:
                errors.append(
                    ValidationErrorDetail(
                        "PLAN_CONFLICT", f"steps[{index}].injected_tool", "duplicate tool injection target"
                    )
                )
            seen_tools.add(step.injected_tool)
        elif step.type == "skill":
            assert step.skill_name is not None and step.mode is not None
            prior = skill_ops.setdefault(step.skill_name, set())
            if "create" in prior or (step.mode == "create" and prior):
                errors.append(
                    ValidationErrorDetail(
                        "PLAN_CONFLICT",
                        f"steps[{index}].skill_name",
                        "create cannot be combined with another operation on the same skill",
                    )
                )
            prior.add(step.mode)

    if ctx.threat_model == "direct":
        prompt_turns = sorted(step.turn_id for step in validated if step.type == "prompt" and step.turn_id is not None)
        if validated and not prompt_turns:
            errors.append(
                ValidationErrorDetail(
                    "PLAN_CONFLICT", "steps", "direct attack plans require at least one jailbreak prompt"
                )
            )
        if prompt_turns:
            expected = list(range(1, len(prompt_turns) + 1))
            if prompt_turns != expected:
                errors.append(
                    ValidationErrorDetail(
                        "PLAN_CONFLICT", "steps", "direct jailbreak prompt turns must be contiguous from turn 1"
                    )
                )
            max_turn = len(prompt_turns)
            for index, step in enumerate(validated):
                if step.type == "environment" and (step.turn_id or 0) > max_turn:
                    errors.append(
                        ValidationErrorDetail(
                            "TURN_OUT_OF_RANGE",
                            f"steps[{index}].turn_id",
                            "environment turn exceeds the direct jailbreak turn count",
                        )
                    )

    return PlanValidationResult(not errors, tuple(errors), tuple(validated) if not errors else ())
