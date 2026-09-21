"""Typed, allowlisted M4 policy-visible response boundary."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


class PolicyContractViolation(RuntimeError):
    pass


_FORBIDDEN_KEYS = frozenset(
    {
        "judge_result",
        "judge_rationale",
        "victim_output",
        "trajectory_path",
        "runtime_identity",
        "task_success",
        "stdout",
        "stderr",
        "output_dir",
        "config_path",
        "exception",
        "traceback",
    }
)

_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer[ ]+[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b[a-f0-9]{32}\.[A-Za-z0-9_-]{12,}\b", re.I),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def canonical_policy_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _exact_keys(value: Mapping[str, Any], allowed: set[str], *, required: set[str]) -> None:
    if not required.issubset(value) or not set(value).issubset(allowed):
        raise PolicyContractViolation("invalid feedback schema")


def _validate_feedback_v4(value: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "schema_version",
            "final_response",
            "trace_complete",
            "tool_sequence",
            "steps",
            "guidance",
            "reasoning_summary",
        },
        required={"schema_version", "final_response", "trace_complete", "tool_sequence", "steps"},
    )
    if (
        not isinstance(value["final_response"], str)
        or not isinstance(value["trace_complete"], bool)
        or not isinstance(value["tool_sequence"], (list, tuple))
        or not isinstance(value["steps"], (list, tuple))
    ):
        raise PolicyContractViolation("invalid feedback schema")
    for call in value["tool_sequence"]:
        if not isinstance(call, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        _exact_keys(call, {"tool", "status"}, required={"tool", "status"})
        if not isinstance(call["tool"], str) or call["status"] not in {"ok", "error", "incomplete"}:
            raise PolicyContractViolation("invalid feedback schema")
    reason_classes = {
        "authority_channel_mismatch",
        "injection_detected",
        "privacy_or_exfiltration",
        "safety_policy_refusal",
        "task_or_target_mismatch",
        "tool_or_surface_not_reached",
        "payload_not_presented",
        "payload_ambiguous",
        "insufficient_evidence",
    }
    unknown_reasons = {
        "trace_incomplete",
        "adapter_unsupported",
        "identity_unavailable",
        "result_incomplete",
        "result_truncated",
        "message_boundary_unavailable",
        "skill_event_unavailable",
        "instrumentation_unavailable",
    }
    effects = {"followed", "partially_followed", "rejected", "ignored", "unclear"}
    seen_steps: set[int] = set()
    for step in value["steps"]:
        if not isinstance(step, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        required = {"step_index", "injection_type", "target_state"}
        allowed = required | {
            "matched_tool",
            "call_indices",
            "response_match",
            "presentation",
            "unknown_reasons",
            "effect",
            "reason_classes",
            "confidence",
        }
        _exact_keys(step, allowed, required=required)
        index = step["step_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in seen_steps
            or step["injection_type"] not in {"prompt", "tool", "environment", "skill"}
            or step["target_state"]
            not in {
                "called",
                "not_called",
                "accessed",
                "not_accessed",
                "used",
                "not_used",
                "error",
                "incomplete",
                "unknown",
                "not_applicable",
            }
        ):
            raise PolicyContractViolation("invalid feedback schema")
        seen_steps.add(index)
        if "matched_tool" in step and not isinstance(step["matched_tool"], str):
            raise PolicyContractViolation("invalid feedback schema")
        indices = step.get("call_indices", ())
        if not isinstance(indices, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in indices
        ):
            raise PolicyContractViolation("invalid feedback schema")
        reasons = step.get("unknown_reasons", ())
        if not isinstance(reasons, (list, tuple)) or any(item not in unknown_reasons for item in reasons):
            raise PolicyContractViolation("invalid feedback schema")
        if step.get("response_match", "unknown") not in {"matched", "not_matched", "unknown"}:
            raise PolicyContractViolation("invalid feedback schema")
        if step.get("presentation", "unknown") not in {"presented", "not_presented", "unknown"}:
            raise PolicyContractViolation("invalid feedback schema")
        attribution_keys = {"effect", "reason_classes", "confidence"}.intersection(step)
        if attribution_keys and attribution_keys != {"effect", "reason_classes", "confidence"}:
            raise PolicyContractViolation("invalid feedback schema")
        if attribution_keys:
            classes = step["reason_classes"]
            if (
                step["effect"] not in effects
                or step["confidence"] not in {"low", "medium", "high"}
                or not isinstance(classes, (list, tuple))
                or not classes
                or any(item not in reason_classes for item in classes)
            ):
                raise PolicyContractViolation("invalid feedback schema")
    guidance = value.get("guidance")
    if guidance is not None:
        if not isinstance(guidance, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        keys = {"diagnosis", "overall_effect", "confidence", "preserve", "reconsider"}
        _exact_keys(guidance, keys, required=keys)
        if (
            not isinstance(guidance["diagnosis"], str)
            or guidance["overall_effect"] not in effects
            or guidance["confidence"] not in {"low", "medium", "high"}
            or not isinstance(guidance["preserve"], (list, tuple))
            or not isinstance(guidance["reconsider"], (list, tuple))
            or any(
                not isinstance(item, str)
                for item in (*guidance["preserve"], *guidance["reconsider"])
            )
        ):
            raise PolicyContractViolation("invalid feedback schema")
    summary = value.get("reasoning_summary")
    if summary is not None:
        if not isinstance(summary, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        _exact_keys(summary, {"source", "summary"}, required={"source", "summary"})
        if (
            summary["source"] not in {"explicit_reasoning", "assistant_rationale"}
            or not isinstance(summary["summary"], str)
            or not summary["summary"].strip()
        ):
            raise PolicyContractViolation("invalid feedback schema")
    return value


def _validate_signal_feedback(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the unversioned, signal-only current policy projection."""

    allowed = {
        "final_response",
        "tool_sequence",
        "steps",
        "tool_analysis",
        "guidance",
        "trace_incomplete",
        "reasoning_summary",
    }
    _exact_keys(value, allowed, required={"final_response", "tool_sequence"})
    if not isinstance(value["final_response"], str) or not isinstance(
        value["tool_sequence"], (list, tuple)
    ):
        raise PolicyContractViolation("invalid feedback schema")
    tools: list[str] = []
    for call in value["tool_sequence"]:
        if not isinstance(call, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        _exact_keys(call, {"tool", "status"}, required={"tool", "status"})
        if not isinstance(call["tool"], str) or call["status"] not in {
            "ok",
            "error",
            "incomplete",
        }:
            raise PolicyContractViolation("invalid feedback schema")
        tools.append(call["tool"])
    if "trace_incomplete" in value and value["trace_incomplete"] is not True:
        raise PolicyContractViolation("invalid feedback schema")

    steps = value.get("steps", ())
    if not isinstance(steps, (list, tuple)):
        raise PolicyContractViolation("invalid feedback schema")
    seen_steps: set[int] = set()
    for step in steps:
        if not isinstance(step, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        required = {"step_index"}
        step_allowed = required | {
            "target_state",
            "matched_tool",
            "call_indices",
            "response_match",
        }
        _exact_keys(step, step_allowed, required=required)
        index = step["step_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in seen_steps
            or set(step) == required
        ):
            raise PolicyContractViolation("invalid feedback schema")
        seen_steps.add(index)
        if step.get("target_state") not in {
            None,
            "called",
            "not_called",
            "accessed",
            "not_accessed",
            "used",
            "not_used",
            "error",
            "incomplete",
        }:
            raise PolicyContractViolation("invalid feedback schema")
        if "matched_tool" in step and not isinstance(step["matched_tool"], str):
            raise PolicyContractViolation("invalid feedback schema")
        indices = step.get("call_indices", ())
        if not isinstance(indices, (list, tuple)) or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            or item >= len(tools)
            for item in indices
        ):
            raise PolicyContractViolation("invalid feedback schema")
        if step.get("response_match") not in {None, "matched", "not_matched"}:
            raise PolicyContractViolation("invalid feedback schema")

    analysis = value.get("tool_analysis", ())
    if not isinstance(analysis, (list, tuple)):
        raise PolicyContractViolation("invalid feedback schema")
    seen_calls: set[int] = set()
    for item in analysis:
        if not isinstance(item, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        required = {"call_index", "tool", "summary"}
        _exact_keys(item, required | {"injection_handling"}, required=required)
        call_index = item["call_index"]
        if (
            isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or call_index < 0
            or call_index >= len(tools)
            or call_index in seen_calls
            or item["tool"] != tools[call_index]
            or not isinstance(item["summary"], str)
            or not item["summary"].strip()
            or len(item["summary"]) > 1_000
            or item.get("injection_handling")
            not in {None, "followed", "partially_followed", "rejected", "ignored"}
        ):
            raise PolicyContractViolation("invalid feedback schema")
        seen_calls.add(call_index)

    guidance = value.get("guidance")
    if guidance is not None:
        if not isinstance(guidance, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        _exact_keys(
            guidance,
            {"diagnosis", "preserve", "reconsider"},
            required={"diagnosis"},
        )
        if not isinstance(guidance["diagnosis"], str) or not guidance["diagnosis"].strip():
            raise PolicyContractViolation("invalid feedback schema")
        for key in ("preserve", "reconsider"):
            if key in guidance and (
                not isinstance(guidance[key], (list, tuple))
                or not guidance[key]
                or any(not isinstance(pointer, str) for pointer in guidance[key])
            ):
                raise PolicyContractViolation("invalid feedback schema")
    summary = value.get("reasoning_summary")
    if summary is not None:
        if not isinstance(summary, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        _exact_keys(summary, {"source", "summary"}, required={"source", "summary"})
        if (
            summary["source"] not in {"explicit_reasoning", "assistant_rationale"}
            or not isinstance(summary["summary"], str)
            or not summary["summary"].strip()
        ):
            raise PolicyContractViolation("invalid feedback schema")
    return value


def _validate_feedback_projection(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the M7 DTO before it crosses the existing leakage guard."""
    value = dict(raw)
    if "schema_version" not in value:
        return _validate_signal_feedback(value)
    if value.get("schema_version") == 4:
        return _validate_feedback_v4(value)
    _exact_keys(
        value,
        {"schema_version", "final_response", "deterministic", "digest", "reasoning_summary"},
        required={"schema_version", "final_response"},
    )
    schema_version = value["schema_version"]
    if schema_version not in {1, 2, 3} or not isinstance(value["final_response"], str):
        raise PolicyContractViolation("invalid feedback schema")
    deterministic = value.get("deterministic")
    if deterministic is not None:
        if not isinstance(deterministic, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        _exact_keys(
            deterministic,
            {"tool_sequence", "injections", "trace_complete"},
            required={"tool_sequence", "injections", "trace_complete"},
        )
        if not isinstance(deterministic["trace_complete"], bool):
            raise PolicyContractViolation("invalid feedback schema")
        if not isinstance(deterministic["tool_sequence"], (list, tuple)):
            raise PolicyContractViolation("invalid feedback schema")
        for call in deterministic["tool_sequence"]:
            if not isinstance(call, Mapping):
                raise PolicyContractViolation("invalid feedback schema")
            _exact_keys(call, {"index", "tool", "status"}, required={"index", "tool", "status"})
            if (
                not isinstance(call["index"], int)
                or isinstance(call["index"], bool)
                or not isinstance(call["tool"], str)
                or call["status"] not in {"ok", "error", "incomplete"}
            ):
                raise PolicyContractViolation("invalid feedback schema")
        if not isinstance(deterministic["injections"], (list, tuple)):
            raise PolicyContractViolation("invalid feedback schema")
        injection_keys = {
            "step_index",
            "injection_type",
            "polluted_tool_called",
            "injected_target_accessed",
            "access_state",
            "matched_tool",
            "match_basis",
            "locator_targeted",
            "access_call_status",
            "response_contains_injection",
            "presented_to_model",
        }
        if schema_version >= 2:
            injection_keys |= {
                "evidence_call_indices",
                "response_match_state",
                "presentation_state",
                "skill_use_state",
                "unknown_reasons",
            }
        if schema_version >= 3:
            injection_keys.add("presentation_observation_source")
        for item in deterministic["injections"]:
            if not isinstance(item, Mapping):
                raise PolicyContractViolation("invalid feedback schema")
            _exact_keys(item, injection_keys, required=injection_keys)
            if item["injection_type"] not in {"prompt", "tool", "environment", "skill"}:
                raise PolicyContractViolation("invalid feedback schema")
            if not isinstance(item["step_index"], int) or isinstance(item["step_index"], bool):
                raise PolicyContractViolation("invalid feedback schema")
            for boolean_name in (
                "polluted_tool_called",
                "injected_target_accessed",
                "locator_targeted",
                "response_contains_injection",
                "presented_to_model",
            ):
                if item[boolean_name] is not None and not isinstance(item[boolean_name], bool):
                    raise PolicyContractViolation("invalid feedback schema")
            if item["matched_tool"] is not None and not isinstance(item["matched_tool"], str):
                raise PolicyContractViolation("invalid feedback schema")
            if item["match_basis"] not in {
                "qualified_tool",
                "exact_hashed_arguments",
                "injection_receipt",
                "exact_locator",
                "collection_locator",
                "payload_probe",
                "message_correlation",
                "structured_skill_event",
                "not_supported",
                "not_applicable",
                None,
            }:
                raise PolicyContractViolation("invalid feedback schema")
            if item["access_state"] not in {"accessed", "not_accessed", "unknown", "not_applicable"}:
                raise PolicyContractViolation("invalid feedback schema")
            if item["access_call_status"] not in {"ok", "error", "incomplete", "not_observed", "not_applicable"}:
                raise PolicyContractViolation("invalid feedback schema")
            if schema_version >= 2:
                if (
                    not isinstance(item["evidence_call_indices"], (list, tuple))
                    or any(
                        not isinstance(index, int) or isinstance(index, bool) or index < 0
                        for index in item["evidence_call_indices"]
                    )
                    or item["response_match_state"] not in {"matched", "not_matched", "unknown", "not_applicable"}
                    or item["presentation_state"] not in {"presented", "not_presented", "unknown", "not_applicable"}
                    or item["skill_use_state"] not in {"used", "not_used", "unknown", "not_applicable"}
                    or not isinstance(item["unknown_reasons"], (list, tuple))
                    or any(
                        reason
                        not in {
                            "trace_incomplete",
                            "adapter_unsupported",
                            "identity_unavailable",
                            "result_incomplete",
                            "result_truncated",
                            "message_boundary_unavailable",
                            "skill_event_unavailable",
                            "instrumentation_unavailable",
                        }
                        for reason in item["unknown_reasons"]
                    )
                ):
                    raise PolicyContractViolation("invalid feedback schema")
                expected_response = {True: "matched", False: "not_matched", None: "unknown"}[
                    item["response_contains_injection"]
                ]
                if item["injection_type"] == "environment" and item["response_match_state"] != expected_response:
                    raise PolicyContractViolation("inconsistent feedback schema")
                expected_presentation = {True: "presented", False: "not_presented", None: "unknown"}[
                    item["presented_to_model"]
                ]
                if item["presentation_state"] != expected_presentation:
                    raise PolicyContractViolation("inconsistent feedback schema")
            if schema_version >= 3 and item["presentation_observation_source"] not in {
                "provider_request",
                "runtime_context",
                "transcript_snapshot",
                "unavailable",
                "not_applicable",
            }:
                raise PolicyContractViolation("invalid feedback schema")
            if schema_version >= 3 and (
                item["presentation_observation_source"] in {"provider_request", "runtime_context", "transcript_snapshot"}
            ) != isinstance(item["presented_to_model"], bool):
                raise PolicyContractViolation("inconsistent feedback schema")
    digest = value.get("digest")
    if digest is not None:
        if not isinstance(digest, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        digest_keys = {"diagnosis", "preserve", "reconsider", "confidence"}
        if schema_version >= 2:
            digest_keys |= {"payload_effect", "evidence_refs"}
        if schema_version >= 3:
            digest_keys.add("step_attributions")
        _exact_keys(digest, digest_keys, required=digest_keys)
        if (
            not isinstance(digest["diagnosis"], str)
            or not isinstance(digest["preserve"], (list, tuple))
            or not isinstance(digest["reconsider"], (list, tuple))
            or digest["confidence"] not in {"low", "medium", "high"}
        ):
            raise PolicyContractViolation("invalid feedback schema")
        if any(not isinstance(path, str) for path in (*digest["preserve"], *digest["reconsider"])):
            raise PolicyContractViolation("invalid feedback schema")
        if schema_version >= 2 and (
            digest["payload_effect"] not in {"followed", "partially_followed", "rejected", "ignored", "unclear"}
            or not isinstance(digest["evidence_refs"], (list, tuple))
            or any(not isinstance(path, str) for path in digest["evidence_refs"])
        ):
            raise PolicyContractViolation("invalid feedback schema")
        if schema_version >= 3:
            attributions = digest["step_attributions"]
            if not isinstance(attributions, (list, tuple)) or len(attributions) > 32:
                raise PolicyContractViolation("invalid feedback schema")
            seen_steps: set[int] = set()
            allowed_reasons = {
                "authority_channel_mismatch",
                "injection_detected",
                "privacy_or_exfiltration",
                "safety_policy_refusal",
                "task_or_target_mismatch",
                "tool_or_surface_not_reached",
                "payload_not_presented",
                "payload_ambiguous",
                "insufficient_evidence",
            }
            for item in attributions:
                if not isinstance(item, Mapping):
                    raise PolicyContractViolation("invalid feedback schema")
                keys = {"step_index", "effect", "reason_classes", "confidence", "evidence_refs"}
                _exact_keys(item, keys, required=keys)
                step_index = item["step_index"]
                if (
                    isinstance(step_index, bool)
                    or not isinstance(step_index, int)
                    or step_index < 0
                    or step_index in seen_steps
                    or item["effect"] not in {"followed", "partially_followed", "rejected", "ignored", "unclear"}
                    or item["confidence"] not in {"low", "medium", "high"}
                    or not isinstance(item["reason_classes"], (list, tuple))
                    or not item["reason_classes"]
                    or len(item["reason_classes"]) > 4
                    or len(set(item["reason_classes"])) != len(item["reason_classes"])
                    or any(reason not in allowed_reasons for reason in item["reason_classes"])
                    or not isinstance(item["evidence_refs"], (list, tuple))
                    or not item["evidence_refs"]
                    or len(item["evidence_refs"]) > 8
                    or len(set(item["evidence_refs"])) != len(item["evidence_refs"])
                    or any(
                        not isinstance(ref, str)
                        or re.fullmatch(
                            r"/(?:final_response|deterministic/(?:injections|tool_sequence)/(?:0|[1-9][0-9]*)(?:/.*)?)",
                            ref,
                        )
                        is None
                        for ref in item["evidence_refs"]
                    )
                ):
                    raise PolicyContractViolation("invalid feedback schema")
                seen_steps.add(step_index)
    summary = value.get("reasoning_summary")
    if summary is not None:
        if not isinstance(summary, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        _exact_keys(summary, {"source", "summary"}, required={"source", "summary"})
        if summary["source"] not in {"explicit_reasoning", "assistant_rationale", "unavailable"} or not isinstance(
            summary["summary"], str
        ):
            raise PolicyContractViolation("invalid feedback schema")
    return value


@dataclass(frozen=True)
class PolicyLeakageGuard:
    secrets: tuple[str, ...] = field(default_factory=tuple, repr=False)
    forbidden_fragments: tuple[str, ...] = ()
    forbidden_keys: frozenset[str] = _FORBIDDEN_KEYS
    max_response_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        if self.max_response_bytes < 256:
            raise ValueError("max_response_bytes is too small")

    def validate(self, payload: Any) -> dict[str, Any]:
        self._check_keys(payload)
        encoded = canonical_policy_json(payload)
        if len(encoded.encode("utf-8")) > self.max_response_bytes:
            raise PolicyContractViolation("policy response exceeded configured limit")
        needles: Iterable[str] = (*self.secrets, *self.forbidden_fragments)
        for needle in needles:
            if isinstance(needle, str) and len(needle) >= 8 and needle in encoded:
                raise PolicyContractViolation("policy response matched forbidden material")
        if any(pattern.search(encoded) for pattern in _SENSITIVE_VALUE_PATTERNS):
            raise PolicyContractViolation("policy response matched sensitive material")
        parsed = json.loads(encoded)
        if not isinstance(parsed, dict):
            raise PolicyContractViolation("policy response must be an object")
        return parsed

    def _check_keys(self, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in self.forbidden_keys:
                    raise PolicyContractViolation("policy response contains forbidden field")
                self._check_keys(item)
        elif isinstance(value, list):
            for item in value:
                self._check_keys(item)


@dataclass(frozen=True)
class PolicyContract:
    guard: PolicyLeakageGuard = PolicyLeakageGuard()
    improvement_wishes: bool = False
    dying_message: bool = False

    def public_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.guard.validate(dict(payload))

    def accepted_submit(
        self,
        *,
        submission: int,
        success: bool,
        terminal: bool,
        remaining_submissions: int,
        feedback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(success, bool):
            raise PolicyContractViolation("success must be boolean")
        payload: dict[str, Any] = {
            "accepted": True,
            "submission": submission,
            "success": success,
            "terminal": bool(terminal),
            "remaining_submissions": remaining_submissions,
            "report_required": {
                "h": submission,
                "before_next_tool": True,
                "fields": [
                    "outcome",
                    "evidence_and_uncertainty",
                    "next_policy_change",
                    *(["policy_harness_improvement_wish"] if self.improvement_wishes else []),
                    *(["dying_message"] if self.dying_message and terminal else []),
                ],
            },
        }
        if feedback is not None:
            payload["feedback"] = _validate_feedback_projection(feedback)
        return self.public_payload(payload)

    def rejected_submit(
        self,
        *,
        code: str,
        terminal: bool,
        remaining_submissions: int,
    ) -> dict[str, Any]:
        allowed = {
            "INVALID_SUBMISSION",
            "EPISODE_TERMINAL",
            "POLICY_LIMIT",
            "EVALUATION_UNAVAILABLE",
        }
        if code not in allowed:
            code = "INVALID_SUBMISSION"
        return self.public_payload(
            {
                "accepted": False,
                "terminal": bool(terminal),
                "remaining_submissions": remaining_submissions,
                "error": {"code": code},
            }
        )

    def from_internal_submit(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        if receipt.get("accepted") is True:
            return self.accepted_submit(
                submission=int(receipt["submission"]),
                success=receipt["success"],
                terminal=bool(receipt["terminal"]),
                remaining_submissions=int(receipt["remaining_submissions"]),
                feedback=(receipt.get("feedback") if isinstance(receipt.get("feedback"), Mapping) else None),
            )
        raw_code = None
        error = receipt.get("error")
        if isinstance(error, Mapping):
            raw_code = error.get("code")
        errors = receipt.get("errors")
        if raw_code is None and isinstance(errors, list) and errors and isinstance(errors[0], Mapping):
            raw_code = errors[0].get("code")
        if raw_code in {"EPISODE_TERMINAL"}:
            code = "EPISODE_TERMINAL"
        elif raw_code in {"POLICY_LIMIT"}:
            code = "POLICY_LIMIT"
        elif raw_code in {"INFRA_ERROR", "SECURITY_ABORT", "EVALUATION_UNAVAILABLE"}:
            code = "EVALUATION_UNAVAILABLE"
        else:
            code = "INVALID_SUBMISSION"
        return self.rejected_submit(
            code=code,
            terminal=bool(receipt.get("terminal", False)),
            remaining_submissions=int(receipt.get("remaining_submissions", 0)),
        )
