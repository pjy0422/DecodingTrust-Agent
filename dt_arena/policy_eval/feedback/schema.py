"""Typed internal and policy-visible M7 feedback data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from collections.abc import Mapping
from typing import Any, Literal


class FeedbackMode(str, Enum):
    DISABLED = "disabled"
    FINAL_ONLY = "final"
    FINAL_DETERMINISTIC = "final+deterministic"
    FINAL_DETERMINISTIC_DIGESTOR = "final+deterministic+digestor"


@dataclass(frozen=True)
class ReasoningSummaryConfig:
    """An independent opt-in inside mode C; it is never silently enabled."""

    enabled: bool = False
    timeout_seconds: float = 20.0
    max_chars: int = 2_000

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.max_chars < 64:
            raise ValueError("invalid reasoning-summary limits")


@dataclass(frozen=True)
class ToolCallObservation:
    index: int
    tool: str
    status: Literal["ok", "error", "incomplete"]


@dataclass(frozen=True)
class InjectionObservation:
    step_index: int
    injection_type: Literal["prompt", "tool", "environment", "skill"]
    polluted_tool_called: bool | None
    injected_target_accessed: bool | None
    access_state: Literal["accessed", "not_accessed", "unknown", "not_applicable"]
    matched_tool: str | None = None
    match_basis: (
        Literal[
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
        ]
        | None
    ) = None
    # These fields prevent `accessed` from being mistaken for content exposure.
    locator_targeted: bool | None = None
    access_call_status: Literal["ok", "error", "incomplete", "not_observed", "not_applicable"] = "not_applicable"
    response_contains_injection: bool | None = None
    presented_to_model: bool | None = None
    evidence_call_indices: tuple[int, ...] = ()
    response_match_state: Literal["matched", "not_matched", "unknown", "not_applicable"] = "not_applicable"
    presentation_state: Literal["presented", "not_presented", "unknown", "not_applicable"] = "not_applicable"
    presentation_observation_source: Literal[
        "provider_request", "runtime_context", "transcript_snapshot", "unavailable", "not_applicable"
    ] = "unavailable"
    skill_use_state: Literal["used", "not_used", "unknown", "not_applicable"] = "not_applicable"
    unknown_reasons: tuple[
        Literal[
            "trace_incomplete",
            "adapter_unsupported",
            "identity_unavailable",
            "result_incomplete",
            "result_truncated",
            "message_boundary_unavailable",
            "skill_event_unavailable",
            "instrumentation_unavailable",
        ],
        ...,
    ] = ()


@dataclass(frozen=True)
class DeterministicFeedback:
    tool_sequence: tuple[ToolCallObservation, ...]
    injections: tuple[InjectionObservation, ...]
    trace_complete: bool


@dataclass(frozen=True)
class VictimTraceItem:
    source: Literal["trajectory", "mcp_event", "provider_reasoning"]
    source_index: int
    kind: Literal["user", "reasoning", "tool_call", "tool_result", "assistant"]
    content: Any


@dataclass(frozen=True)
class VictimVisibleTrace:
    """Allowlisted evidence streams, not a claim of a complete model transcript."""

    items: tuple[VictimTraceItem, ...]
    interaction_complete: bool
    reasoning_source: Literal["explicit_reasoning", "assistant_rationale", "unavailable", "disabled"]


DigestEffect = Literal["followed", "partially_followed", "rejected", "ignored", "unclear"]
DigestConfidence = Literal["low", "medium", "high"]
DigestReasonClass = Literal[
    "authority_channel_mismatch",
    "injection_detected",
    "privacy_or_exfiltration",
    "safety_policy_refusal",
    "task_or_target_mismatch",
    "tool_or_surface_not_reached",
    "payload_not_presented",
    "payload_ambiguous",
    "insufficient_evidence",
]


@dataclass(frozen=True)
class StepAttribution:
    """Bounded semantic attribution for one submitted step, never causal proof."""

    step_index: int
    effect: DigestEffect
    reason_classes: tuple[DigestReasonClass, ...]
    confidence: DigestConfidence
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class RepairDigest:
    diagnosis: str
    preserve: tuple[str, ...]
    reconsider: tuple[str, ...]
    confidence: DigestConfidence
    payload_effect: DigestEffect = "unclear"
    evidence_refs: tuple[str, ...] = ()
    step_attributions: tuple[StepAttribution, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReasoningSummary:
    source: Literal["explicit_reasoning", "assistant_rationale", "unavailable"]
    summary: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FeedbackEvidence:
    final_response: str
    deterministic: DeterministicFeedback
    victim_trace: VictimVisibleTrace


def deterministic_to_dict(value: DeterministicFeedback) -> dict[str, Any]:
    return asdict(value)


def _compact_target_state(item: Mapping[str, Any]) -> str:
    injection_type = item.get("injection_type")
    if injection_type == "tool":
        return {True: "called", False: "not_called", None: "unknown"}.get(
            item.get("polluted_tool_called"), "unknown"
        )
    if injection_type == "skill":
        return {
            "used": "used",
            "not_used": "not_used",
            "unknown": "unknown",
        }.get(item.get("skill_use_state"), "unknown")
    if injection_type == "environment":
        status = item.get("access_call_status")
        if status in {"error", "incomplete"}:
            return str(status)
        return {
            "accessed": "accessed",
            "not_accessed": "not_accessed",
            "unknown": "unknown",
        }.get(item.get("access_state"), "unknown")
    return "not_applicable"


def compact_feedback_v4(full: Mapping[str, Any]) -> dict[str, Any]:
    """Project detailed sanitized v3 evidence into the compact policy DTO."""

    deterministic = full.get("deterministic")
    deterministic = deterministic if isinstance(deterministic, Mapping) else {}
    digest = full.get("digest")
    digest = digest if isinstance(digest, Mapping) else {}
    attributions = digest.get("step_attributions", ())
    attribution_by_step = {
        item.get("step_index"): item
        for item in attributions
        if isinstance(item, Mapping) and isinstance(item.get("step_index"), int)
    }
    steps = []
    for raw in deterministic.get("injections", ()):
        if not isinstance(raw, Mapping):
            continue
        item: dict[str, Any] = {
            "step_index": raw.get("step_index"),
            "injection_type": raw.get("injection_type"),
            "target_state": _compact_target_state(raw),
        }
        for source, target in (
            ("matched_tool", "matched_tool"),
            ("evidence_call_indices", "call_indices"),
            ("unknown_reasons", "unknown_reasons"),
        ):
            value = raw.get(source)
            if value not in (None, (), []):
                item[target] = value
        if raw.get("response_match_state") != "not_applicable":
            item["response_match"] = raw.get("response_match_state", "unknown")
        if raw.get("presentation_state") != "not_applicable":
            item["presentation"] = raw.get("presentation_state", "unknown")
        attribution = attribution_by_step.get(raw.get("step_index"))
        if attribution is not None:
            item.update(
                {
                    "effect": attribution.get("effect"),
                    "reason_classes": attribution.get("reason_classes"),
                    "confidence": attribution.get("confidence"),
                }
            )
        steps.append(item)
    result: dict[str, Any] = {
        "schema_version": 4,
        "final_response": full.get("final_response", ""),
        "trace_complete": bool(deterministic.get("trace_complete", False)),
        "tool_sequence": [
            {"tool": item.get("tool"), "status": item.get("status")}
            for item in deterministic.get("tool_sequence", ())
            if isinstance(item, Mapping)
        ],
        "steps": steps,
    }
    if digest:
        result["guidance"] = {
            "diagnosis": digest.get("diagnosis"),
            "overall_effect": digest.get("payload_effect"),
            "confidence": digest.get("confidence"),
            "preserve": digest.get("preserve", ()),
            "reconsider": digest.get("reconsider", ()),
        }
    reasoning = full.get("reasoning_summary")
    if (
        isinstance(reasoning, Mapping)
        and reasoning.get("source") != "unavailable"
        and isinstance(reasoning.get("summary"), str)
        and reasoning["summary"].strip()
    ):
        result["reasoning_summary"] = dict(reasoning)
    return result
