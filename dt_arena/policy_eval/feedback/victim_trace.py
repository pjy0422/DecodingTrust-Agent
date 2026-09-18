"""Allowlisted views over victim artifacts; hidden benchmark metadata is ignored."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deterministic import ParsedMCPTrace, _qualified, _status
from .schema import VictimTraceItem, VictimVisibleTrace


_SENSITIVE_PATTERNS = (
    re.compile(r"(?i)\bbearer[ ]+[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b[a-f0-9]{32}\.[A-Za-z0-9_-]{12,}\b", re.I),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def sanitize_trace_value(
    value: Any,
    *,
    redactions: tuple[str, ...] = (),
    max_chars: int = 16_000,
    depth: int = 0,
) -> Any:
    if depth > 6:
        return "<truncated>"
    if isinstance(value, str):
        result = value[:max_chars]
        for secret in redactions:
            if isinstance(secret, str) and len(secret) >= 6:
                result = result.replace(secret, "<redacted>")
        for pattern in _SENSITIVE_PATTERNS:
            result = pattern.sub("<redacted>", result)
        return result
    if isinstance(value, list):
        return [
            sanitize_trace_value(item, redactions=redactions, max_chars=max_chars, depth=depth + 1)
            for item in value[:100]
        ]
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_trace_value(item, redactions=redactions, max_chars=max_chars, depth=depth + 1)
            for key, item in list(value.items())[:100]
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "<unsupported>"


def load_trajectory(path: Path, *, max_bytes: int = 8 * 1024 * 1024) -> Mapping[str, Any]:
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError("victim trajectory exceeds configured limit")
    value = json.loads(data)
    if not isinstance(value, Mapping):
        raise ValueError("victim trajectory must be an object")
    return value


def extract_final_response(artifact: Mapping[str, Any], *, max_chars: int = 16_000) -> str:
    traj_info = artifact.get("traj_info")
    if isinstance(traj_info, Mapping):
        value = traj_info.get("agent_final_response")
        if isinstance(value, str):
            return value[:max_chars]
    trajectory = artifact.get("trajectory")
    if isinstance(trajectory, list):
        for item in reversed(trajectory):
            if not isinstance(item, Mapping):
                continue
            role = item.get("role")
            metadata = item.get("metadata")
            candidates = (
                item.get("content"),
                item.get("state") if role in {"assistant", "agent"} else None,
                metadata.get("message") if isinstance(metadata, Mapping) else None,
            )
            for candidate in candidates:
                if isinstance(candidate, str):
                    return candidate[:max_chars]
    return ""


def build_victim_trace(
    artifact: Mapping[str, Any],
    mcp_trace: ParsedMCPTrace,
    *,
    include_reasoning: bool,
    max_item_chars: int = 16_000,
    max_items: int = 1_000,
    redactions: tuple[str, ...] = (),
) -> VictimVisibleTrace:
    items: list[VictimTraceItem] = []
    reasoning_seen = False
    assistant_rationale_seen = False
    trajectory = artifact.get("trajectory")
    if isinstance(trajectory, list):
        for index, raw in enumerate(trajectory[:max_items]):
            if not isinstance(raw, Mapping):
                continue
            role = str(raw.get("role") or "").lower()
            metadata = raw.get("metadata")
            kind = None
            content: Any = None
            if role == "user":
                kind = "user"
                content = raw.get("content", raw.get("state"))
            elif role in {"assistant", "agent"}:
                kind = "assistant"
                content = raw.get("content")
                if content is None and isinstance(metadata, Mapping):
                    content = metadata.get("message")
                if include_reasoning and isinstance(metadata, Mapping):
                    rationale = metadata.get("reasoning")
                    if isinstance(rationale, str) and rationale:
                        items.append(
                            VictimTraceItem(
                                "provider_reasoning",
                                index,
                                "reasoning",
                                rationale[:max_item_chars],
                            )
                        )
                        assistant_rationale_seen = True
            elif role in {"reasoning", "analysis"} and include_reasoning:
                kind = "reasoning"
                content = raw.get("content", raw.get("state"))
                reasoning_seen = True
            elif role in {"tool", "tool_result"}:
                kind = "tool_result"
                content = raw.get("content", raw.get("state"))
            if kind is None or not isinstance(content, (str, dict, list)):
                continue
            content = sanitize_trace_value(content, redactions=redactions, max_chars=max_item_chars)
            items.append(VictimTraceItem("trajectory", index, kind, content))  # type: ignore[arg-type]

    # MCP records are an execution stream, not reconstructed model messages.
    for index, event in enumerate(mcp_trace.started):
        if len(items) >= max_items:
            break
        items.append(
            VictimTraceItem(
                "mcp_event",
                index,
                "tool_call",
                {"tool": _qualified(event), "status": _status(event, mcp_trace.completed)},
            )
        )
    reasoning_source = (
        "explicit_reasoning"
        if include_reasoning and reasoning_seen
        else (
            "assistant_rationale"
            if include_reasoning and assistant_rationale_seen
            else "unavailable" if include_reasoning else "disabled"
        )
    )
    return VictimVisibleTrace(
        tuple(items),
        interaction_complete=mcp_trace.trace_complete,
        reasoning_source=reasoning_source,  # type: ignore[arg-type]
    )
