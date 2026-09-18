"""Parse DTAP victim and attack-policy traces into one viewer data model."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any


def find_openclaw_trace(root: str | Path) -> Path | None:
    """Find a DTAP victim OpenClaw runtime trace without selecting policy JSONL."""
    root = Path(root)
    if root.is_file():
        return root
    hits = [
        path
        for path in sorted(root.rglob("traces/openclaw_runtime/*.jsonl"))
        if not path.name.endswith(".mcp-events.jsonl")
    ]
    if hits:
        return hits[0]
    hits = [
        p
        for p in sorted(root.rglob("*.jsonl"))
        if "policy" not in p.name.lower() and not p.name.endswith("mcp-events.jsonl")
    ]
    return hits[0] if hits else None


def find_victim_mcp_events(root: str | Path) -> Path | None:
    root = Path(root)
    if root.is_file():
        return root if root.name.endswith("mcp-events.jsonl") else None
    hits = sorted(root.rglob("*mcp-events.jsonl"))
    return hits[0] if hits else None


def find_victim_trace(root: str | Path) -> Path | None:
    """Find either an OpenClaw JSONL or DTAP's framework-neutral trajectory JSON."""
    root = Path(root)
    if root.is_file():
        return root
    openclaw = find_openclaw_trace(root)
    if openclaw is not None:
        return openclaw
    for pattern in ("victim-trajectory.json", "*trajectory*.json", "*.json"):
        for candidate in sorted(root.rglob(pattern)):
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(value, dict) and isinstance(value.get("trajectory"), list):
                return candidate
    return None


def find_policy_trace(root: str | Path) -> Path | None:
    """Find Claude Code ``--output-format stream-json`` captured for the policy."""
    root = Path(root)
    if root.is_file():
        return root
    patterns = (
        "policy.jsonl",
        "policy-trajectory.jsonl",
        "traces/policy/*.jsonl",
        "*policy*.jsonl",
    )
    for pattern in patterns:
        hits = sorted(root.rglob(pattern))
        if hits:
            return hits[0]
    return None


def extract_attack_payloads(yaml_path: str | Path | None) -> list[dict[str, Any]]:
    """Extract authored payload strings from a DTAP config/attack YAML."""
    if yaml_path is None:
        return []
    try:
        import yaml

        cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    attack = cfg.get("Attack") if isinstance(cfg, dict) else None
    if not attack:
        attack = cfg
    out: list[dict[str, Any]] = []
    if not isinstance(attack, dict):
        return out
    for turn in attack.get("attack_turns", []) or []:
        if not isinstance(turn, dict):
            continue
        for step in turn.get("attack_steps", []) or []:
            if not isinstance(step, dict):
                continue
            kind = step.get("type")
            common = {
                "kind": kind,
                "mode": step.get("mode"),
                "tool": step.get("injected_tool") or step.get("injection_mcp_tool") or step.get("tool"),
                "skill": step.get("skill_name"),
            }
            if step.get("content") is not None:
                out.append({**common, "text": str(step["content"])})
            for field, value in (step.get("kwargs") or {}).items():
                if isinstance(value, str) and len(value) > 10:
                    out.append({**common, "field": field, "text": value})
    return out


def extract_tool_descriptions(trace_path: str | Path, tool_quals: set[str]) -> dict[str, str]:
    """Pull post-injection descriptions embedded in an OpenClaw trace."""
    if not tool_quals:
        return {}
    try:
        raw = Path(trace_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for qual in tool_quals:
        if ":" not in qual:
            continue
        server, name = qual.split(":", 1)
        marker = f'"name":"{server}__{name}","description":"'
        start = raw.find(marker)
        if start < 0:
            continue
        start += len(marker)
        index = start
        while index < len(raw):
            if raw[index] == "\\":
                index += 2
                continue
            if raw[index] == '"':
                break
            index += 1
        if index >= len(raw):
            continue
        try:
            out[qual] = json.loads('"' + raw[start:index] + '"')
        except Exception:
            out[qual] = raw[start:index]
    return out


def _split_tool(name: str) -> tuple[str, str]:
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return parts[1], parts[2]
    if "__" in name:
        return tuple(name.split("__", 1))  # type: ignore[return-value]
    if "." in name:
        return tuple(name.split(".", 1))  # type: ignore[return-value]
    return "", name


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") == "text":
            return str(value.get("text", ""))
        if "content" in value:
            return _text(value["content"])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _append_message(timeline: list[dict[str, Any]], message: Any) -> None:
    if not isinstance(message, dict):
        return
    role = message.get("role")
    content = message.get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if role == "user" and kind == "text":
            timeline.append({"kind": "user", "text": str(block.get("text", ""))})
        elif role in {"tool", "toolResult"} or kind == "tool_result":
            timeline.append({"kind": "tool_result", "text": _text(block.get("content", block))})
        elif role == "assistant" and kind in {"thinking", "reasoning"}:
            timeline.append({"kind": "thinking", "text": str(block.get("thinking") or block.get("text") or "")})
        elif role == "assistant" and kind in {"text", "output_text"}:
            text = str(block.get("text", ""))
            if text.strip():
                timeline.append({"kind": "say", "text": text})
        elif role == "assistant" and kind in {"toolCall", "tool_use", "server_tool_use"}:
            server, tool = _split_tool(str(block.get("name", "")))
            timeline.append(
                {
                    "kind": "tool_call",
                    "server": server,
                    "tool": tool,
                    "args": block.get("arguments") or block.get("input") or {},
                }
            )


def _mark_final(timeline: list[dict[str, Any]]) -> None:
    for event in reversed(timeline):
        if event.get("kind") == "say":
            event["kind"] = "final"
            return


def parse_openclaw_timeline(trace_path: str | Path) -> list[dict[str, Any]]:
    """Parse the final ``model.completed.messagesSnapshot`` from a DTAP trace."""
    snapshot = None
    for line in Path(trace_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue
        if event.get("type") == "model.completed":
            snapshot = event.get("data", {}).get("messagesSnapshot") or snapshot
    timeline: list[dict[str, Any]] = []
    for message in snapshot or []:
        _append_message(timeline, message)
    _mark_final(timeline)
    return timeline


def parse_dtap_trajectory(trace_path: str | Path) -> list[dict[str, Any]]:
    """Parse DTAP's framework-neutral ``Trajectory.save`` JSON representation."""
    try:
        payload = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    timeline: list[dict[str, Any]] = []
    for step in payload.get("trajectory", []) if isinstance(payload, dict) else []:
        if not isinstance(step, dict):
            continue
        role = step.get("role")
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        if role == "user":
            timeline.append({"kind": "user", "text": _text(step.get("state", ""))})
        elif role == "tool":
            timeline.append({"kind": "tool_result", "text": _text(step.get("state", ""))})
        elif role == "agent" and metadata.get("tool_name"):
            timeline.append(
                {
                    "kind": "tool_call",
                    "server": str(metadata.get("server") or ""),
                    "tool": str(metadata["tool_name"]),
                    "args": metadata.get("tool_params") or {},
                }
            )
        elif role == "agent":
            message = metadata.get("message")
            text = message if message is not None else step.get("action", "")
            if str(text).strip():
                timeline.append({"kind": "say", "text": _text(text)})
    _mark_final(timeline)
    return timeline


def parse_victim_timeline(trace_path: str | Path) -> list[dict[str, Any]]:
    path = Path(trace_path)
    return parse_dtap_trajectory(path) if path.suffix.lower() == ".json" else parse_openclaw_timeline(path)


def parse_victim_mcp_events(trace_path: str | Path) -> list[dict[str, Any]]:
    """Parse the proxy-owned redacted tool audit stream."""
    timeline: list[dict[str, Any]] = []
    for line in Path(trace_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue
        if event.get("schema") != "dtap-openclaw-mcp-event":
            continue
        common = {
            "episode_id": str(event.get("episode_id") or ""),
            "server": str(event.get("server") or ""),
            "tool": str(event.get("tool") or ""),
            "call_id": str(event.get("call_id") or ""),
            "timestamp": str(event.get("timestamp") or ""),
            "source": "mcp_proxy",
        }
        if event.get("type") == "tool.started":
            timeline.append(
                {
                    "kind": "tool_call",
                    "args": event.get("arguments") or {},
                    "arguments_digest": event.get("arguments_digest"),
                    **common,
                }
            )
        elif event.get("type") == "tool.completed":
            timeline.append(
                {
                    "kind": "tool_result",
                    "text": (
                        f"{common['server']}:{common['tool']} completed; "
                        f"is_error={bool(event.get('is_error'))}; "
                        f"result_sha256={event.get('result_digest', '')}"
                    ),
                    "is_error": bool(event.get("is_error")),
                    "result_digest": event.get("result_digest"),
                    **common,
                }
            )
    return timeline


def parse_policy_timeline(trace_path: str | Path) -> list[dict[str, Any]]:
    """Parse Claude Code stream-json without duplicating cumulative snapshots."""
    timeline: list[dict[str, Any]] = []
    seen_blocks: set[tuple[str, str]] = set()
    for line in Path(trace_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue
        event_type = event.get("type")
        if event_type not in {"assistant", "user"}:
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        # Claude reuses one message id for separately emitted thinking, text,
        # and tool-use blocks. Deduplicate only identical block payloads (as can
        # happen with partial-message streams), never the entire message id.
        signature = (
            str(message.get("id") or ""),
            json.dumps(message.get("content"), ensure_ascii=False, sort_keys=True),
        )
        if signature in seen_blocks:
            continue
        seen_blocks.add(signature)
        _append_message(timeline, message)
    _mark_final(timeline)
    return timeline


def find_payload_spans(text: str, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for payload in payloads:
        needle = str(payload.get("text") or "").strip()
        if len(needle) < 8:
            continue
        for probe in (needle, needle[:80]):
            start = text.find(probe)
            if start >= 0:
                spans.append(
                    {
                        "start": start,
                        "end": start + len(probe),
                        "kind": payload.get("kind"),
                        "mode": payload.get("mode"),
                        "tool": payload.get("tool"),
                        "field": payload.get("field"),
                    }
                )
                break
    spans.sort(key=lambda item: item["start"])
    result: list[dict[str, Any]] = []
    last_end = -1
    for span in spans:
        if span["start"] >= last_end:
            result.append(span)
            last_end = span["end"]
    return result


def mark_injections(
    timeline: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    tool_descriptions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    tool_descriptions = tool_descriptions or {}
    for event in timeline:
        spans = find_payload_spans(str(event.get("text") or ""), payloads)
        if spans:
            event["injection_spans"] = spans
        if event.get("kind") != "tool_call":
            continue
        qual = f"{event.get('server')}:{event.get('tool')}"
        arg_spans = {}
        for key, value in (event.get("args") or {}).items():
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            hits = find_payload_spans(rendered, payloads)
            if hits:
                arg_spans[key] = hits
        if arg_spans:
            event["arg_injection_spans"] = arg_spans
        description = tool_descriptions.get(qual)
        matching = [p for p in payloads if p.get("kind") == "tool" and p.get("tool") == qual]
        if description and matching:
            event["injected_tool_desc"] = {
                "text": description,
                "spans": find_payload_spans(description, matching),
                "modes": sorted({p.get("mode") or "suffix" for p in matching}),
            }
            event["injection_target"] = True
        elif any(p.get("tool") == qual for p in payloads):
            event["injection_target"] = True
    return timeline


def compare_configs(original: str | Path | None, submitted: str | Path | None) -> dict[str, Any] | None:
    if original is None or submitted is None:
        return None
    original_path, submitted_path = Path(original), Path(submitted)
    try:
        before = original_path.read_text(encoding="utf-8")
        after = submitted_path.read_text(encoding="utf-8")
    except OSError:
        return None
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="original/config.yaml",
            tofile="submitted/config.yaml",
        )
    )
    return {
        "original_path": str(original_path),
        "submitted_path": str(submitted_path),
        "original": before,
        "submitted": after,
        "identical": before == after,
        "diff": diff,
    }


def build_timeline(
    trace_path: str | Path | None,
    yaml_path: str | Path | None = None,
    meta: dict[str, Any] | None = None,
    *,
    policy_trace_path: str | Path | None = None,
    policy_prompt_path: str | Path | None = None,
    original_yaml_path: str | Path | None = None,
    submitted_yaml_path: str | Path | None = None,
    victim_mcp_events_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a combined viewer payload; ``yaml_path`` remains the submitted-YAML alias."""
    submitted = submitted_yaml_path or yaml_path
    payloads = extract_attack_payloads(submitted)
    victim_timeline: list[dict[str, Any]] = []
    descriptions: dict[str, str] = {}
    if trace_path is not None:
        victim_timeline = parse_victim_timeline(trace_path)
        targets = {p["tool"] for p in payloads if p.get("kind") == "tool" and p.get("tool")}
        descriptions = extract_tool_descriptions(trace_path, targets)
        mark_injections(victim_timeline, payloads, descriptions)
    if victim_mcp_events_path is not None:
        proxy_events = parse_victim_mcp_events(victim_mcp_events_path)
        insert_at = len(victim_timeline)
        if victim_timeline and victim_timeline[-1].get("kind") == "final":
            insert_at -= 1
        victim_timeline[insert_at:insert_at] = proxy_events
    policy_timeline = parse_policy_timeline(policy_trace_path) if policy_trace_path else []
    if policy_prompt_path is not None:
        try:
            policy_prompt = Path(policy_prompt_path).read_text(encoding="utf-8")
        except OSError:
            policy_prompt = ""
        if policy_prompt:
            policy_timeline.insert(0, {"kind": "user", "text": policy_prompt})
    mark_injections(policy_timeline, payloads)
    result = {
        "trace": str(trace_path) if trace_path else None,
        "policy_trace": str(policy_trace_path) if policy_trace_path else None,
        "policy_prompt": str(policy_prompt_path) if policy_prompt_path else None,
        "yaml": str(submitted) if submitted else None,
        "original_yaml": str(original_yaml_path) if original_yaml_path else None,
        "victim_mcp_events": (str(victim_mcp_events_path) if victim_mcp_events_path else None),
        "payloads": payloads,
        "timeline": victim_timeline,
        "policy_timeline": policy_timeline,
        "config_comparison": compare_configs(original_yaml_path, submitted),
        **(meta or {}),
    }
    event_episode_ids = {
        event["episode_id"]
        for event in victim_timeline
        if event.get("source") == "mcp_proxy" and event.get("episode_id")
    }
    if not result.get("episode_id") and len(event_episode_ids) == 1:
        result["episode_id"] = next(iter(event_episode_ids))
    elif result.get("episode_id") and event_episode_ids and event_episode_ids != {result["episode_id"]}:
        result.setdefault("trajectory_warnings", []).append("victim MCP events do not match the bundle episode_id")
    return result
