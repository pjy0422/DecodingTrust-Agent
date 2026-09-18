"""Filesystem-to-SQLite metadata indexer for DTAP trajectory bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .db import TrajectoryDB

_METADATA_FILES = (
    "result.json",
    "episode-manifest.json",
    "judge-result.json",
    "judge-verdict.json",
    "policy.jsonl",
    "victim-trajectory.json",
    "victim-mcp-events.jsonl",
    "original-config.yaml",
    "original_config.yaml",
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _victim_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    trajectory = value.get("trajectory") if isinstance(value, dict) else None
    return len(trajectory) if isinstance(trajectory, list) else None


def _number(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, int(value))


def normalize_usage(
    usage: Any,
    *,
    estimated_reasoning_tokens: int | None = None,
) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    input_tokens = _number(usage.get("input_tokens", usage.get("inputTokens", usage.get("input"))))
    output_tokens = _number(usage.get("output_tokens", usage.get("outputTokens", usage.get("output"))))
    if input_tokens is None and output_tokens is None:
        return None
    cache_read = (
        _number(
            usage.get(
                "cache_read_input_tokens",
                usage.get(
                    "cacheReadInputTokens",
                    usage.get("cacheRead", usage.get("cache_read_tokens")),
                ),
            )
        )
        or 0
    )
    cache_write = (
        _number(
            usage.get(
                "cache_creation_input_tokens",
                usage.get(
                    "cacheCreationInputTokens",
                    usage.get("cacheWrite", usage.get("cache_write_tokens")),
                ),
            )
        )
        or 0
    )
    exact_reasoning = _number(
        usage.get(
            "reasoning_tokens",
            usage.get(
                "reasoningTokens",
                _nested(usage, "output_tokens_details", "reasoning_tokens")
                or _nested(usage, "output_tokens_details", "thinking_tokens")
                or _nested(usage, "completion_tokens_details", "reasoning_tokens"),
            ),
        )
    )
    reasoning = exact_reasoning
    reasoning_source = "provider" if exact_reasoning is not None else "unavailable"
    if reasoning is None and estimated_reasoning_tokens is not None:
        reasoning = max(0, estimated_reasoning_tokens)
        reasoning_source = "stream_estimate"
    prompt_tokens = (input_tokens or 0) + cache_read + cache_write
    output_tokens = output_tokens or 0
    with_reasoning = prompt_tokens + output_tokens
    without_reasoning = prompt_tokens + max(0, output_tokens - reasoning) if reasoning is not None else None
    return {
        "input_tokens": input_tokens or 0,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning,
        "reasoning_source": reasoning_source,
        "tokens_with_reasoning": with_reasoning,
        "tokens_without_reasoning": without_reasoning,
    }


def _policy_metrics(path: Path) -> tuple[int | None, dict[str, Any] | None]:
    if not path.is_file():
        return None, None
    tool_call_ids: set[str] = set()
    anonymous_tool_calls = 0
    result_usage: dict[str, Any] | None = None
    estimated_reasoning = 0
    has_reasoning_estimate = False
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "system" and event.get("subtype") == "thinking_tokens":
                    delta = _number(event.get("estimated_tokens_delta"))
                    if delta is not None:
                        estimated_reasoning += delta
                        has_reasoning_estimate = True
                message = event.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict) or block.get("type") != "tool_use":
                                continue
                            tool_call_id = _text(block.get("id"))
                            if tool_call_id:
                                tool_call_ids.add(tool_call_id)
                            else:
                                anonymous_tool_calls += 1
                if event.get("type") == "result" and isinstance(event.get("usage"), dict):
                    result_usage = event["usage"]
    except OSError:
        return None, None
    return len(tool_call_ids) + anonymous_tool_calls, normalize_usage(
        result_usage,
        estimated_reasoning_tokens=(estimated_reasoning if has_reasoning_estimate else None),
    )


def _text(value: Any) -> str | None:
    if not isinstance(value, (str, int)):
        return None
    result = str(value).strip()
    return result or None


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _run_summary(path: Path, root: Path) -> tuple[dict[str, Any], Path | None]:
    """Return the nearest run summary without walking outside the indexed root."""
    current = path
    while True:
        candidate = current / "summary.json"
        if candidate.is_file():
            return _json(candidate), candidate
        if current == root:
            break
        try:
            current.relative_to(root)
        except ValueError:
            break
        current = current.parent
    return {}, None


def _original_task_id(path: Path) -> str | None:
    for name in ("original-config.yaml", "original_config.yaml"):
        candidate = path / name
        if not candidate.is_file():
            continue
        try:
            value = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        task_id = _text(_nested(value, "Task", "task_id"))
        if task_id:
            return task_id
    return None


def _policy_trace_model(path: Path) -> str | None:
    """Read only a bounded trace prefix; Claude stream-json records model early."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= 100:
                    break
                try:
                    event = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                for value in (
                    event.get("model"),
                    _nested(event, "message", "model"),
                    _nested(event, "modelUsage", "canonicalModel"),
                ):
                    if model := _text(value):
                        return model
    except OSError:
        return None
    return None


def _victim_trace_metadata(
    path: Path,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    value = _json(path)
    task_id = _text(_nested(value, "task_info", "task_id"))
    model = next(
        (
            found
            for candidate in (
                value.get("victim_model"),
                value.get("model"),
                _nested(value, "traj_info", "model"),
                _nested(value, "traj_info", "metadata", "model"),
            )
            if (found := _text(candidate))
        ),
        None,
    )
    usage = _nested(value, "traj_info", "metadata", "token_usage") or _nested(value, "traj_info", "metadata", "usage")
    return task_id, model, normalize_usage(usage)


def _attack_outcome(path: Path) -> bool | None:
    for name in ("judge-verdict.json", ".m4-verdict.json"):
        value = _json(path / name).get("attack_success")
        if isinstance(value, bool):
            return value
    return None


def _attempt_outcomes(path: Path, result: dict[str, Any]) -> tuple[int, bool | None, bool | None]:
    attempts: dict[int, bool | None] = {}
    attempts_root = path / "attempts"
    if attempts_root.is_dir():
        for candidate in attempts_root.glob("attempt-*"):
            if not candidate.is_dir():
                continue
            try:
                index = int(candidate.name.removeprefix("attempt-"))
            except ValueError:
                continue
            attempts[index] = _attack_outcome(candidate)
    if not attempts:
        submissions = _number(result.get("submissions")) or 0
        latest = result.get("attack_success")
        if submissions == 1 and isinstance(latest, bool):
            attempts[1] = latest
        elif submissions >= 2 and isinstance(latest, bool):
            attempts[submissions] = latest
    return len(attempts), attempts.get(1), attempts.get(2)


def _dataset_path(merged: dict[str, Any], domain: Any, threat_model: Any) -> str | None:
    """Use DTAP's dataset directory convention as the human-facing name."""
    task_dir = _text(merged.get("task_dir"))
    if task_dir:
        parts = Path(task_dir).parts
        dataset_indexes = [index for index, part in enumerate(parts) if part == "dataset"]
        if dataset_indexes:
            relative = parts[dataset_indexes[-1] + 1 :]
            if relative:
                return "/".join(relative)

    components = (
        _text(domain),
        "malicious",
        _text(threat_model),
        _text(merged.get("risk_category")),
        _text(merged.get("task_id")),
    )
    return "/".join(components) if all(components) else None


def discover_episode_dirs(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return []
    candidates: set[Path] = set()
    for name in ("episode-manifest.json", "result.json"):
        for hit in root.rglob(name):
            parent = hit.parent.resolve()
            try:
                parent.relative_to(root)
            except ValueError:
                continue
            if (parent / "policy.jsonl").is_file() or (parent / "victim-trajectory.json").is_file():
                candidates.add(parent)
    if root.is_dir() and ((root / "policy.jsonl").is_file() or (root / "victim-trajectory.json").is_file()):
        candidates.add(root)
    return sorted(candidates)


def _source_mtime_ns(path: Path, *extra_paths: Path | None) -> int:
    values = []
    for name in _METADATA_FILES:
        target = path / name
        if target.exists():
            try:
                values.append(target.stat().st_mtime_ns)
            except OSError:
                pass
    for target in extra_paths:
        if target is None:
            continue
        try:
            values.append(target.stat().st_mtime_ns)
        except OSError:
            pass
    for pattern in (
        "attempts/attempt-*/judge-verdict.json",
        "attempts/attempt-*/.m4-verdict.json",
    ):
        for target in path.glob(pattern):
            try:
                values.append(target.stat().st_mtime_ns)
            except OSError:
                pass
    return max(values, default=0)


def extract_episode_metadata(
    path: str | Path,
    root: str | Path,
    *,
    include_event_counts: bool = True,
) -> dict[str, Any]:
    path = Path(path).resolve()
    root = Path(root).resolve()
    relative = path.relative_to(root)
    result = _json(path / "result.json")
    manifest = _json(path / "episode-manifest.json")
    merged = {**manifest, **result}
    run_summary, run_summary_path = _run_summary(path, root)
    victim_task_id, victim_trace_model, victim_usage = _victim_trace_metadata(path / "victim-trajectory.json")
    attempt_count, h1_attack_success, h2_attack_success = _attempt_outcomes(path, result)

    rel_parts = relative.parts
    domain = merged.get("domain")
    threat = merged.get("threat_model")
    if not domain and len(rel_parts) >= 2:
        domain = rel_parts[-2] if rel_parts[-1] in {"direct", "indirect"} else rel_parts[0]
    if not threat:
        threat = next((part for part in reversed(rel_parts) if part in {"direct", "indirect"}), None)

    episode_id = str(merged.get("episode_id") or "").strip()
    if not episode_id:
        episode_id = f"{domain or 'episode'}:{threat or 'unknown'}:{path.name}"

    # A root can be either one run (domain/threat) or a collection of runs
    # (run/domain/threat). Keep the single-run behavior while making the Run
    # facet useful for collections.
    run_name = root.name
    if len(rel_parts) >= 3:
        run_name = rel_parts[0]

    task_id = _original_task_id(path) or victim_task_id or _text(merged.get("task_id"))
    policy_model = (
        _text(merged.get("policy_model"))
        or _text(run_summary.get("policy_model"))
        or _policy_trace_model(path / "policy.jsonl")
    )
    victim_model = _text(merged.get("victim_model")) or _text(run_summary.get("victim_model")) or victim_trace_model
    dataset_path = _dataset_path(merged, domain, threat)
    policy_events, policy_usage = _policy_metrics(path / "policy.jsonl") if include_event_counts else (None, None)

    return {
        "episode_id": episode_id,
        "run_name": run_name,
        "task_id": task_id,
        "dataset_path": dataset_path,
        "policy_model": policy_model,
        "victim_model": victim_model,
        "policy_usage": policy_usage,
        "victim_usage": victim_usage,
        "attempt_count": attempt_count,
        "h1_attack_success": h1_attack_success,
        "h2_attack_success": h2_attack_success,
        "domain": str(domain) if domain is not None else None,
        "threat_model": str(threat) if threat is not None else None,
        "status": merged.get("status"),
        "episode_status": merged.get("episode_status"),
        "risk_category": merged.get("risk_category"),
        "attack_success": merged.get("attack_success"),
        "evaluation_completed": merged.get("evaluation_completed"),
        "placement_applicable": merged.get("placement_applicable"),
        "placement_covered": merged.get("placement_covered"),
        "placement_actions": merged.get("placement_actions"),
        "placements_verified": merged.get("placements_verified"),
        "environment_steps": merged.get("environment_steps"),
        "policy_events": policy_events,
        "victim_events": _victim_count(path / "victim-trajectory.json") if include_event_counts else None,
        "artifact_path": str(path),
        "source_mtime_ns": _source_mtime_ns(path, run_summary_path),
    }


def index_root(root: str | Path, db: TrajectoryDB) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    scanned = 0
    updated = 0
    for episode_dir in discover_episode_dirs(root):
        scanned += 1
        item = extract_episode_metadata(episode_dir, root, include_event_counts=False)
        existing = db.get_episode(item["episode_id"])
        if (
            existing
            and existing.get("source_mtime_ns") == item["source_mtime_ns"]
            and existing.get("artifact_path") == item["artifact_path"]
            and all(
                existing.get(key) == item.get(key)
                for key in (
                    "task_id",
                    "dataset_path",
                    "policy_model",
                    "victim_model",
                    "attempt_count",
                    "h1_attack_success",
                    "h2_attack_success",
                )
            )
        ):
            continue
        item["policy_events"], item["policy_usage"] = _policy_metrics(episode_dir / "policy.jsonl")
        item["victim_events"] = _victim_count(episode_dir / "victim-trajectory.json")
        db.upsert_episode(item)
        updated += 1
    return {"root": str(root), "scanned": scanned, "updated": updated}
