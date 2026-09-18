"""Load one viewer-ready DTAP episode bundle using the existing parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .indexer import normalize_usage
from .parser import build_timeline, find_policy_trace, find_victim_mcp_events, find_victim_trace


def _first(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for name in names:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _judge_source(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return "deterministic"
    if metadata.get("judge_tool") == "llm_judge" or metadata.get("verdict_source") == "llm_judge":
        return "llm_as_judge"
    if any(
        metadata.get(key) is not None for key in ("llm_judge", "llm_model", "gpt_model", "gpt_score", "gpt_rationale")
    ):
        return "llm_as_judge"
    return "deterministic"


def _judge_component(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    success_key = f"{name}_success"
    message_key = f"{name}_message"
    metadata_key = f"{name}_metadata"
    if not any(key in result for key in (success_key, message_key, metadata_key)):
        return None
    metadata = result.get(metadata_key)
    return {
        "name": name,
        "success": result.get(success_key),
        "message": result.get(message_key),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "source": _judge_source(metadata),
    }


def load_judge_results(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    result = _read_json(_first(root, ("judge-result.json", "judge_result.json")))
    verdict = _read_json(_first(root, ("judge-verdict.json", ".m4-verdict.json")))
    components = [
        component for name in ("task", "attack") if (component := _judge_component(result, name)) is not None
    ]
    return {
        "available": bool(result or verdict),
        "components": components,
        "error": result.get("error") if result else None,
        "reward_firewall": verdict,
        "raw": result,
    }


def _attempt_directories(root: Path) -> list[tuple[int, Path]]:
    attempts_root = root / "attempts"
    found: list[tuple[int, Path]] = []
    if not attempts_root.is_dir():
        return found
    for candidate in attempts_root.glob("attempt-*"):
        if not candidate.is_dir():
            continue
        try:
            index = int(candidate.name.removeprefix("attempt-"))
        except ValueError:
            continue
        found.append((index, candidate))
    return sorted(found)


def load_episode_bundle(path: str | Path, *, attempt_index: int | None = None) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    attempt_directories = _attempt_directories(root)
    if attempt_index is not None and not any(index == attempt_index for index, _ in attempt_directories):
        raise ValueError(f"attempt {attempt_index} not found")
    selected_index, selected_root = (
        next(item for item in attempt_directories if item[0] == attempt_index)
        if attempt_index is not None
        else (attempt_directories[-1] if attempt_directories else (None, root))
    )
    victim = find_victim_trace(selected_root)
    victim_payload = _read_json(victim)
    policy = find_policy_trace(root)
    mcp = find_victim_mcp_events(selected_root)
    original = _first(root, ("original-config.yaml", "original_config.yaml"))
    submitted = _first(
        selected_root,
        ("submitted-config.yaml", "submitted_config.yaml", "attack.yaml"),
    )
    prompt = _first(root, ("policy-prompt.txt", "policy_prompt.txt"))
    manifest = _read_json(_first(root, ("episode-manifest.json",)))
    meta = {"episode_id": str(manifest.get("episode_id"))} if manifest.get("episode_id") else {}
    data = build_timeline(
        victim,
        meta=meta,
        policy_trace_path=policy,
        policy_prompt_path=prompt,
        original_yaml_path=original,
        submitted_yaml_path=submitted,
        victim_mcp_events_path=mcp,
    )
    result = _read_json(_first(root, ("result.json",)))
    judges = load_judge_results(selected_root)
    evaluation = {
        key: result[key]
        for key in (
            "status",
            "evaluation_completed",
            "failure_class",
            "attack_success",
            "episode_status",
            "placement_applicable",
            "placement_covered",
            "placement_actions",
            "placements_verified",
        )
        if key in result
    }
    if judges["raw"]:
        evaluation["judge"] = judges["raw"]
    elif judges["reward_firewall"]:
        evaluation["judge"] = judges["reward_firewall"]
    if evaluation:
        data["evaluation"] = evaluation
    data["judges"] = judges
    victim_metadata = (
        victim_payload.get("traj_info", {}).get("metadata", {})
        if isinstance(victim_payload.get("traj_info"), dict)
        and isinstance(victim_payload.get("traj_info", {}).get("metadata"), dict)
        else {}
    )
    victim_usage = victim_metadata.get("token_usage") or victim_metadata.get("usage")
    data["victim_usage"] = normalize_usage(victim_usage)
    data["attempt_index"] = selected_index
    data["attempts"] = [
        {
            "index": index,
            "attack_success": _read_json(_first(directory, ("judge-verdict.json", ".m4-verdict.json"))).get(
                "attack_success"
            ),
        }
        for index, directory in attempt_directories
    ]
    return data
