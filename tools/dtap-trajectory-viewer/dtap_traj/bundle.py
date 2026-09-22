"""Load one viewer-ready DTAP episode bundle using the existing parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

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


def _read_text(path: Path | None, *, max_bytes: int = 32 * 1024 * 1024) -> str | None:
    if path is None:
        return None
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _system_prompt(path: Path | None) -> str | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        config = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    if not isinstance(config, dict):
        return None
    agent = config.get("Agent")
    if not isinstance(agent, dict):
        agent = config.get("agent")
    if not isinstance(agent, dict):
        return None
    value = agent.get("system_prompt")
    return value if isinstance(value, str) and value else None


def _prompt_snapshot_records(
    path: Path,
    *,
    source: str,
    attempt_index: int | None,
) -> list[dict[str, Any]]:
    raw = _read_text(path)
    if raw is None:
        return []
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if (
            not isinstance(value, dict)
            or value.get("schema") != "dtap-policy-eval-prompt-snapshot"
            or value.get("schema_version") != 1
            or value.get("role") not in {"system", "user"}
            or not isinstance(value.get("prompt"), str)
        ):
            continue
        record_attempt = value.get("attempt_index")
        if (
            attempt_index is not None
            and isinstance(record_attempt, int)
            and not isinstance(record_attempt, bool)
            and record_attempt != attempt_index
        ):
            continue
        records.append(
            {
                "component": str(value.get("component") or "unknown"),
                "label": str(value.get("label") or value.get("component") or "Prompt"),
                "role": value["role"],
                "prompt": value["prompt"],
                "source": str(value.get("source") or source),
                "exact": True,
                "sequence": value.get("sequence"),
                "attempt_index": record_attempt if isinstance(record_attempt, int) else None,
            }
        )
    return records


def load_prompt_components(
    root: Path,
    *,
    selected_root: Path,
    original: Path | None,
    submitted: Path | None,
    judges: dict[str, Any],
    feedback_evidence: dict[str, Any],
    attempt_index: int | None,
) -> dict[str, Any]:
    """Return only prompts retained by the episode; never reconstruct history from source."""

    components: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    policy_prompt = _read_text(root / "policy-prompt.txt") or _read_text(root / "policy_prompt.txt")
    if policy_prompt is not None:
        components.append(
            {
                "component": "policy",
                "label": "Policy launch prompt",
                "role": "user",
                "prompt": policy_prompt,
                "source": "policy-prompt.txt",
                "exact": True,
            }
        )
        unavailable.append(
            {
                "component": "policy_runtime",
                "label": "Policy runtime system prompt",
                "reason": "Claude Code owns this prompt and does not expose it to the evaluator artifact.",
            }
        )
    else:
        unavailable.append(
            {
                "component": "policy",
                "label": "Policy prompt",
                "reason": "This episode did not retain a policy prompt artifact.",
            }
        )

    effective_prompt = _system_prompt(submitted) or _system_prompt(original)
    if effective_prompt is not None:
        source = "submitted-config.yaml#/Agent/system_prompt" if _system_prompt(submitted) is not None else (
            "original-config.yaml#/Agent/system_prompt"
        )
        components.append(
            {
                "component": "victim",
                "label": "Victim system prompt",
                "role": "system",
                "prompt": effective_prompt,
                "source": source,
                "exact": True,
            }
        )
    else:
        unavailable.append(
            {
                "component": "victim",
                "label": "Victim system prompt",
                "reason": "No Agent.system_prompt was retained in the selected configuration.",
            }
        )

    snapshot_paths = [(root / "prompt-snapshots.jsonl", "prompt-snapshots.jsonl")]
    if selected_root != root:
        snapshot_paths.append(
            (selected_root / "prompt-snapshots.jsonl", "attempts/selected/prompt-snapshots.jsonl")
        )
    for path, source in snapshot_paths:
        if path.is_file():
            components.extend(
                _prompt_snapshot_records(path, source=source, attempt_index=attempt_index)
            )

    snapshotted = {item["component"] for item in components}
    if feedback_evidence and "digestor" not in snapshotted:
        unavailable.append(
            {
                "component": "digestor",
                "label": "Digestor request prompt",
                "reason": "Legacy feedback evidence exists, but this episode predates exact prompt snapshots.",
            }
        )
    for judge in judges.get("components", []):
        if not isinstance(judge, dict):
            continue
        name = str(judge.get("name") or "judge")
        if judge.get("source") == "deterministic":
            unavailable.append(
                {
                    "component": f"{name}_judge",
                    "label": f"{name.title()} judge",
                    "reason": "Deterministic judge; no LLM system prompt is used.",
                }
            )
        elif f"{name}_judge" not in snapshotted:
            unavailable.append(
                {
                    "component": f"{name}_judge",
                    "label": f"{name.title()} judge system prompt",
                    "reason": "The judge result is retained, but its runtime did not retain the exact prompt.",
                }
            )
    return {"available": bool(components), "components": components, "unavailable": unavailable}


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


def _dt_arms_judge_history(root: Path) -> dict[str, Any]:
    retained = _read_json(root / "dt-arms" / "judge-history.json")
    if retained.get("schema") == "dtap-policy-eval-dt-arms-judge-history" and isinstance(
        retained.get("iterations"), list
    ):
        return retained
    trajectory = _read_json(root / "dt-arms" / "trajectory.json")
    steps = trajectory.get("attack_trajectory")
    if not isinstance(steps, list):
        return {}
    config = trajectory.get("attack_config") if isinstance(trajectory.get("attack_config"), dict) else {}
    iterations: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        role = step.get("role")
        state = step.get("state")
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        if role == "verifiable_judge" and isinstance(state, dict):
            attack = state.get("attack")
            task = state.get("task")
            if not isinstance(attack, bool) and not isinstance(task, bool):
                continue
            pending = {
                "iteration": len(iterations) + 1,
                "verifiable": {
                    "step_id": step.get("step_id"),
                    "attack_success": attack if isinstance(attack, bool) else None,
                    "task_success": task if isinstance(task, bool) else None,
                    "metadata": metadata,
                },
                "feedback": None,
            }
            iterations.append(pending)
        elif role == "feedback_judge" and pending is not None and pending["feedback"] is None:
            pending["feedback"] = {
                "step_id": step.get("step_id"),
                "attack_success": state if isinstance(state, bool) else None,
                "metadata": metadata,
            }
    if not iterations:
        return {}
    return {
        "schema": "dtap-policy-eval-dt-arms-judge-history",
        "schema_version": 1,
        "source": "dt-arms-upstream",
        "authoritative_replay": False,
        "judge_model": config.get("judge_model"),
        "iterations": iterations,
    }


def load_judge_results(root: str | Path, *, episode_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    episode_root = Path(episode_root).expanduser().resolve() if episode_root is not None else root
    result = _read_json(_first(root, ("judge-result.json", "judge_result.json")))
    verdict = _read_json(_first(root, ("judge-verdict.json", ".m4-verdict.json")))
    dt_arms_history = _dt_arms_judge_history(episode_root)
    components = [
        component for name in ("task", "attack") if (component := _judge_component(result, name)) is not None
    ]
    return {
        "available": bool(result or verdict or dt_arms_history),
        "components": components,
        "error": result.get("error") if result else None,
        "reward_firewall": verdict,
        "raw": result,
        "dt_arms_history": dt_arms_history,
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
    judges = load_judge_results(selected_root, episode_root=root)
    feedback_evidence = _read_json(_first(selected_root, ("feedback-evidence.json",)))
    data["prompts"] = load_prompt_components(
        root,
        selected_root=selected_root,
        original=original,
        submitted=submitted,
        judges=judges,
        feedback_evidence=feedback_evidence,
        attempt_index=selected_index,
    )
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
    verdict = judges.get("reward_firewall")
    attack_success = verdict.get("attack_success") if isinstance(verdict, dict) else None
    if "attack_success" not in evaluation and isinstance(attack_success, bool):
        evaluation["attack_success"] = attack_success
        evaluation.setdefault("evaluation_completed", True)
    if judges["raw"]:
        evaluation["judge"] = judges["raw"]
    elif judges["reward_firewall"]:
        evaluation["judge"] = judges["reward_firewall"]
    if evaluation:
        data["evaluation"] = evaluation
    data["judges"] = judges
    if feedback_evidence:
        data["feedback_evidence"] = feedback_evidence
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
