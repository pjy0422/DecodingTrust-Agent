"""Strict, credential-free configuration for reproducible policy-eval matrices."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "dtap.policy-eval/experiment-v1"
# Keep these two values local: the trajectory viewer intentionally loads this
# validator directly from its file path, outside package import context.
HARNESS_PROTOCOL_V1 = "v1"
HARNESS_PROTOCOLS = (HARNESS_PROTOCOL_V1, "lazy-schema-v2")
POLICY_ENGINE_CLAUDE = "claude-code"
POLICY_ENGINE_DT_ARMS = "dt-arms-upstream"
POLICY_ENGINES = (POLICY_ENGINE_CLAUDE, POLICY_ENGINE_DT_ARMS)
DT_ARMS_VICTIM_ARCHITECTURES = (
    "openaisdk",
    "pocketflow",
    "langchain",
    "claudesdk",
    "googleadk",
    "openclaw",
)


class ExperimentConfigError(ValueError):
    """An experiment file is malformed or contains an unsupported field."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentConfigError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ExperimentConfigError(f"{path} keys must be strings")
    return value


def _section(root: Mapping[str, Any], name: str, allowed: set[str]) -> Mapping[str, Any]:
    value = _mapping(root.get(name, {}), name)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ExperimentConfigError(f"unknown {name} field(s): {', '.join(unknown)}")
    return value


def _required(section: Mapping[str, Any], name: str, path: str) -> Any:
    if name not in section:
        raise ExperimentConfigError(f"missing required field: {path}.{name}")
    return section[name]


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExperimentConfigError(f"{path} must be a positive integer")
    return value


def _positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ExperimentConfigError(f"{path} must be positive")
    return float(value)


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigError(f"{path} must be a non-empty string")
    return value


def _string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ExperimentConfigError(f"{path} must be a non-empty list")
    result = [_string(item, f"{path}[]") for item in value]
    if len(set(result)) != len(result):
        raise ExperimentConfigError(f"{path} must not contain duplicates")
    return result


def _optional_string_list(value: Any, path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ExperimentConfigError(f"{path} must be a list")
    result = [_string(item, f"{path}[]") for item in value]
    if len(set(result)) != len(result):
        raise ExperimentConfigError(f"{path} must not contain duplicates")
    return result


def _dataset_task_list(value: Any, path: str) -> list[str]:
    result = _string_list(value, path)
    for item in result:
        parts = item.split("/")
        if (
            len(parts) != 5
            or parts[1] != "malicious"
            or any(not part or part in {".", ".."} for part in parts)
            or "\\" in item
        ):
            raise ExperimentConfigError(
                f"{path} entries must match "
                "<domain>/malicious/<threat_model>/<risk_category>/<task_id>"
            )
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ExperimentConfigError(f"{path} must be true or false")
    return value


def _relative_path(value: Any, path: str, config_dir: Path) -> Path:
    raw = Path(_string(value, path)).expanduser()
    return (config_dir / raw).resolve() if not raw.is_absolute() else raw.resolve()


def _turn_budget(value: Any, path: str) -> int | None:
    if value == "auto":
        return None
    return _positive_int(value, path)


def _harness_protocol(value: Any) -> str:
    raw = _string(value, "policy.harness_protocol")
    if raw not in HARNESS_PROTOCOLS:
        raise ExperimentConfigError(f"unsupported policy harness protocol: {raw!r}")
    return raw


def _policy_engine(value: Any) -> str:
    raw = _string(value, "policy.engine")
    if raw not in POLICY_ENGINES:
        raise ExperimentConfigError(f"unsupported policy engine: {raw!r}")
    return raw


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load one matrix configuration into argparse-compatible defaults.

    The schema is intentionally closed. In particular, provider URLs, API keys,
    tokens, and arbitrary environment variables cannot be stored here; runtime
    credentials continue to come only from the process environment.
    """

    config_path = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExperimentConfigError(f"cannot read experiment config: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(f"invalid YAML: {exc}") from exc
    root = _mapping(raw, "config")
    allowed_root = {
        "schema",
        "paths",
        "selection",
        "models",
        "policy",
        "victim",
        "budgets",
        "placement",
        "feedback",
        "execution",
        "dt_arms",
    }
    unknown = sorted(set(root) - allowed_root)
    if unknown:
        raise ExperimentConfigError(f"unknown top-level field(s): {', '.join(unknown)}")
    if root.get("schema") != SCHEMA:
        raise ExperimentConfigError(f"schema must be {SCHEMA!r}")

    paths = _section(root, "paths", {"dtap_root", "artifacts_root", "python"})
    selection = _section(root, "selection", {"domains", "threat_models", "profile", "tasks"})
    models = _section(root, "models", {"policy", "victim"})
    policy = _section(
        root,
        "policy",
        {
            "engine",
            "harness_protocol",
            "planning_strategy",
            "max_turns",
            "improvement_wishes",
            "dying_message",
        },
    )
    victim = _section(root, "victim", {"harness", "max_turns"})
    budgets = _section(
        root,
        "budgets",
        {"h_victim_executions", "q_submit_calls", "max_placement_actions"},
    )
    placement = _section(root, "placement", {"enabled"})
    feedback = _section(
        root,
        "feedback",
        {
            "mode",
            "digestor_model",
            "digestor_max_tokens",
            "digestor_timeout_seconds",
            "reasoning_summary",
        },
    )
    execution = _section(
        root,
        "execution",
        {"max_parallel", "timeout_seconds", "port_range_start", "port_range_stride", "resume"},
    )
    dt_arms = _section(
        root,
        "dt_arms",
        {
            "judge_model",
            "max_iterations",
            "use_memory",
            "update_memory",
            "memory_save_mode",
            "auto_aggregate_memory",
            "allow_quit",
            "multi_turn",
            "max_turns_per_session",
            "injection_override",
            "allowed_skill_names",
            "allowed_skill_types",
        },
    )

    config_dir = config_path.parent
    h = _positive_int(_required(budgets, "h_victim_executions", "budgets"), "budgets.h_victim_executions")
    q = _positive_int(_required(budgets, "q_submit_calls", "budgets"), "budgets.q_submit_calls")
    if q < h:
        raise ExperimentConfigError("budgets.q_submit_calls must be >= budgets.h_victim_executions")
    policy_engine = _policy_engine(policy.get("engine", POLICY_ENGINE_CLAUDE))
    harness_protocol = _harness_protocol(
        policy.get("harness_protocol", HARNESS_PROTOCOL_V1)
    )
    if (
        policy_engine == POLICY_ENGINE_DT_ARMS
        and harness_protocol != HARNESS_PROTOCOL_V1
    ):
        raise ExperimentConfigError(
            "policy.harness_protocol applies only to policy.engine='claude-code'; "
            "DT Arms uses its pinned native red-team loop"
        )
    victim_agent_type = _string(victim.get("harness", "openclaw"), "victim.harness")
    if (
        policy_engine == POLICY_ENGINE_DT_ARMS
        and victim_agent_type not in DT_ARMS_VICTIM_ARCHITECTURES
    ):
        raise ExperimentConfigError(
            f"DT Arms does not support victim harness: {victim_agent_type!r}"
        )
    memory_save_mode = _string(dt_arms.get("memory_save_mode", "success"), "dt_arms.memory_save_mode")
    if memory_save_mode not in {"all", "success"}:
        raise ExperimentConfigError("dt_arms.memory_save_mode must be 'all' or 'success'")
    injection_override = _optional_string_list(
        dt_arms.get("injection_override"), "dt_arms.injection_override"
    )
    unknown_injections = sorted(
        set(injection_override) - {"prompt", "tool", "environment", "skill", "a2a"}
    )
    if unknown_injections:
        raise ExperimentConfigError(
            f"unsupported dt_arms injection type: {unknown_injections[0]!r}"
        )
    multi_turn = _boolean(dt_arms.get("multi_turn", False), "dt_arms.multi_turn")
    configured_threats = _string_list(
        _required(selection, "threat_models", "selection"), "selection.threat_models"
    )
    if multi_turn and any(value != "direct" for value in configured_threats):
        raise ExperimentConfigError("dt_arms.multi_turn requires direct-only task selection")
    defaults: dict[str, Any] = {
        "config": config_path,
        "dtap_root": _relative_path(
            _required(paths, "dtap_root", "paths"), "paths.dtap_root", config_dir
        ),
        "artifacts_root": _relative_path(
            _required(paths, "artifacts_root", "paths"), "paths.artifacts_root", config_dir
        ),
        "python": _string(paths.get("python", "python"), "paths.python"),
        "domains": _string_list(_required(selection, "domains", "selection"), "selection.domains"),
        "threat_models": configured_threats,
        "selection_profile": _string(selection.get("profile", "release-v1"), "selection.profile"),
        "selected_tasks": (
            _dataset_task_list(selection["tasks"], "selection.tasks")
            if "tasks" in selection
            else []
        ),
        "policy_model": _string(_required(models, "policy", "models"), "models.policy"),
        "victim_model": _string(_required(models, "victim", "models"), "models.victim"),
        "policy_engine": policy_engine,
        "planning_strategy": _string(
            policy.get("planning_strategy", "current"), "policy.planning_strategy"
        ),
        "harness_protocol": harness_protocol,
        "policy_max_turns": _turn_budget(policy.get("max_turns", "auto"), "policy.max_turns"),
        "improvement_wishes": _boolean(
            policy.get("improvement_wishes", False), "policy.improvement_wishes"
        ),
        "dying_message": _boolean(policy.get("dying_message", False), "policy.dying_message"),
        "victim_agent_type": victim_agent_type,
        "victim_max_turns": _positive_int(victim.get("max_turns", 80), "victim.max_turns"),
        "max_submissions": h,
        "max_submit_calls": q,
        "max_placement_actions": _positive_int(
            _required(budgets, "max_placement_actions", "budgets"),
            "budgets.max_placement_actions",
        ),
        "placement_enabled": _boolean(placement.get("enabled", True), "placement.enabled"),
        "feedback_mode": _string(feedback.get("mode", "disabled"), "feedback.mode"),
        "digestor_model": _string(feedback.get("digestor_model", "glm-5.2"), "feedback.digestor_model"),
        "digestor_max_tokens": _positive_int(
            feedback.get("digestor_max_tokens", 50_000), "feedback.digestor_max_tokens"
        ),
        "digestor_timeout": _positive_number(
            feedback.get("digestor_timeout_seconds", 30.0),
            "feedback.digestor_timeout_seconds",
        ),
        "reasoning_summary": _boolean(
            feedback.get("reasoning_summary", False), "feedback.reasoning_summary"
        ),
        "max_parallel": _positive_int(execution.get("max_parallel", 2), "execution.max_parallel"),
        "timeout": _positive_int(execution.get("timeout_seconds", 1800), "execution.timeout_seconds"),
        "port_range_start": _positive_int(
            execution.get("port_range_start", 20_000), "execution.port_range_start"
        ),
        "port_range_stride": _positive_int(
            execution.get("port_range_stride", 1_024), "execution.port_range_stride"
        ),
        "resume": _boolean(execution.get("resume", False), "execution.resume"),
        "dt_arms_judge_model": _string(
            dt_arms.get("judge_model", _required(models, "victim", "models")),
            "dt_arms.judge_model",
        ),
        "dt_arms_max_iterations": _positive_int(
            dt_arms.get("max_iterations", 10), "dt_arms.max_iterations"
        ),
        "dt_arms_use_memory": _boolean(
            dt_arms.get("use_memory", False), "dt_arms.use_memory"
        ),
        "dt_arms_update_memory": _boolean(
            dt_arms.get("update_memory", False), "dt_arms.update_memory"
        ),
        "dt_arms_memory_save_mode": memory_save_mode,
        "dt_arms_auto_aggregate_memory": _boolean(
            dt_arms.get("auto_aggregate_memory", False), "dt_arms.auto_aggregate_memory"
        ),
        "dt_arms_allow_quit": _boolean(
            dt_arms.get("allow_quit", True), "dt_arms.allow_quit"
        ),
        "dt_arms_multi_turn": multi_turn,
        "dt_arms_max_turns_per_session": _positive_int(
            dt_arms.get("max_turns_per_session", 5), "dt_arms.max_turns_per_session"
        ),
        "dt_arms_injection_override": injection_override,
        "dt_arms_allowed_skill_names": _optional_string_list(
            dt_arms.get("allowed_skill_names"), "dt_arms.allowed_skill_names"
        ),
        "dt_arms_allowed_skill_types": _optional_string_list(
            dt_arms.get("allowed_skill_types"), "dt_arms.allowed_skill_types"
        ),
    }
    return defaults


def resolved_experiment_document(args: Any) -> dict[str, Any]:
    """Return the canonical, credential-free configuration actually executed."""

    return {
        "schema": SCHEMA,
        "paths": {
            "dtap_root": str(args.dtap_root),
            "artifacts_root": str(args.artifacts_root),
            "python": args.python,
        },
        "selection": {
            "domains": list(args.domains),
            "threat_models": list(args.threat_models),
            "profile": args.selection_profile,
            **(
                {"tasks": list(args.selected_tasks)}
                if getattr(args, "selected_tasks", ())
                else {}
            ),
        },
        "models": {"policy": args.policy_model, "victim": args.victim_model},
        "policy": {
            "engine": args.policy_engine,
            "harness_protocol": args.harness_protocol,
            "planning_strategy": args.planning_strategy,
            "max_turns": args.policy_max_turns,
            "improvement_wishes": args.improvement_wishes,
            "dying_message": args.dying_message,
        },
        "victim": {"harness": args.victim_agent_type, "max_turns": args.victim_max_turns},
        "budgets": {
            "h_victim_executions": args.max_submissions,
            "q_submit_calls": args.max_submit_calls,
            "max_placement_actions": args.max_placement_actions,
        },
        "placement": {"enabled": args.placement_enabled},
        "feedback": {
            "mode": args.feedback_mode,
            "digestor_model": args.digestor_model,
            "digestor_max_tokens": args.digestor_max_tokens,
            "digestor_timeout_seconds": args.digestor_timeout,
            "reasoning_summary": args.reasoning_summary,
        },
        "execution": {
            "max_parallel": args.max_parallel,
            "timeout_seconds": args.timeout,
            "port_range_start": args.port_range_start,
            "port_range_stride": args.port_range_stride,
            "resume": args.resume,
        },
        "dt_arms": {
            "judge_model": args.dt_arms_judge_model,
            "max_iterations": args.dt_arms_max_iterations,
            "use_memory": args.dt_arms_use_memory,
            "update_memory": args.dt_arms_update_memory,
            "memory_save_mode": args.dt_arms_memory_save_mode,
            "auto_aggregate_memory": args.dt_arms_auto_aggregate_memory,
            "allow_quit": args.dt_arms_allow_quit,
            "multi_turn": args.dt_arms_multi_turn,
            "max_turns_per_session": args.dt_arms_max_turns_per_session,
            "injection_override": list(args.dt_arms_injection_override),
            "allowed_skill_names": list(args.dt_arms_allowed_skill_names),
            "allowed_skill_types": list(args.dt_arms_allowed_skill_types),
        },
    }


def write_resolved_experiment(args: Any) -> Path:
    destination = args.artifacts_root / "experiment-config.resolved.yaml"
    temporary = args.artifacts_root / ".experiment-config.resolved.yaml.tmp"
    temporary.write_text(
        yaml.safe_dump(resolved_experiment_document(args), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
