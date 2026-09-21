"""Run profile-selected or exact dataset policy E2E tasks in parallel."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dt_arena.policy_eval.benchmark_manifest import (
    ALL_CASES,
    ALL_DOMAINS,
    BENCHMARK_MANIFEST,
    DOMAINS,
    EXCLUDED_PLATFORM_DOMAINS,
    MANIFEST_SHA256,
    SELECTION_PROFILES,
    THREAT_MODELS,
    matrix_cases,
)
from dt_arena.policy_eval.experiment_config import (
    ExperimentConfigError,
    load_experiment_config,
    write_resolved_experiment,
)
from dt_arena.policy_eval.protocol import HARNESS_PROTOCOLS
from dt_arena.policy_eval.security_policy import policy_max_turn_budget


MAX_PARALLEL = 16


@dataclass(frozen=True)
class MatrixTask:
    domain: str
    threat_model: str
    risk_category: str
    task_id: str
    task_dir: Path
    benchmark_index: int | None
    explicit: bool = False


def _failure_class(result: dict[str, Any]) -> str | None:
    if result.get("status") == "passed":
        return None
    if result.get("failure_class"):
        return str(result["failure_class"])
    tail = str(result.get("error_tail") or "").lower()
    if "unsupported_placement" in tail:
        return "unsupported_placement"
    if "placement_mismatch" in tail or "placement" in tail and "read-back" in tail:
        return "placement"
    if "judge" in tail:
        return "judge"
    if "victim" in tail or "openclaw" in tail and "policy" not in tail:
        return "victim"
    if "invalid_submission" in tail or "validation" in tail:
        return "validation"
    if "policy" in tail or "glm did not" in tail:
        return "policy"
    return "infrastructure"


def _summary_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("evaluation_completed")]
    applicable = [item for item in completed if item.get("placement_applicable")]
    tool_attempts = Counter(str(tool) for item in completed for tool in (item.get("environment_tools") or ()))
    tool_verified = Counter()
    for item in applicable:
        if item.get("placement_covered"):
            tool_verified.update(map(str, item.get("environment_tools") or ()))

    def grouped(field: str) -> dict[str, dict[str, int]]:
        keys = sorted({str(item.get(field)) for item in completed if item.get(field)})
        return {
            key: {
                "evaluations": sum(str(item.get(field)) == key for item in completed),
                "applicable": sum(
                    str(item.get(field)) == key and item.get("placement_applicable") is True for item in completed
                ),
                "covered": sum(
                    str(item.get(field)) == key and item.get("placement_covered") is True for item in completed
                ),
            }
            for key in keys
        }

    server_attempts = Counter()
    server_verified = Counter()
    for item in completed:
        servers = {str(tool).split(":", 1)[0] for tool in item.get("environment_tools") or ()}
        server_attempts.update(servers)
        if item.get("placement_covered"):
            server_verified.update(servers)
    return {
        "evaluation_completed": len(completed),
        "attack_successes": sum(item.get("attack_success") is True for item in completed),
        "action_applied": sum(item.get("action_applied") is True for item in completed),
        "placement_applicable": len(applicable),
        "placement_covered": sum(item.get("placement_covered") is True for item in applicable),
        "placement_actions": sum(int(item.get("placement_actions") or 0) for item in completed),
        "placements_verified": sum(int(item.get("placements_verified") or 0) for item in completed),
        "placement_by_tool": {
            tool: {"attempted": count, "verified": tool_verified[tool]}
            for tool, count in sorted(tool_attempts.items())
        },
        "placement_by_injection_mcp": {
            server: {"attempted": count, "verified": server_verified[server]}
            for server, count in sorted(server_attempts.items())
        },
        "placement_by_domain": grouped("domain"),
        "placement_by_threat_model": grouped("threat_model"),
        "failures_by_class": {
            name: sum(_failure_class(item) == name for item in results)
            for name in (
                "policy",
                "validation",
                "unsupported_placement",
                "placement",
                "victim",
                "judge",
                "infrastructure",
            )
        },
    }


def _first_record(path: Path) -> dict[str, Any]:
    return _selected_record(path, "release-v1")[0]


def _selected_record(
    path: Path,
    selection_profile: str,
) -> tuple[dict[str, Any], int]:
    index = SELECTION_PROFILES[selection_profile]
    selected: dict[str, Any] | None = None
    record_index = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if record_index == index:
                value = json.loads(line)
                if isinstance(value, dict):
                    selected = value
                break
            record_index += 1
    if selected is None:
        raise ValueError(f"benchmark profile {selection_profile!r} index {index} is unavailable: {path}")
    return selected, index


def _task_dir(dtap_root: Path, record: dict[str, Any]) -> Path:
    return dtap_root.joinpath(
        "dataset",
        str(record["domain"]),
        "malicious",
        str(record["threat_model"]),
        str(record["risk_category"]),
        str(record["task_id"]),
    )


def _explicit_task(dtap_root: Path, relative_path: str) -> MatrixTask:
    parts = relative_path.split("/")
    if (
        len(parts) != 5
        or parts[1] != "malicious"
        or any(not part or part in {".", ".."} for part in parts)
        or "\\" in relative_path
    ):
        raise ValueError(
            "explicit tasks must match "
            "<domain>/malicious/<threat_model>/<risk_category>/<task_id>"
        )
    domain, _, threat_model, risk_category, task_id = parts
    dataset_root = (dtap_root / "dataset").resolve()
    task_dir = dataset_root.joinpath(*parts).resolve()
    try:
        task_dir.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"explicit task escapes dataset root: {relative_path}") from exc
    if not (task_dir / "config.yaml").is_file():
        raise ValueError(f"explicit task has no config.yaml: {relative_path}")
    return MatrixTask(
        domain=domain,
        threat_model=threat_model,
        risk_category=risk_category,
        task_id=task_id,
        task_dir=task_dir,
        benchmark_index=None,
        explicit=True,
    )


def _matrix_tasks(args: argparse.Namespace) -> list[MatrixTask]:
    if args.selected_tasks:
        return [_explicit_task(args.dtap_root, path) for path in args.selected_tasks]
    tasks = []
    for domain, threat_model in matrix_cases(args.domains, args.threat_models):
        record, benchmark_index = _selected_record(
            args.dtap_root / "benchmark" / domain / f"{threat_model}.jsonl",
            args.selection_profile,
        )
        tasks.append(
            MatrixTask(
                domain=domain,
                threat_model=threat_model,
                risk_category=str(record["risk_category"]),
                task_id=str(record["task_id"]),
                task_dir=_task_dir(args.dtap_root, record),
                benchmark_index=benchmark_index,
            )
        )
    return tasks


def _case_dir(root: Path, task: MatrixTask) -> Path:
    if task.explicit:
        return root / task.domain / task.threat_model / task.risk_category / task.task_id
    return root / task.domain / task.threat_model


def _passed_payload(stdout: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    matches: list[tuple[int, dict[str, Any]]] = []
    for index, character in enumerate(stdout):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("status") == "passed":
            matches.append((end, value))
    return max(matches, key=lambda item: item[0])[1] if matches else None


def _stored_results(root: Path, tasks: list[MatrixTask]) -> list[dict[str, Any]]:
    if not tasks or not all(task.explicit for task in tasks):
        # Preserve the profile-matrix resume summary: it includes every valid
        # domain/threat result already stored under the run root, even when the
        # current invocation selected only a subset.
        results: list[dict[str, Any]] = []
        for domain, threat_model in ALL_CASES:
            path = root / domain / threat_model / "result.json"
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                value = {"result_error": type(exc).__name__}
            if (
                not isinstance(value, dict)
                or value.get("domain") != domain
                or value.get("threat_model") != threat_model
                or value.get("status") not in ("passed", "failed")
            ):
                value = {
                    "domain": domain,
                    "threat_model": threat_model,
                    "status": "failed",
                    "result_error": "invalid stored result",
                }
            results.append(value)
        return results

    results: list[dict[str, Any]] = []
    for task in tasks:
        path = _case_dir(root, task) / "result.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            value = {"result_error": type(exc).__name__}
        if (
            not isinstance(value, dict)
            or value.get("domain") != task.domain
            or value.get("threat_model") != task.threat_model
            or value.get("task_id") != task.task_id
            or value.get("status") not in ("passed", "failed")
        ):
            value = {
                "domain": task.domain,
                "threat_model": task.threat_model,
                "risk_category": task.risk_category,
                "task_id": task.task_id,
                "status": "failed",
                "result_error": "invalid stored result",
            }
        results.append(value)
    return results


def _failure_count(results: list[dict[str, Any]]) -> int:
    """Platform names never exempt failures from the test verdict."""
    return sum(item.get("status") != "passed" for item in results)


def _matching_resume_result(
    path: Path,
    *,
    selection_profile: str,
    benchmark_index: int | None,
    task_id: str,
    risk_category: str,
) -> dict[str, Any] | None:
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(previous, dict):
        return None
    previous_profile = previous.get("selection_profile", "release-v1")
    previous_index = previous.get("benchmark_index", 0)
    if (
        previous.get("status") == "passed"
        and previous.get("task_id") == task_id
        and previous.get("risk_category") == risk_category
        and previous_profile == selection_profile
        and previous_index == benchmark_index
    ):
        return previous
    return None


async def _run_case(
    args: argparse.Namespace,
    *,
    task: MatrixTask,
    slot: int,
) -> dict[str, Any]:
    case_dir = _case_dir(args.artifacts_root, task)
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    if args.resume and result_path.exists():
        previous = _matching_resume_result(
            result_path,
            selection_profile=("explicit" if task.explicit else args.selection_profile),
            benchmark_index=task.benchmark_index,
            task_id=task.task_id,
            risk_category=task.risk_category,
        )
        if previous is not None:
            return previous

    start = args.port_range_start + slot * args.port_range_stride
    end = start + 511
    command = [
        args.python,
        "-m",
        "dt_arena.policy_eval.scripts.run_policy_e2e",
        "--task-dir",
        str(task.task_dir),
        "--dtap-root",
        str(args.dtap_root),
        "--python",
        args.python,
        "--policy-model",
        args.policy_model,
        "--planning-strategy",
        args.planning_strategy,
        "--harness-protocol",
        args.harness_protocol,
        "--victim-model",
        args.victim_model,
        "--victim-agent-type",
        args.victim_agent_type,
        "--max-submissions",
        str(args.max_submissions),
        "--max-submit-calls",
        str(args.max_submit_calls),
        "--max-placement-actions",
        str(args.max_placement_actions),
        "--policy-max-turns",
        str(args.policy_max_turns),
        "--victim-max-turns",
        str(args.victim_max_turns),
        "--timeout",
        str(args.timeout),
        "--feedback-mode",
        args.feedback_mode,
        "--digestor-model",
        args.digestor_model,
        "--digestor-max-tokens",
        str(args.digestor_max_tokens),
        "--digestor-timeout",
        str(args.digestor_timeout),
        "--port-range-start",
        str(start),
        "--artifacts-dir",
        str(case_dir),
    ]
    if args.placement_enabled:
        command.append("--placement")
    if args.reasoning_summary:
        command.append("--reasoning-summary")
    if args.improvement_wishes:
        command.append("--improvement-wishes")
    if args.dying_message:
        command.append("--dying-message")
    env = os.environ.copy()
    env.update(
        {
            "DT_DISABLE_DEFAULT_PORTS": "1",
            "DT_PORT_RANGE": f"{start}-{end}",
            "DTAP_ENV_VERIFICATION": "placement",
            "DTAP_ENV_VERIFICATION_STRICT": "1",
            "PYTHONPATH": os.pathsep.join(part for part in (str(args.dtap_root), env.get("PYTHONPATH", "")) if part),
        }
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=args.dtap_root,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        raw_out, raw_err = await process.communicate()
    except asyncio.CancelledError:
        if process.returncode is None:
            try:
                # Let the smoke's asyncio cancellation unwind its nested DTAP
                # runner, which owns a separate process group for victim work.
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        raise
    stdout = raw_out.decode(errors="replace")
    stderr = raw_err.decode(errors="replace")
    (case_dir / "run.stdout.log").write_text(stdout, encoding="utf-8")
    (case_dir / "run.stderr.log").write_text(stderr, encoding="utf-8")
    payload = _passed_payload(stdout)
    result: dict[str, Any] = {
        "status": "passed" if process.returncode == 0 and payload else "failed",
        "domain": task.domain,
        "threat_model": task.threat_model,
        "task_dir": str(task.task_dir),
        "task_id": task.task_id,
        "risk_category": task.risk_category,
        "selection_mode": "explicit" if task.explicit else "profile",
        "selection_profile": "explicit" if task.explicit else args.selection_profile,
        "benchmark_index": task.benchmark_index,
        "policy_model": args.policy_model,
        "victim_model": args.victim_model,
        "victim_agent_type": args.victim_agent_type,
        "max_submissions": args.max_submissions,
        "max_submit_calls": args.max_submit_calls,
        "max_placement_actions": args.max_placement_actions,
        "improvement_wishes_enabled": args.improvement_wishes,
        "dying_message_enabled": args.dying_message,
        "returncode": process.returncode,
        "port_range": f"{start}-{end}",
    }
    if payload:
        result.update(
            {
                key: payload.get(key)
                for key in (
                    "attack_success",
                    "episode_status",
                    "environment_steps",
                    "submissions",
                    "placement_actions",
                    "placements_verified",
                    "matches_source_template",
                    "evaluation_completed",
                    "failure_class",
                    "plan_generated",
                    "action_applied",
                    "episode_id",
                    "placement_applicable",
                    "placement_covered",
                    "placement_verified",
                    "victim_completed",
                    "judge_completed",
                    "victim_mcp_events",
                    "judge_artifacts",
                    "environment_tools",
                    "policy_model",
                    "victim_model",
                    "victim_agent_type",
                    "feedback_mode",
                    "reasoning_summary_enabled",
                    "improvement_wishes_enabled",
                    "dying_message_enabled",
                    "digestor_usage",
                )
            }
        )
    else:
        tail = (stderr or stdout)[-2000:]
        result["error_tail"] = tail
        result["evaluation_completed"] = False
        result["failure_class"] = _failure_class(result)
        try:
            state = json.loads((case_dir / "episode-state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        result.update(
            {
                "plan_generated": bool(state.get("plan_generated") or (case_dir / "submitted-config.yaml").is_file()),
                "action_applied": bool(state.get("evaluation_delegate_completed")),
                "victim_completed": bool(state.get("evaluation_delegate_completed")),
                "judge_completed": bool(state.get("judge_artifacts_retained")),
                "victim_mcp_events": int(bool(state.get("victim_mcp_log_retained"))),
                "judge_artifacts": int(state.get("judge_artifacts_retained") or 0),
                "placement_verified": False,
            }
        )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


async def _main(args: argparse.Namespace) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    if shutil.which("openclaw") is None:
        raise RuntimeError("openclaw CLI is required")
    highest = args.port_range_start + (args.max_parallel - 1) * args.port_range_stride + 511
    if args.port_range_stride < 512 or highest > 65535:
        raise ValueError("parallel workers require disjoint valid 512-port ranges")
    tasks = _matrix_tasks(args)
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    write_resolved_experiment(args)
    queue: asyncio.Queue[MatrixTask] = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)
    results: list[dict[str, Any]] = []

    async def worker(slot: int) -> None:
        while not queue.empty():
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results.append(
                    await _run_case(
                        args,
                        task=task,
                        slot=slot,
                    )
                )
            finally:
                queue.task_done()

    await asyncio.gather(*(worker(slot) for slot in range(args.max_parallel)))
    def result_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            item["domain"],
            item["threat_model"],
            item.get("risk_category", ""),
            item.get("task_id", ""),
        )
    results.sort(key=result_key)
    stored = _stored_results(args.artifacts_root, tasks)
    stored.sort(key=result_key)
    explicit_selection = bool(args.selected_tasks)
    summary = {
        "benchmark_manifest": {
            "schema_version": BENCHMARK_MANIFEST["schema_version"],
            "sha256": MANIFEST_SHA256,
        },
        "selection_mode": "explicit" if explicit_selection else "profile",
        "selection_profile": "explicit" if explicit_selection else args.selection_profile,
        "benchmark_index": (
            None if explicit_selection else SELECTION_PROFILES[args.selection_profile]
        ),
        "selected_tasks": list(args.selected_tasks),
        "policy_model": args.policy_model,
        "victim_model": args.victim_model,
        "victim_agent_type": args.victim_agent_type,
        "max_submissions": args.max_submissions,
        "max_submit_calls": args.max_submit_calls,
        "max_placement_actions": args.max_placement_actions,
        "placement_enabled": args.placement_enabled,
        "improvement_wishes_enabled": args.improvement_wishes,
        "dying_message_enabled": args.dying_message,
        "feedback_mode": args.feedback_mode,
        "reasoning_summary_enabled": args.reasoning_summary,
        "total": len(stored),
        "passed": sum(item["status"] == "passed" for item in stored),
        "failed": _failure_count(stored),
        "selected_total": len(results),
        "selected_failed": _failure_count(results),
        "metrics": _summary_metrics(stored),
        "selected_metrics": _summary_metrics(results),
        "excluded_platform_domains": sorted(EXCLUDED_PLATFORM_DOMAINS),
        "results": stored,
    }
    (args.artifacts_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if summary["selected_failed"] else 0


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    bootstrap_args, _ = bootstrap.parse_known_args()
    config_defaults: dict[str, Any] = {}
    if bootstrap_args.config is not None:
        try:
            config_defaults = load_experiment_config(bootstrap_args.config)
        except ExperimentConfigError as exc:
            bootstrap.error(str(exc))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="strict experiment-v1 YAML; explicit CLI options override its values",
    )
    parser.add_argument(
        "--dtap-root",
        type=Path,
        required=not config_defaults,
        default=config_defaults.get("dtap_root"),
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        required=not config_defaults,
        default=config_defaults.get("artifacts_root"),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=ALL_DOMAINS,
        default=config_defaults.get("domains", list(DOMAINS)),
        help=(
            "manifest domains to run; VM-backed domains are excluded by default "
            "and naming them here is the explicit opt-in"
        ),
    )
    parser.add_argument(
        "--threat-models",
        nargs="+",
        choices=THREAT_MODELS,
        default=config_defaults.get("threat_models", list(THREAT_MODELS)),
    )
    parser.add_argument(
        "--selection-profile",
        choices=tuple(SELECTION_PROFILES),
        default=config_defaults.get("selection_profile", "release-v1"),
        help="manifest-defined benchmark record selection (holdout-v1 is disjoint)",
    )
    parser.add_argument(
        "--tasks",
        dest="selected_tasks",
        nargs="+",
        default=config_defaults.get("selected_tasks", []),
        help=(
            "exact dataset paths relative to dataset/, each shaped as "
            "<domain>/malicious/<threat_model>/<risk_category>/<task_id>"
        ),
    )
    parser.add_argument("--max-parallel", type=int, default=config_defaults.get("max_parallel", 2))
    parser.add_argument(
        "--port-range-start", type=int, default=config_defaults.get("port_range_start", 20_000)
    )
    parser.add_argument(
        "--port-range-stride", type=int, default=config_defaults.get("port_range_stride", 1_024)
    )
    parser.add_argument("--python", default=config_defaults.get("python", sys.executable))
    parser.add_argument("--policy-model", default=config_defaults.get("policy_model", "deepseek-v4-flash"))
    parser.add_argument(
        "--planning-strategy", default=config_defaults.get("planning_strategy", "current")
    )
    parser.add_argument(
        "--harness-protocol",
        choices=HARNESS_PROTOCOLS,
        default=config_defaults.get("harness_protocol", "v1"),
    )
    parser.add_argument(
        "--improvement-wishes",
        action=argparse.BooleanOptionalAction,
        default=config_defaults.get("improvement_wishes", False),
    )
    parser.add_argument(
        "--dying-message",
        action=argparse.BooleanOptionalAction,
        default=config_defaults.get("dying_message", False),
    )
    parser.add_argument("--victim-model", default=config_defaults.get("victim_model", "deepseek-v4-flash"))
    parser.add_argument(
        "--victim-agent-type", default=config_defaults.get("victim_agent_type", "openclaw")
    )
    parser.add_argument(
        "--policy-max-turns",
        type=int,
        default=config_defaults.get("policy_max_turns"),
        help="Claude policy turn budget; defaults to max(64, 32 * H)",
    )
    parser.add_argument(
        "--victim-max-turns", type=int, default=config_defaults.get("victim_max_turns", 80)
    )
    parser.add_argument(
        "--max-submissions", type=int, default=config_defaults.get("max_submissions", 2)
    )
    parser.add_argument(
        "--max-submit-calls",
        type=int,
        default=config_defaults.get("max_submit_calls"),
        help="Q submit-call budget; defaults to max(6, 3 * H)",
    )
    parser.add_argument(
        "--max-placement-actions",
        type=int,
        default=config_defaults.get("max_placement_actions", 8),
    )
    parser.add_argument(
        "--placement",
        dest="placement_enabled",
        action=argparse.BooleanOptionalAction,
        default=config_defaults.get("placement_enabled", True),
    )
    parser.add_argument(
        "--feedback-mode",
        choices=("disabled", "final", "final+deterministic", "final+deterministic+digestor"),
        default=config_defaults.get("feedback_mode", "disabled"),
    )
    parser.add_argument("--digestor-model", default=config_defaults.get("digestor_model", "glm-5.2"))
    parser.add_argument(
        "--digestor-max-tokens",
        type=int,
        default=config_defaults.get("digestor_max_tokens", 50_000),
    )
    parser.add_argument(
        "--digestor-timeout", type=float, default=config_defaults.get("digestor_timeout", 30.0)
    )
    parser.add_argument(
        "--reasoning-summary",
        action=argparse.BooleanOptionalAction,
        default=config_defaults.get("reasoning_summary", False),
    )
    parser.add_argument("--timeout", type=int, default=config_defaults.get("timeout", 1800))
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=config_defaults.get("resume", False),
    )
    args = parser.parse_args()
    args.dtap_root = args.dtap_root.expanduser().resolve()
    args.artifacts_root = args.artifacts_root.expanduser().resolve()
    invalid_domains = sorted(set(args.domains) - set(ALL_DOMAINS))
    if invalid_domains:
        parser.error(f"invalid configured domain(s): {', '.join(invalid_domains)}")
    invalid_threat_models = sorted(set(args.threat_models) - set(THREAT_MODELS))
    if invalid_threat_models:
        parser.error(
            f"invalid configured threat model(s): {', '.join(invalid_threat_models)}"
        )
    if args.selection_profile not in SELECTION_PROFILES:
        parser.error(f"invalid configured selection profile: {args.selection_profile}")
    if len(set(args.selected_tasks)) != len(args.selected_tasks):
        parser.error("--tasks must not contain duplicates")
    if args.max_parallel > MAX_PARALLEL:
        parser.error(f"--max-parallel must be <= {MAX_PARALLEL}")
    for name in (
        "max_parallel",
        "victim_max_turns",
        "max_submissions",
        "timeout",
        "port_range_start",
        "port_range_stride",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_submit_calls is None:
        args.max_submit_calls = max(6, args.max_submissions * 3)
    if args.max_submit_calls < args.max_submissions:
        parser.error("--max-submit-calls must be >= --max-submissions")
    if args.max_placement_actions < 1:
        parser.error("--max-placement-actions must be positive")
    if args.digestor_max_tokens < 64:
        parser.error("--digestor-max-tokens must be >= 64")
    try:
        args.policy_max_turns = policy_max_turn_budget(
            args.max_submissions, args.policy_max_turns
        )
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
