"""Run one representative policy E2E task per domain and threat model in parallel."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import sys
from collections import Counter
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


def _stored_results(root: Path) -> list[dict[str, Any]]:
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


def _failure_count(results: list[dict[str, Any]]) -> int:
    """Platform names never exempt failures from the test verdict."""
    return sum(item.get("status") != "passed" for item in results)


def _matching_resume_result(
    path: Path,
    *,
    selection_profile: str,
    benchmark_index: int,
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
    domain: str,
    threat_model: str,
    slot: int,
) -> dict[str, Any]:
    record, benchmark_index = _selected_record(
        args.dtap_root / "benchmark" / domain / f"{threat_model}.jsonl",
        args.selection_profile,
    )
    task_dir = _task_dir(args.dtap_root, record)
    case_dir = args.artifacts_root / domain / threat_model
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    if args.resume and result_path.exists():
        previous = _matching_resume_result(
            result_path,
            selection_profile=args.selection_profile,
            benchmark_index=benchmark_index,
            task_id=str(record["task_id"]),
            risk_category=str(record["risk_category"]),
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
        str(task_dir),
        "--dtap-root",
        str(args.dtap_root),
        "--python",
        args.python,
        "--policy-model",
        args.policy_model,
        "--planning-strategy",
        args.planning_strategy,
        "--victim-model",
        args.victim_model,
        "--victim-agent-type",
        args.victim_agent_type,
        "--max-submissions",
        str(args.max_submissions),
        "--policy-max-turns",
        str(args.policy_max_turns),
        "--victim-max-turns",
        str(args.victim_max_turns),
        "--timeout",
        str(args.timeout),
        "--placement",
        "--feedback-mode",
        args.feedback_mode,
        "--digestor-model",
        args.digestor_model,
        "--digestor-timeout",
        str(args.digestor_timeout),
        "--port-range-start",
        str(start),
        "--artifacts-dir",
        str(case_dir),
    ]
    if args.reasoning_summary:
        command.append("--reasoning-summary")
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
        "domain": domain,
        "threat_model": threat_model,
        "task_dir": str(task_dir),
        "task_id": str(record["task_id"]),
        "risk_category": str(record["risk_category"]),
        "selection_profile": args.selection_profile,
        "benchmark_index": benchmark_index,
        "policy_model": args.policy_model,
        "victim_model": args.victim_model,
        "victim_agent_type": args.victim_agent_type,
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
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    for case in matrix_cases(args.domains, args.threat_models):
        queue.put_nowait(case)
    results: list[dict[str, Any]] = []

    async def worker(slot: int) -> None:
        while not queue.empty():
            try:
                domain, threat_model = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results.append(
                    await _run_case(
                        args,
                        domain=domain,
                        threat_model=threat_model,
                        slot=slot,
                    )
                )
            finally:
                queue.task_done()

    await asyncio.gather(*(worker(slot) for slot in range(args.max_parallel)))
    results.sort(key=lambda item: (item["domain"], item["threat_model"]))
    stored = _stored_results(args.artifacts_root)
    stored.sort(key=lambda item: (item["domain"], item["threat_model"]))
    summary = {
        "benchmark_manifest": {
            "schema_version": BENCHMARK_MANIFEST["schema_version"],
            "sha256": MANIFEST_SHA256,
        },
        "selection_profile": args.selection_profile,
        "benchmark_index": SELECTION_PROFILES[args.selection_profile],
        "policy_model": args.policy_model,
        "victim_model": args.victim_model,
        "victim_agent_type": args.victim_agent_type,
        "max_submissions": args.max_submissions,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtap-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=ALL_DOMAINS,
        default=list(DOMAINS),
        help=(
            "manifest domains to run; VM-backed domains are excluded by default "
            "and naming them here is the explicit opt-in"
        ),
    )
    parser.add_argument(
        "--threat-models",
        nargs="+",
        choices=THREAT_MODELS,
        default=list(THREAT_MODELS),
    )
    parser.add_argument(
        "--selection-profile",
        choices=tuple(SELECTION_PROFILES),
        default="release-v1",
        help="manifest-defined benchmark record selection (holdout-v1 is disjoint)",
    )
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--port-range-start", type=int, default=20_000)
    parser.add_argument("--port-range-stride", type=int, default=1_024)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--policy-model", default="deepseek-v4-flash")
    parser.add_argument("--planning-strategy", default="current")
    parser.add_argument("--victim-model", default="deepseek-v4-flash")
    parser.add_argument("--victim-agent-type", default="openclaw")
    parser.add_argument("--policy-max-turns", type=int, default=64)
    parser.add_argument("--victim-max-turns", type=int, default=80)
    parser.add_argument("--max-submissions", type=int, default=2)
    parser.add_argument(
        "--feedback-mode",
        choices=("disabled", "final", "final+deterministic", "final+deterministic+digestor"),
        default="disabled",
    )
    parser.add_argument("--digestor-model", default="glm-5.2")
    parser.add_argument("--digestor-timeout", type=float, default=30.0)
    parser.add_argument("--reasoning-summary", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.dtap_root = args.dtap_root.expanduser().resolve()
    args.artifacts_root = args.artifacts_root.expanduser().resolve()
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
