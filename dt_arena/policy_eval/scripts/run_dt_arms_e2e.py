"""Run pinned DT Arms natively, then replay its successful YAML once in fresh DTAP."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import shutil
from pathlib import Path
from typing import Any

import yaml

from dt_arena.policy_eval.dtap_compat import load_dtap_api
from dt_arena.policy_eval.dt_arms_integration import (
    DtArmsIntegrationError,
    discover_generation_artifacts,
    episode_id,
    file_sha256,
    merge_generated_attack,
    normalize_dt_arms_trajectory,
    replay_candidate,
    snapshot_provenance,
    summarize_dt_arms_judge_history,
    task_identity,
    write_dt_arms_judge_history,
    write_task_list,
)


DT_ARMS_OPENAI_API_KEY_ENV = "DTAP_ARMS_OPENAI_API_KEY"
DT_ARMS_OPENAI_BASE_URL_ENV = "DTAP_ARMS_OPENAI_BASE_URL"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stage(generation: Path, stage: str, **extra: Any) -> None:
    _atomic_json(generation / "status.json", {"stage": stage, **extra})


def _environment_steps(config: dict[str, Any]) -> list[dict[str, Any]]:
    attack = config.get("Attack") if isinstance(config, dict) else None
    turns = attack.get("attack_turns", []) if isinstance(attack, dict) else []
    return [
        step
        for turn in turns
        if isinstance(turn, dict)
        for step in turn.get("attack_steps", []) or []
        if isinstance(step, dict) and step.get("type") == "environment"
    ]


def _emit(payload: dict[str, Any], artifacts: Path) -> None:
    _atomic_json(artifacts / "result.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _generation_command(args: argparse.Namespace, task_list: Path, generation: Path) -> list[str]:
    command = [
        args.python,
        "-m",
        "dt_arms.run",
        "--task-list",
        str(task_list),
        "--max-parallel",
        "1",
        "--run-all",
        "--agent-model",
        args.attacker_model,
        "--victim-model",
        args.victim_model,
        "--victim-arch",
        args.victim_agent_type,
        "--judge-model",
        args.judge_model,
        "--max-iterations",
        str(args.max_iterations),
        "--output-dir",
        str(generation / "raw"),
        "--port-range",
        f"{args.port_range_start}-{args.port_range_start + 511}",
        "--memory-save-mode",
        args.memory_save_mode,
        "--max-turns-per-session",
        str(args.max_turns_per_session),
    ]
    if args.use_memory:
        command.append("--use-memory")
    if args.update_memory:
        command.append("--update-memory")
    if args.auto_aggregate_memory:
        command.append("--auto-aggregate-memory")
    if not args.allow_quit:
        command.append("--no-allow-quit")
    if args.multi_turn:
        command.append("--multi-turn")
    if args.injection_override:
        command.extend(["--injection-override", args.injection_override])
    if args.allowed_skill_names:
        command.extend(["--allowed-skill-names", args.allowed_skill_names])
    if args.allowed_skill_types:
        command.extend(["--allowed-skill-types", args.allowed_skill_types])
    return command


def _generation_environment(args: argparse.Namespace) -> dict[str, str]:
    """Build the native DT Arms environment without persisting credentials.

    DT Arms intentionally retains upstream provider routing. Its ordinary
    model-name path uses the OpenAI SDK, while the DTAP viewer commonly receives
    one Ollama credential through provider-neutral server configuration. The
    two DTAP_ARMS_* variables form an explicit process-boundary adapter; they
    are consumed here and are never written to YAML or artifacts.
    """

    env = os.environ.copy()
    api_key = env.pop(DT_ARMS_OPENAI_API_KEY_ENV, "").strip()
    base_url = env.pop(DT_ARMS_OPENAI_BASE_URL_ENV, "").strip()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    if base_url:
        if not base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError(f"invalid {DT_ARMS_OPENAI_BASE_URL_ENV}")
        env["OPENAI_BASE_URL"] = base_url.rstrip("/")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(args.dtap_root), env.get("PYTHONPATH", "")) if value
    )
    env["DTAP_DATASET_ROOT"] = str(args.dtap_root / "dataset")
    return env


async def _stop_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def _run(args: argparse.Namespace) -> int:
    artifacts = args.artifacts_dir.resolve()
    generation = artifacts / "dt-arms"
    generation.mkdir(parents=True, exist_ok=True)
    identity = task_identity(args.task_dir)
    provenance = {**snapshot_provenance(), "policy_engine": "dt-arms-upstream"}
    _atomic_json(generation / "provenance.json", provenance)
    shutil.copy2(args.task_dir / "config.yaml", artifacts / "original-config.yaml")
    task_list = write_task_list(args.task_dir, generation / "task.jsonl")
    _stage(generation, "dt_arms_starting", max_iterations=args.max_iterations)
    try:
        env = _generation_environment(args)
    except ValueError as exc:
        _stage(generation, "pipeline_error", error="dt_arms_provider_config")
        _emit(
            {
                "status": "failed",
                **identity,
                "policy_engine": "dt-arms-upstream",
                "failure_class": "provider_config",
                "evaluation_completed": False,
                "error": str(exc),
            },
            artifacts,
        )
        return 1
    process = await asyncio.create_subprocess_exec(
        *_generation_command(args, task_list, generation),
        cwd=args.dtap_root,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=args.timeout)
    except asyncio.TimeoutError:
        await _stop_process_group(process)
        _stage(generation, "pipeline_error", error="dt_arms_timeout")
        _emit(
            {
                "status": "failed",
                **identity,
                "policy_engine": "dt-arms-upstream",
                "failure_class": "dt_arms_runtime",
                "evaluation_completed": False,
            },
            artifacts,
        )
        return 1
    (generation / "stdout.log").write_bytes(stdout)
    (generation / "stderr.log").write_bytes(stderr)
    try:
        attack_path, trajectory_path = discover_generation_artifacts(generation / "raw")
    except DtArmsIntegrationError as exc:
        _stage(generation, "pipeline_error", error=str(exc))
        _emit(
            {
                "status": "failed",
                **identity,
                "policy_engine": "dt-arms-upstream",
                "failure_class": "dt_arms_artifacts",
                "evaluation_completed": False,
                "error": str(exc),
            },
            artifacts,
        )
        return 1
    if process.returncode != 0:
        _stage(generation, "pipeline_error", error="dt_arms_process", returncode=process.returncode)
        _emit(
            {
                "status": "failed",
                **identity,
                "policy_engine": "dt-arms-upstream",
                "failure_class": "dt_arms_runtime",
                "evaluation_completed": False,
                "returncode": process.returncode,
            },
            artifacts,
        )
        return 1
    if trajectory_path is not None:
        raw_trajectory = generation / "trajectory.json"
        shutil.copy2(trajectory_path, raw_trajectory)
        try:
            normalize_dt_arms_trajectory(raw_trajectory, artifacts / "policy.jsonl")
            write_dt_arms_judge_history(raw_trajectory, generation / "judge-history.json")
        except (DtArmsIntegrationError, OSError, ValueError) as exc:
            _stage(generation, "pipeline_error", error=str(exc))
            _emit(
                {
                    "status": "failed",
                    **identity,
                    "policy_engine": "dt-arms-upstream",
                    "failure_class": "dt_arms_trajectory",
                    "evaluation_completed": False,
                    "error": str(exc),
                },
                artifacts,
            )
            return 1
    if attack_path is None:
        native_summary: dict[str, Any] = {}
        history_path = generation / "judge-history.json"
        if history_path.is_file():
            try:
                native_summary = summarize_dt_arms_judge_history(
                    json.loads(history_path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                native_summary = {}
        native_evaluated = native_summary.get("evaluation_completed") is True
        _stage(generation, "generation_exhausted", candidate_generated=False)
        _emit(
            {
                "status": "passed",
                **identity,
                "episode_id": episode_id(),
                "episode_status": "generation_exhausted",
                "policy_engine": "dt-arms-upstream",
                "policy_model": args.attacker_model,
                "victim_model": args.victim_model,
                "victim_agent_type": args.victim_agent_type,
                "dt_arms_success": False,
                "dt_arms_max_iterations": args.max_iterations,
                "candidate_generated": False,
                "evaluation_completed": native_evaluated,
                "attack_success": native_summary.get("attack_success"),
                "task_success": native_summary.get("task_success"),
                "plan_generated": False,
                "action_applied": False,
                "submissions": native_summary.get("submissions", 0),
                "victim_completed": native_summary.get("victim_completed", False),
                "judge_completed": native_summary.get("judge_completed", False),
            },
            artifacts,
        )
        return 0
    retained_attack = generation / "attack-result.yaml"
    shutil.copy2(attack_path, retained_attack)
    try:
        merged = merge_generated_attack(args.task_dir / "config.yaml", retained_attack)
    except DtArmsIntegrationError as exc:
        _stage(generation, "pipeline_error", error=str(exc))
        _emit(
            {
                "status": "failed",
                **identity,
                "policy_engine": "dt-arms-upstream",
                "failure_class": "candidate_validation",
                "candidate_generated": True,
                "evaluation_completed": False,
            },
            artifacts,
        )
        return 1
    submitted = artifacts / "submitted-config.yaml"
    submitted.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _stage(
        generation,
        "authoritative_replay",
        candidate_generated=True,
        candidate_sha256=file_sha256(retained_attack),
    )
    result, judge_artifacts = await replay_candidate(
        source_task_dir=args.task_dir,
        merged_config=merged,
        artifacts_dir=artifacts,
        dtap_root=args.dtap_root,
        python_executable=args.python,
        victim_model=args.victim_model,
        victim_agent_type=args.victim_agent_type,
        victim_max_turns=args.victim_max_turns,
        timeout_seconds=args.timeout,
        port_range_start=args.port_range_start,
    )
    if result.is_infrastructure_failure:
        _stage(generation, "pipeline_error", error=result.infrastructure_stage)
        _emit(
            {
                "status": "failed",
                **identity,
                "policy_engine": "dt-arms-upstream",
                "failure_class": result.infrastructure_stage or "infrastructure",
                "candidate_generated": True,
                "evaluation_completed": False,
            },
            artifacts,
        )
        return 1
    env_steps = _environment_steps(merged)
    _stage(generation, "judge_complete", attack_success=result.attack_success)
    payload = {
        "status": "passed",
        **identity,
        "episode_id": episode_id(),
        "episode_status": "succeeded" if result.attack_success else "evaluated",
        "policy_engine": "dt-arms-upstream",
        "policy_model": args.attacker_model,
        "victim_model": args.victim_model,
        "victim_agent_type": args.victim_agent_type,
        "dt_arms_success": True,
        "dt_arms_max_iterations": args.max_iterations,
        "candidate_generated": True,
        "evaluation_completed": True,
        "attack_success": result.attack_success,
        "task_success": result.task_success,
        "plan_generated": True,
        "action_applied": True,
        "submissions": 1,
        "environment_steps": len(env_steps),
        "environment_tools": [str(step.get("injection_mcp_tool")) for step in env_steps],
        "placement_applicable": False,
        "placement_covered": False,
        "placement_actions": 0,
        "placements_verified": 0,
        "victim_completed": True,
        "judge_completed": True,
        "victim_mcp_events": int(result.mcp_events_path is not None),
        "judge_artifacts": judge_artifacts,
    }
    _emit(payload, artifacts)
    return 0


def main() -> None:
    load_dtap_api()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--dtap-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--attacker-model", required=True)
    parser.add_argument("--victim-model", required=True)
    parser.add_argument("--victim-agent-type", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--victim-max-turns", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--port-range-start", type=int, default=20_000)
    parser.add_argument("--use-memory", action="store_true")
    parser.add_argument("--update-memory", action="store_true")
    parser.add_argument("--memory-save-mode", choices=("all", "success"), default="success")
    parser.add_argument("--auto-aggregate-memory", action="store_true")
    parser.add_argument("--allow-quit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-turn", action="store_true")
    parser.add_argument("--max-turns-per-session", type=int, default=5)
    parser.add_argument("--injection-override")
    parser.add_argument("--allowed-skill-names")
    parser.add_argument("--allowed-skill-types")
    args = parser.parse_args()
    args.task_dir = args.task_dir.resolve()
    args.dtap_root = args.dtap_root.resolve()
    args.artifacts_dir = args.artifacts_dir.resolve()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
