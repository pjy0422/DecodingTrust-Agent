"""Run exactly one DTAP task in one fresh TaskExecutor process."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path


VICTIM_BASE_URL_ENV = "DTAP_VICTIM_ANTHROPIC_BASE_URL"
VICTIM_AUTH_FROM_API_KEY_ENV = "DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN"
MEDICAL_AUX_MODE_ENV = "DTAP_MEDICAL_AUX_MODE"


def judge_infrastructure_error(result: object) -> str | None:
    """Use DTAP's structured stage status at the reward boundary."""

    if not isinstance(result, dict):
        return "judge_result"
    if result.get("error"):
        return "judge_error"
    for name in ("task", "attack"):
        status = result.get(f"{name}_status")
        if status == "unavailable":
            return "judge_unavailable"
        if status not in {"completed", "not_applicable", "not_requested"}:
            return "judge_result"
    return None


def _configure_victim_provider(model: str) -> None:
    """Translate trusted runner-only aliases inside the isolated DTAP child.

    The M4 parent still refuses to inherit policy adapter variables directly.
    These aliases must be explicitly allowlisted by the trusted runner config.
    """

    base_url = os.environ.pop(VICTIM_BASE_URL_ENV, "").strip()
    auth_from_key = os.environ.pop(VICTIM_AUTH_FROM_API_KEY_ENV, "") == "1"
    if base_url:
        if not base_url.startswith(("https://", "http://")):
            raise RuntimeError("invalid trusted victim provider URL")
        os.environ["ANTHROPIC_BASE_URL"] = base_url
    if auth_from_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("victim API key is required for auth-token mapping")
        os.environ["ANTHROPIC_AUTH_TOKEN"] = api_key
        # Some DTAP environments (notably Medical) contain an auxiliary model
        # behind an OpenAI-compatible client. Keep that simulator on the same
        # trusted provider and credential as the victim instead of silently
        # falling back to api.openai.com with an empty key.
        if base_url:
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_BASE_URL"] = f"{base_url.rstrip('/')}/v1"
            os.environ["OPENAI_MODEL"] = model
            os.environ["PATIENT_MODEL"] = model
            os.environ["MEASUREMENT_MODEL"] = model
            os.environ["JUDGE_MODEL"] = model
            os.environ["DTAP_HOSPITAL_OPENAI_COMPAT"] = "1"
            os.environ["DTAP_JUDGE_HTTP_TIMEOUT_SECONDS"] = "180"
            os.environ["OPENCLAW_MCP_TOOL_TIMEOUT_SECONDS"] = "600"


@contextmanager
def _task_environment(instances):
    updates: dict[str, str] = {}
    for instance in instances.values():
        updates.update({str(key): str(value) for key, value in instance.ports.items()})
        name = instance.env_name.upper().replace("-", "_")
        updates[f"{name}_PROJECT_NAME"] = str(instance.project_name)
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _restored_environment(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def _run(args) -> int:
    _configure_victim_provider(args.model)
    from dt_arena.src.types.task import AttackConfig, TaskConfig
    from eval.task_runner import run_single_task
    from utils.task_executor import ScheduledTask, TaskExecutor, get_task_environments

    task_cfg = TaskConfig.from_yaml(str(args.task_dir / "config.yaml"))
    attack_cfg = AttackConfig.from_yaml(str(args.task_dir / "config.yaml"))
    scheduled = ScheduledTask(
        task_dir=args.task_dir,
        environments=frozenset(get_task_environments(args.task_dir)),
        original_index=0,
        domain=getattr(task_cfg, "domain", None),
        threat_model=getattr(attack_cfg, "threat_model", None),
        risk_category=getattr(attack_cfg, "risk_category", None),
        task_id=getattr(task_cfg, "task_id", None),
    )
    executor = TaskExecutor(max_parallel=1)
    auxiliary_mode = os.environ.pop(MEDICAL_AUX_MODE_ENV, "")
    victim_credentials: dict[str, str] = {}
    if getattr(task_cfg, "domain", None) == "medical" and auxiliary_mode == "deterministic":
        # Retained for compatibility with an unpatched DTAP checkout. The
        # managed Hospital compatibility entrypoint accepts arbitrary
        # OpenAI-compatible model names, so release runs no longer select this
        # mode. Hiding credentials also disables the diagnosis judge because it
        # lives in the same container as the patient simulator.
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "REPLICATE_API_TOKEN"):
            value = os.environ.pop(name, None)
            if value is not None:
                victim_credentials[name] = value

    async def callback(task, instances):
        if args.started_path is not None:
            args.started_path.parent.mkdir(parents=True, exist_ok=True)
            args.started_path.write_bytes(b"1")
        with _restored_environment(victim_credentials):
            with _task_environment(instances):
                return await run_single_task(
                    task.task_dir,
                    agent_type=args.agent_type,
                    model=args.model,
                    temperature=args.temperature,
                    max_turns=args.max_turns,
                    skip_mcp=False,
                    skip_judge=False,
                    debug=args.debug,
                    direct_prompt=False,
                )

    try:
        results = await executor.run_all([scheduled], callback)
        if not results:
            return 1
        if args.verdict_path is not None:
            output_root = Path(os.environ["EVAL_RESULTS_ROOT"])
            matches = list(output_root.rglob("judge_result.json"))
            if len(matches) == 1 and matches[0].is_file() and not matches[0].is_symlink():
                judge = json.loads(matches[0].read_bytes())
                failure_stage = judge_infrastructure_error(judge)
                if failure_stage is not None:
                    print(f"[DTAP_STATUS] phase=judge_unavailable stage={failure_stage}")
                    return 2
                verdict = judge.get("attack_success") if isinstance(judge, dict) else None
                if isinstance(verdict, bool):
                    args.verdict_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = args.verdict_path.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps({"attack_success": verdict}, sort_keys=True),
                        encoding="utf-8",
                    )
                    temporary.replace(args.verdict_path)
        return int(results[0][1])
    finally:
        await executor.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--agent-type", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--started-path", type=Path)
    parser.add_argument("--verdict-path", type=Path)
    raise SystemExit(asyncio.run(_run(parser.parse_args())))


if __name__ == "__main__":
    main()
