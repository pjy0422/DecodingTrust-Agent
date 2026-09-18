"""Run an M5 route/placement smoke against one real DTAP task without an LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any


async def run(task_dir: Path, *, strict: bool) -> dict[str, Any]:
    from dt_arena.src.env_verification import (
        SubprocessDockerInspector,
        build_injection_server_overrides,
        build_readback_environments,
        verify_placement_batch,
        verify_started_routes,
    )
    from dt_arena.src.types.agent import AgentConfig
    from dt_arena.src.types.task import AttackConfig

    from utils import PROJECT_ROOT
    from utils.env_helpers import task_setup
    from utils.injection_helpers import (
        apply_environment_injections_async,
        get_env_injections_from_attack,
        get_required_injection_servers,
    )
    from utils.injection_mcp_helpers import start_injection_mcp_servers, wait_for_injection_mcp_ready
    from utils.resource_manager import ResourceManager
    from utils.task_executor import TaskExecutor, get_task_environments

    task_dir = task_dir.resolve()
    config_path = task_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing task config: {config_path}")

    attack = AttackConfig.from_yaml(str(config_path))
    injections = get_env_injections_from_attack(attack)
    if not injections:
        return {"task": str(task_dir), "status": "no_environment_injections"}

    agent = AgentConfig.from_yaml(str(config_path))
    required = get_required_injection_servers(injections)
    executor = TaskExecutor(max_parallel=1)
    injection_manager = None
    run_id = f"m5-smoke-{os.getpid()}"
    original_environment: dict[str, str | None] = {}

    try:
        for environment in sorted(get_task_environments(task_dir)):
            # TaskExecutor has no public single-instance smoke API yet.
            instance = await executor._start_instance(environment)
            if instance is None:
                raise RuntimeError(f"failed to start DTAP environment: {environment}")
            for name, port in instance.ports.items():
                original_environment.setdefault(name, os.environ.get(name))
                os.environ[name] = str(port)
            project_key = f"{environment.upper().replace('-', '_')}_PROJECT_NAME"
            original_environment.setdefault(project_key, os.environ.get(project_key))
            os.environ[project_key] = instance.project_name

        task_setup(task_dir, task_id=run_id)
        injection_config = {
            "environment_enabled": True,
            "environment_servers": required,
        }
        injection_manager, injection_config = start_injection_mcp_servers(
            injection_config,
            resource_manager=ResourceManager.instance(),
            task_id=run_id,
            task_env_overrides=build_injection_server_overrides(agent, list(required)),
        )
        if injection_manager is None:
            raise RuntimeError("no injection MCP server was started")
        wait_for_injection_mcp_ready(injection_config, timeout=30)
        urls = {name: value["url"] for name, value in injection_config["environment_servers"].items()}

        docker = SubprocessDockerInspector.auto()
        route_proofs = verify_started_routes(PROJECT_ROOT, injection_manager, list(required), os.environ, docker)
        results = await apply_environment_injections_async(injections, urls)
        placement_proofs = await verify_placement_batch(
            injections,
            results,
            os.environ,
            docker,
            strict=strict,
            server_environments=build_readback_environments(agent, injection_manager, list(required)),
        )
        return {
            "task": str(task_dir),
            "status": "passed",
            "strict": strict,
            "routes": [
                {
                    "server": server,
                    "environment": proof.environment,
                    "network": list(proof.modes),
                }
                for server, proofs in route_proofs.items()
                for proof in proofs
            ],
            "injections": [
                {
                    "server": result["server_name"],
                    "tool": result["tool_name"],
                    "success": result["success"],
                    "error": result["error"],
                }
                for result in results
            ],
            "placements": [
                {
                    "server": proof.server,
                    "tool": proof.tool,
                    "status": proof.status.value,
                    "locator": proof.locator,
                    "digest": proof.digest,
                }
                for proof in placement_proofs
            ],
        }
    finally:
        if injection_manager is not None:
            injection_manager.stop_all()
        await executor.shutdown()
        ResourceManager.instance().cleanup_task(run_id, verbose=False)
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        result = asyncio.run(run(args.task_dir, strict=args.strict))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
