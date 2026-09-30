"""Episode-scoped DTAP attempt worker.

The worker keeps Python imports and TaskExecutor environment containers warm
across H attempts. Each request still follows TaskExecutor's reset path, runs
the task setup script, creates a fresh victim agent, and invokes a fresh judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from dt_arena.policy_eval.scripts.run_dtap_attempt import (
    MEDICAL_AUX_MODE_ENV,
    _configure_victim_provider,
    _run,
)
from utils.task_executor import TaskExecutor
from eval.task_runner import ReusableTaskServices
from utils.resource_manager import ResourceManager


RESULT_PREFIX = "[DTAP_WARM_RESULT]"
READY_LINE = "[DTAP_WARM_READY]"


@contextmanager
def _request_environment(values: dict[str, str]):
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


def _request_args(payload: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        task_dir=Path(payload["task_dir"]).resolve(),
        agent_type=str(payload["agent_type"]),
        model=str(payload["model"]),
        max_turns=int(payload["max_turns"]),
        temperature=payload.get("temperature"),
        debug=bool(payload.get("debug", False)),
        started_path=(Path(payload["started_path"]) if payload.get("started_path") else None),
        verdict_path=(Path(payload["verdict_path"]) if payload.get("verdict_path") else None),
    )


async def _serve(model: str) -> int:
    _configure_victim_provider(model)
    executor = TaskExecutor(max_parallel=1, retain_idle_instances=True)
    reusable_services = ReusableTaskServices(ResourceManager.instance())
    auxiliary_mode = os.environ.get(MEDICAL_AUX_MODE_ENV, "")
    print(READY_LINE, flush=True)
    try:
        while True:
            raw = await asyncio.to_thread(sys.stdin.readline)
            if not raw:
                return 0
            payload: dict[str, Any] = {}
            try:
                payload = json.loads(raw)
                if payload.get("command") == "shutdown":
                    return 0
                if payload.get("command") != "run":
                    raise ValueError("unsupported worker command")
                request_id = str(payload["request_id"])
                env = {
                    "EVAL_RESULTS_ROOT": str(payload["output_root"]),
                    "DTAP_M4_ATTEMPT_INDEX": str(payload["attempt_index"]),
                }
                if auxiliary_mode:
                    env[MEDICAL_AUX_MODE_ENV] = auxiliary_mode
                with _request_environment(env):
                    output_root = Path(env["EVAL_RESULTS_ROOT"])
                    output_root.mkdir(parents=True, exist_ok=True)
                    with (output_root / ".dtap-run.log").open("w", encoding="utf-8") as log:
                        with redirect_stdout(log), redirect_stderr(log):
                            return_code = await _run(
                                _request_args(payload),
                                executor=executor,
                                configure_provider=False,
                                reusable_services=reusable_services,
                            )
                response = {"request_id": request_id, "returncode": return_code}
            except Exception as exc:
                response = {
                    "request_id": str(payload.get("request_id", "")) if isinstance(payload, dict) else "",
                    "returncode": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            print(f"{RESULT_PREFIX} {json.dumps(response, sort_keys=True)}", flush=True)
    finally:
        reusable_services.close()
        await executor.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    raise SystemExit(asyncio.run(_serve(parser.parse_args().model)))


if __name__ == "__main__":
    main()
