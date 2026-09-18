"""Fresh-process DTAP attempt runner used at the M3 mutation boundary."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate_config import AttemptWorkspace
from .reward_firewall import JudgeVerdictReader, VerdictError, find_single_judge_result
from .scheduler import AttemptScheduler, PortRangePool, SchedulerSaturated
from .security_policy import EvaluationSecurityPolicy


@dataclass(frozen=True)
class AttemptResult:
    evaluation_started: bool
    attack_success: bool | None
    task_success: bool | None = None
    judge_result: Mapping[str, Any] | None = None
    victim_output: str | None = None
    trajectory_path: Path | None = None
    runtime_identity: str | None = None
    runtime_destroyed: bool = False
    infrastructure_stage: str | None = None

    @classmethod
    def infrastructure_failure(
        cls,
        *,
        stage: str,
        evaluation_started: bool,
        runtime_identity: str | None = None,
        runtime_destroyed: bool = True,
    ) -> AttemptResult:
        return cls(
            evaluation_started=evaluation_started,
            attack_success=None,
            runtime_identity=runtime_identity,
            runtime_destroyed=runtime_destroyed,
            infrastructure_stage=stage,
        )

    @property
    def is_infrastructure_failure(self) -> bool:
        return self.infrastructure_stage is not None or not isinstance(self.attack_success, bool)


class DtapAttemptRunner:
    """Run one candidate via a fresh DTAP subprocess and consume its judge file.

    A subprocess is deliberately used as the reset primitive: DTAP module globals,
    ResourceManager state, victim context, and Docker-pool lifetime are not shared
    between H-counted victim executions.
    """

    def __init__(
        self,
        *,
        max_parallel: int = 1,
        agent_type: str = "openaisdk",
        model: str = "gpt-5.4",
        max_turns: int = 200,
        temperature: float | None = None,
        debug: bool = False,
        timeout_seconds: float | None = None,
        python_executable: str | None = None,
        dtap_root: Path | str | None = None,
        extra_env: Mapping[str, str] | None = None,
        security_policy: EvaluationSecurityPolicy | None = None,
        scheduler: AttemptScheduler | None = None,
        port_range_start: int = 20_000,
        port_pool: PortRangePool | None = None,
    ) -> None:
        if isinstance(max_parallel, bool) or max_parallel < 1:
            raise ValueError("max_parallel must be positive")
        self._semaphore = asyncio.Semaphore(max_parallel)
        self.agent_type = agent_type
        self.model = model
        self.max_turns = max_turns
        self.temperature = temperature
        self.debug = debug
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable or sys.executable
        self.dtap_root = Path(dtap_root).resolve() if dtap_root is not None else None
        self.extra_env = dict(extra_env or {})
        self.security_policy = security_policy
        self.policy_hardened = security_policy is not None
        self.scheduler = scheduler
        if isinstance(port_range_start, bool) or not isinstance(port_range_start, int):
            raise ValueError("port_range_start must be an integer")
        if port_range_start < 1024 or port_range_start + 511 > 65535:
            raise ValueError("port_range_start must reserve 512 valid user ports")
        self.port_range_start = port_range_start
        if security_policy is not None:
            if scheduler is None:
                raise ValueError("M4 requires one explicit worker-scoped scheduler")
            if (
                scheduler.max_parallel != security_policy.max_parallel_attempts
                or scheduler.max_queued != security_policy.max_queued_attempts
            ):
                raise ValueError("scheduler limits do not match the M4 security policy")
            self.verdict_reader = JudgeVerdictReader(security_policy)
            self.port_pool = port_pool or PortRangePool(
                start=port_range_start,
                slots=security_policy.max_parallel_attempts,
            )
            if self.port_pool.slots != security_policy.max_parallel_attempts or self.port_pool.width != 512:
                raise ValueError("port pool does not match the M4 parallel worker policy")
        else:
            self.verdict_reader = None
            if port_pool is not None:
                raise ValueError("port_pool requires an M4 security policy")
            self.port_pool = None

    def _command(self, workspace: AttemptWorkspace) -> list[str]:
        helper = Path(__file__).resolve().parent / "scripts" / "run_dtap_attempt.py"
        command = [
            self.python_executable,
            str(helper),
            "--task-dir",
            str(workspace.task_dir),
            "--agent-type",
            self.agent_type,
            "--model",
            self.model,
            "--max-turns",
            str(self.max_turns),
        ]
        if self.temperature is not None:
            command.extend(["--temperature", str(self.temperature)])
        if self.debug:
            command.append("--debug")
        if self.security_policy is not None:
            command.extend(
                [
                    "--started-path",
                    str(workspace.output_root / ".m4-started"),
                    "--verdict-path",
                    str(workspace.output_root / ".m4-verdict.json"),
                ]
            )
        return command

    @staticmethod
    async def _kill_process_group(process: Any) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
        await process.wait()

    @staticmethod
    def _retain_stderr_diagnostic(workspace: AttemptWorkspace, stderr: bytes, env: Mapping[str, str]) -> None:
        """Keep a bounded trusted diagnostic without retaining credentials."""
        if not stderr:
            return
        text = stderr.decode("utf-8", errors="replace")[-32_768:]
        for name, value in env.items():
            if value and len(value) >= 6 and re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", name, re.I):
                text = text.replace(value, "<redacted>")
        text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]{12,}", r"\1<redacted>", text)
        (workspace.output_root / ".dtap-stderr.log").write_text(text, encoding="utf-8")

    async def _run_once(self, workspace: AttemptWorkspace) -> AttemptResult:
        workspace.output_root.mkdir(parents=True, exist_ok=True)
        async with AsyncExitStack() as stack:
            if self.security_policy is None:
                env = os.environ.copy()
                env.update(self.extra_env)
            else:
                env = self.security_policy.build_dtap_child_env(explicit_env=self.extra_env)
                assert self.port_pool is not None
                port_start, port_end = await stack.enter_async_context(self.port_pool.lease())
                env["DT_DISABLE_DEFAULT_PORTS"] = "1"
                env["DT_PORT_RANGE_START"] = str(port_start)
                env["DT_PORT_RANGE_END"] = str(port_end)
            env["EVAL_RESULTS_ROOT"] = str(workspace.output_root)
            env["DTAP_M4_ATTEMPT_INDEX"] = str(workspace.attempt_index)
            process = None
            runtime_identity = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *self._command(workspace),
                    cwd=str(self.dtap_root) if self.dtap_root is not None else None,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                runtime_identity = str(process.pid)
                communicate = process.communicate()
                if self.timeout_seconds is None:
                    stdout, stderr = await communicate
                else:
                    stdout, stderr = await asyncio.wait_for(communicate, timeout=self.timeout_seconds)
            except asyncio.TimeoutError:
                if process is not None:
                    await self._kill_process_group(process)
                return AttemptResult.infrastructure_failure(
                    stage="timeout",
                    evaluation_started=False,
                    runtime_identity=runtime_identity,
                )
            except asyncio.CancelledError:
                if process is not None:
                    await self._kill_process_group(process)
                raise
            except Exception:
                if process is not None:
                    await self._kill_process_group(process)
                return AttemptResult.infrastructure_failure(
                    stage="process_start",
                    evaluation_started=False,
                    runtime_identity=runtime_identity,
                )

            output = stdout.decode("utf-8", errors="replace")
            self._retain_stderr_diagnostic(workspace, stderr, env)
        if self.security_policy is None:
            evaluation_started = "[DTAP_STATUS] phase=running" in output
            try:
                judge_path = find_single_judge_result(workspace.output_root)
            except VerdictError:
                return AttemptResult.infrastructure_failure(
                    stage="judge_result",
                    evaluation_started=evaluation_started,
                    runtime_identity=runtime_identity,
                )
            try:
                judge = json.loads(judge_path.read_text(encoding="utf-8"))
            except Exception:
                return AttemptResult.infrastructure_failure(
                    stage="judge_result",
                    evaluation_started=evaluation_started,
                    runtime_identity=runtime_identity,
                )
            attack_success = judge.get("attack_success") if isinstance(judge, dict) else None
            if not isinstance(attack_success, bool):
                return AttemptResult.infrastructure_failure(
                    stage="judge_verdict",
                    evaluation_started=evaluation_started,
                    runtime_identity=runtime_identity,
                )
            task_success = judge.get("task_success")
            return AttemptResult(
                evaluation_started=evaluation_started,
                attack_success=attack_success,
                task_success=task_success if isinstance(task_success, bool) else None,
                judge_result=judge,
                runtime_identity=runtime_identity,
                runtime_destroyed=True,
            )

        started_path = workspace.output_root / ".m4-started"
        try:
            started_info = started_path.lstat()
            evaluation_started = (
                not started_path.is_symlink() and started_info.st_size == 1 and started_path.read_bytes() == b"1"
            )
        except OSError:
            evaluation_started = False
        verdict_path = workspace.output_root / ".m4-verdict.json"
        try:
            assert self.verdict_reader is not None
            verdict = self.verdict_reader.read(
                verdict_path,
                result_root=workspace.output_root,
            )
        except (VerdictError, OSError):
            return AttemptResult.infrastructure_failure(
                stage="judge_result",
                evaluation_started=evaluation_started,
                runtime_identity=runtime_identity,
            )
        return AttemptResult(
            evaluation_started=evaluation_started,
            attack_success=verdict.attack_success,
            runtime_identity=runtime_identity,
            runtime_destroyed=True,
        )

    async def run(self, workspace: AttemptWorkspace) -> AttemptResult:
        if self.scheduler is not None:
            try:
                return await self.scheduler.run(lambda: self._run_once(workspace))
            except SchedulerSaturated:
                return AttemptResult.infrastructure_failure(
                    stage="scheduler",
                    evaluation_started=False,
                )
        async with self._semaphore:
            return await self._run_once(workspace)
