"""Fresh-process DTAP attempt runner used at the M3 mutation boundary."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
import time
import uuid
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
    mcp_events_path: Path | None = None
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
    """Run candidates through an isolated DTAP subprocess and consume judge files.

    Cold mode creates one subprocess per H. Warm mode keeps one process,
    TaskExecutor pool, and compatible MCP services for the episode, while each
    H still resets its environments and creates a fresh victim and judge run.
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
        attempt_runtime: str = "cold",
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
        if attempt_runtime not in {"cold", "warm"}:
            raise ValueError("attempt_runtime must be 'cold' or 'warm'")
        if attempt_runtime == "warm" and max_parallel != 1:
            raise ValueError("warm attempt runtime requires max_parallel=1")
        self.attempt_runtime = attempt_runtime
        self._warm_process: Any | None = None
        self._warm_stack: AsyncExitStack | None = None
        self._warm_lock = asyncio.Lock()
        self._warm_runtime_identity: str | None = None
        self._warm_fallbacks = 0
        self._warm_worker_starts = 0
        self._warm_attempts = 0
        self._warm_disabled = False
        self._attempt_durations_seconds: list[float] = []
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
        command.extend(
            ["--started-path", str(workspace.output_root / ".m4-started")]
        )
        if self.security_policy is not None:
            command.extend(
                [
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

    @staticmethod
    def _single_regular_artifact(root: Path, pattern: str) -> Path | None:
        try:
            resolved_root = root.resolve(strict=True)
        except OSError:
            return None
        matches: list[Path] = []
        for candidate in resolved_root.rglob(pattern):
            try:
                info = candidate.lstat()
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(resolved_root)
            except (OSError, ValueError):
                continue
            if candidate.is_symlink() or not candidate.is_file() or info.st_size < 0:
                continue
            matches.append(resolved)
        return matches[0] if len(matches) == 1 else None

    @classmethod
    def _single_trajectory_artifact(cls, root: Path) -> Path | None:
        try:
            resolved_root = root.resolve(strict=True)
        except OSError:
            return None
        matches: list[Path] = []
        for candidate in resolved_root.rglob("*.json"):
            if candidate.name in {"judge_result.json", ".m4-verdict.json"}:
                continue
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(resolved_root)
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("trajectory"), list):
                matches.append(resolved)
        return matches[0] if len(matches) == 1 else None

    def _worker_command(self) -> list[str]:
        helper = Path(__file__).resolve().parent / "scripts" / "run_dtap_attempt_worker.py"
        return [self.python_executable, str(helper), "--model", self.model]

    async def _start_warm_worker(self) -> Any:
        if self._warm_process is not None and self._warm_process.returncode is None:
            return self._warm_process
        stack = AsyncExitStack()
        await stack.__aenter__()
        process = None
        try:
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
            process = await asyncio.create_subprocess_exec(
                *self._worker_command(),
                cwd=str(self.dtap_root) if self.dtap_root is not None else None,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                limit=8 * 1024 * 1024,
            )
            assert process.stdout is not None
            while True:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=60)
                if not line:
                    raise RuntimeError("warm DTAP worker exited before ready")
                if line.decode("utf-8", errors="replace").strip() == "[DTAP_WARM_READY]":
                    break
        except Exception:
            if process is not None:
                await self._kill_process_group(process)
            await stack.aclose()
            raise
        self._warm_process = process
        self._warm_stack = stack
        self._warm_runtime_identity = str(process.pid)
        self._warm_worker_starts += 1
        return process

    async def _stop_warm_worker(self, *, interrupt: bool = False) -> None:
        process = self._warm_process
        stack = self._warm_stack
        self._warm_process = None
        self._warm_stack = None
        if process is not None and process.returncode is None:
            try:
                if interrupt:
                    os.killpg(process.pid, signal.SIGINT)
                else:
                    assert process.stdin is not None
                    process.stdin.write(b'{"command":"shutdown"}\n')
                    await process.stdin.drain()
                await asyncio.wait_for(process.communicate(), timeout=30)
            except Exception:
                await self._kill_process_group(process)
        if stack is not None:
            await stack.aclose()

    async def aclose(self) -> None:
        """Release the episode-scoped worker and all retained environments."""
        async with self._warm_lock:
            await self._stop_warm_worker()

    @staticmethod
    def _started_marker(workspace: AttemptWorkspace) -> bool:
        started_path = workspace.output_root / ".m4-started"
        try:
            info = started_path.lstat()
            return (
                not started_path.is_symlink()
                and info.st_size == 1
                and started_path.read_bytes() == b"1"
            )
        except OSError:
            return False

    async def _run_warm_once(self, workspace: AttemptWorkspace) -> tuple[str, str | None, bool]:
        process = await self._start_warm_worker()
        self._warm_attempts += 1
        request_id = uuid.uuid4().hex
        payload = {
            "command": "run",
            "request_id": request_id,
            "task_dir": str(workspace.task_dir),
            "output_root": str(workspace.output_root),
            "attempt_index": workspace.attempt_index,
            "agent_type": self.agent_type,
            "model": self.model,
            "max_turns": self.max_turns,
            "temperature": self.temperature,
            "debug": self.debug,
            "started_path": str(workspace.output_root / ".m4-started"),
            "verdict_path": str(workspace.output_root / ".m4-verdict.json") if self.security_policy else None,
        }
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write((json.dumps(payload, sort_keys=True) + "\n").encode())
        await process.stdin.drain()
        async def receive() -> None:
            while True:
                raw = await process.stdout.readline()
                if not raw:
                    raise RuntimeError("warm DTAP worker exited during attempt")
                line = raw.decode("utf-8", errors="replace")
                if line.startswith("[DTAP_WARM_RESULT] "):
                    response = json.loads(line.removeprefix("[DTAP_WARM_RESULT] "))
                    if response.get("request_id") != request_id:
                        raise RuntimeError("warm DTAP worker response identity mismatch")
                    if response.get("error"):
                        raise RuntimeError(str(response["error"]))
                    return

        if self.timeout_seconds is None:
            await receive()
        else:
            await asyncio.wait_for(receive(), timeout=self.timeout_seconds)
        return "", self._warm_runtime_identity, False

    @property
    def runtime_metrics(self) -> Mapping[str, Any]:
        return {
            "mode": self.attempt_runtime,
            "effective_mode": "cold" if self._warm_disabled else self.attempt_runtime,
            "warm_worker_starts": self._warm_worker_starts,
            "warm_attempts": self._warm_attempts,
            "warm_reused_attempts": max(0, self._warm_attempts - self._warm_worker_starts),
            "warm_fallbacks": self._warm_fallbacks,
            "attempt_durations_seconds": list(self._attempt_durations_seconds),
        }

    async def _run_once(self, workspace: AttemptWorkspace) -> AttemptResult:
        if self.attempt_runtime == "warm" and not self._warm_disabled:
            workspace.output_root.mkdir(parents=True, exist_ok=True)
            async with self._warm_lock:
                try:
                    output, runtime_identity, runtime_destroyed = await self._run_warm_once(workspace)
                except asyncio.CancelledError:
                    await self._stop_warm_worker(interrupt=True)
                    raise
                except Exception:
                    started = self._started_marker(workspace)
                    await self._stop_warm_worker(interrupt=True)
                    if started:
                        return AttemptResult.infrastructure_failure(
                            stage="warm_runtime",
                            evaluation_started=True,
                            runtime_identity=self._warm_runtime_identity,
                        )
                    self._warm_fallbacks += 1
                    self._warm_disabled = True
                    return await self._run_cold_once(workspace)
            return self._collect_result(
                workspace,
                output=output,
                runtime_identity=runtime_identity,
                runtime_destroyed=runtime_destroyed,
            )
        return await self._run_cold_once(workspace)

    async def _run_cold_once(self, workspace: AttemptWorkspace) -> AttemptResult:
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
                    evaluation_started=self._started_marker(workspace),
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
                    evaluation_started=self._started_marker(workspace),
                    runtime_identity=runtime_identity,
                )

            output = stdout.decode("utf-8", errors="replace")
            self._retain_stderr_diagnostic(workspace, stderr, env)

        return self._collect_result(
            workspace,
            output=output,
            runtime_identity=runtime_identity,
            runtime_destroyed=True,
        )

    def _collect_result(
        self,
        workspace: AttemptWorkspace,
        *,
        output: str,
        runtime_identity: str | None,
        runtime_destroyed: bool,
    ) -> AttemptResult:
        # Resolve artifact identity once, at the trusted attempt boundary, and
        # pass these exact paths to feedback/export consumers.  Never let those
        # consumers independently choose a different "first" or "latest" file.
        trajectory_path = self._single_trajectory_artifact(workspace.output_root)
        mcp_events_path = self._single_regular_artifact(
            workspace.output_root,
            "*.mcp-events.jsonl",
        )
        if self.security_policy is None:
            evaluation_started = (
                "[DTAP_STATUS] phase=running" in output
                or self._started_marker(workspace)
            )
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
                trajectory_path=trajectory_path,
                mcp_events_path=mcp_events_path,
                runtime_identity=runtime_identity,
                runtime_destroyed=runtime_destroyed,
            )

        evaluation_started = self._started_marker(workspace)
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
            trajectory_path=trajectory_path,
            mcp_events_path=mcp_events_path,
            runtime_identity=runtime_identity,
            runtime_destroyed=runtime_destroyed,
        )

    async def run(self, workspace: AttemptWorkspace) -> AttemptResult:
        start = time.monotonic()
        try:
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
        finally:
            self._attempt_durations_seconds.append(round(time.monotonic() - start, 3))
