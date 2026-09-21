"""M6 bounded apply/read-back receipts for environment actions."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import stat
import sys
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate_config import materialize_attempt_dir
from .integrity import BenchmarkManifest
from .policy_contract import PolicyContract, canonical_policy_json
from .scheduler import AttemptScheduler, PortRangePool, SchedulerSaturated
from .security_policy import EvaluationSecurityPolicy, PolicyInputLimitError
from .validation import ValidationContext, validate_attack_step


@dataclass(frozen=True)
class PlacementRunResult:
    available: bool
    applied: bool = False
    valid: bool = False
    status: str = "unavailable"
    locator: str = ""
    code: str = "EVALUATION_UNAVAILABLE"
    locator_fields: tuple[str, ...] = ()
    repair_fields: tuple[str, ...] = ()
    retryable: bool = False


class DtapPlacementRunner:
    """Apply one candidate in a fresh DTAP sandbox and read one sealed result."""

    def __init__(
        self,
        *,
        dtap_root: Path | str,
        security_policy: EvaluationSecurityPolicy,
        scheduler: AttemptScheduler,
        python_executable: str | None = None,
        timeout_seconds: float = 300.0,
        extra_env: Mapping[str, str] | None = None,
        port_pool: PortRangePool | None = None,
    ) -> None:
        self.dtap_root = Path(dtap_root).resolve()
        self.security_policy = security_policy
        self.scheduler = scheduler
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = timeout_seconds
        self.extra_env = dict(extra_env or {})
        self.port_pool = port_pool or PortRangePool(
            start=20_000,
            slots=security_policy.max_parallel_attempts,
        )
        if self.port_pool.slots != security_policy.max_parallel_attempts or self.port_pool.width != 512:
            raise ValueError("port pool does not match the M4 parallel worker policy")
        if (
            scheduler.max_parallel != security_policy.max_parallel_attempts
            or scheduler.max_queued != security_policy.max_queued_attempts
        ):
            raise ValueError("placement scheduler and security policy differ")

    async def _kill(self, process: Any) -> None:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    def _read(self, path: Path, root: Path) -> PlacementRunResult:
        try:
            resolved_root = root.resolve()
            resolved = path.resolve(strict=True)
            info = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(info.st_mode):
                raise ValueError
            if not resolved.is_relative_to(resolved_root) or info.st_size > 64 * 1024:
                raise ValueError
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return PlacementRunResult(False)
        allowed = {
            "schema", "applied", "valid", "status", "locator", "code",
            "locator_fields", "repair_fields", "retryable",
        }
        if (
            not isinstance(value, dict)
            or set(value) - allowed
            or value.get("schema") not in {"m6-placement-v1", "m6-placement-v2"}
        ):
            return PlacementRunResult(False)
        locator_fields = value.get("locator_fields", [])
        fields = value.get("repair_fields", [])
        if (
            not isinstance(locator_fields, list)
            or not all(isinstance(item, str) for item in locator_fields)
            or not isinstance(fields, list)
            or not all(isinstance(item, str) for item in fields)
        ):
            return PlacementRunResult(False)
        scalar = (
            value.get("applied"),
            value.get("valid"),
            value.get("status"),
            value.get("locator", ""),
            value.get("code"),
        )
        if not isinstance(scalar[0], bool) or not isinstance(scalar[1], bool):
            return PlacementRunResult(False)
        if not all(isinstance(item, str) for item in scalar[2:]):
            return PlacementRunResult(False)
        if scalar[2] not in {"verified", "unsupported", "not_applicable", "invalid"}:
            return PlacementRunResult(False)
        allowed_codes = {
            "PLACEMENT_VERIFIED",
            "PLACEMENT_MISMATCH",
            "INJECTION_FAILED",
            "UNSUPPORTED_PLACEMENT",
        }
        if scalar[4] not in allowed_codes or len(scalar[3].encode("utf-8")) > 4096:
            return PlacementRunResult(False)
        if scalar[1] != (scalar[2] == "verified" and scalar[4] == "PLACEMENT_VERIFIED"):
            return PlacementRunResult(False)
        retryable = value.get("retryable", bool(fields))
        if not isinstance(retryable, bool) or retryable != bool(retryable and fields):
            return PlacementRunResult(False)
        for bounded_fields in (locator_fields, fields):
            if len(bounded_fields) > 8 or any(
                len(item) > 128 or not item.startswith("kwargs.") for item in bounded_fields
            ):
                return PlacementRunResult(False)
        return PlacementRunResult(
            available=True,
            applied=scalar[0],
            valid=scalar[1],
            status=scalar[2],
            locator=scalar[3],
            code=scalar[4],
            locator_fields=tuple(locator_fields),
            repair_fields=tuple(fields),
            retryable=retryable,
        )

    async def _run_once(self, workspace: Any) -> PlacementRunResult:
        workspace.output_root.mkdir(parents=True, exist_ok=True)
        result_path = workspace.output_root / ".m6-placement.json"
        helper = Path(__file__).resolve().parent / "scripts" / "run_dtap_placement_probe.py"
        async with AsyncExitStack() as stack:
            env = self.security_policy.build_dtap_child_env(explicit_env=self.extra_env)
            port_start, port_end = await stack.enter_async_context(self.port_pool.lease())
            env["DT_DISABLE_DEFAULT_PORTS"] = "1"
            env["DT_PORT_RANGE_START"] = str(port_start)
            env["DT_PORT_RANGE_END"] = str(port_end)
            env["EVAL_RESULTS_ROOT"] = str(workspace.output_root)
            command = [
                self.python_executable,
                str(helper),
                "--task-dir",
                str(workspace.task_dir),
                "--result-path",
                str(result_path),
                "--target-step-index",
                str(workspace.placement_target_index or 0),
            ]
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(self.dtap_root),
                    env=env,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    start_new_session=True,
                )
                await asyncio.wait_for(process.communicate(), timeout=self.timeout_seconds)
            except asyncio.TimeoutError:
                if process is not None:
                    await self._kill(process)
                return PlacementRunResult(False)
            except asyncio.CancelledError:
                if process is not None:
                    await self._kill(process)
                raise
            except Exception:
                if process is not None:
                    await self._kill(process)
                return PlacementRunResult(False)
        if process.returncode != 0:
            return PlacementRunResult(False)
        return self._read(result_path, workspace.output_root)

    async def run(self, workspace: Any) -> PlacementRunResult:
        try:
            return await self.scheduler.run(lambda: self._run_once(workspace))
        except SchedulerSaturated:
            return PlacementRunResult(False)


class PlacementCoordinator:
    """Episode-scoped M6 action registry; it cannot query arbitrary state."""

    def __init__(
        self,
        *,
        validation_context: ValidationContext,
        source_task_dir: Path | str,
        episode_root: Path | str,
        runner: Any,
        security_policy: EvaluationSecurityPolicy,
        policy_contract: PolicyContract,
        source_manifest: BenchmarkManifest | None = None,
        max_actions: int = 8,
        candidate_validator: Any = None,
    ) -> None:
        self.validation_context = validation_context
        self.source_task_dir = Path(source_task_dir)
        self.episode_root = Path(episode_root)
        self.runner = runner
        self.security_policy = security_policy
        self.policy_contract = policy_contract
        self.source_manifest = source_manifest
        self.max_actions = max_actions
        if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 1:
            raise ValueError("max_actions must be positive")
        self.candidate_validator = candidate_validator
        self._receipts: dict[str, PlacementRunResult] = {}
        self._receipt_steps: dict[str, str] = {}
        self._receipt_step_objects: dict[str, Any] = {}
        self._receipt_replay_steps: dict[str, tuple[Any, ...]] = {}
        self._receipt_ids_by_tool: dict[str, list[str]] = {}
        self._pending_id: str | None = None
        self._attempts = 0
        self._validated_ids: set[str] = set()
        self._lock = asyncio.Lock()

    @staticmethod
    def _qualified_name(step: Any) -> str:
        return str(getattr(step, "injection_mcp_tool", "") or "")

    def _resource_contract(self, step: Any) -> Any:
        qualified = self._qualified_name(step)
        return next(
            (
                getattr(tool, "placement_resource", None)
                for tool in self.validation_context.attack_surface.environment_tools
                if tool.qualified_name == qualified
            ),
            None,
        )

    def _dependency_error(self, code: str) -> dict[str, Any]:
        return self.policy_contract.public_payload(
            {"accepted": False, "error": {"code": code}}
        )

    def _latest_receipt_id(self, tool_name: str, *, require_valid: bool) -> str | None:
        for receipt_id in reversed(self._receipt_ids_by_tool.get(tool_name, ())):
            if receipt_id not in self._validated_ids:
                continue
            if require_valid and not self._receipts[receipt_id].valid:
                continue
            return receipt_id
        return None

    def _resolve_dependencies(self, raw: Any, step: Any) -> tuple[tuple[Any, ...], tuple[str, ...]] | dict[str, Any]:
        if raw is None:
            dependency_tools: tuple[str, ...] = ()
        elif (
            not isinstance(raw, list)
            or len(raw) > self.max_actions
            or any(
                not isinstance(item, str) or not item or len(item) > 256
                for item in raw
            )
            or len(set(raw)) != len(raw)
        ):
            return self._dependency_error("INVALID_DEPENDENCY")
        else:
            dependency_tools = tuple(raw)

        current_contract = self._resource_contract(step)
        replay: list[Any] = []
        seen: set[str] = set()
        dependency_ids: list[str] = []
        for tool_name in dependency_tools:
            if tool_name not in self._receipt_ids_by_tool:
                return self._dependency_error("UNKNOWN_DEPENDENCY")
            action_id = self._latest_receipt_id(tool_name, require_valid=True)
            if action_id is None:
                return self._dependency_error("UNVERIFIED_DEPENDENCY")
            dependency_ids.append(action_id)
            dependency_step = self._receipt_step_objects[action_id]
            dependency_contract = self._resource_contract(dependency_step)
            if (
                current_contract is None
                or current_contract.role != "consumer"
                or dependency_contract is None
                or dependency_contract.role != "provider"
                or dependency_contract.kind != current_contract.kind
            ):
                return self._dependency_error("INCOMPATIBLE_DEPENDENCY")
            for replay_step in self._receipt_replay_steps[action_id]:
                canonical = canonical_policy_json(replay_step.to_dict())
                if canonical not in seen:
                    seen.add(canonical)
                    replay.append(replay_step)

        if current_contract is not None and current_contract.role == "consumer":
            candidates = tuple(
                tool_name
                for tool_name in sorted(self._receipt_ids_by_tool)
                if (action_id := self._latest_receipt_id(tool_name, require_valid=True)) is not None
                and (dependency := self._resource_contract(self._receipt_step_objects[action_id])) is not None
                and dependency.role == "provider"
                and dependency.kind == current_contract.kind
            )
            if candidates and not set(candidates).intersection(dependency_tools):
                return self.policy_contract.public_payload(
                    {
                        "accepted": False,
                        "error": {"code": "DEPENDENCY_REQUIRED"},
                        "dependency_tool_names": list(candidates),
                    }
                )
        return tuple(replay), tuple(dependency_ids)

    async def apply(self, raw_step: Any, *, depends_on: Any = None) -> dict[str, Any]:
        async with self._lock:
            if self._pending_id is not None:
                return self.policy_contract.public_payload(
                    {
                        "accepted": False,
                        "tool_name": self._qualified_name(self._receipt_step_objects[self._pending_id]),
                        "error": {"code": "PLACEMENT_VALIDATION_REQUIRED"},
                    }
                )
            if self._attempts >= self.max_actions:
                return self.policy_contract.public_payload({"accepted": False, "error": {"code": "PLACEMENT_LIMIT"}})
            try:
                self.security_policy.preflight_plan({"steps": [raw_step]})
            except PolicyInputLimitError:
                return self.policy_contract.public_payload({"accepted": False, "error": {"code": "INVALID_ACTION"}})
            validated = validate_attack_step(raw_step, self.validation_context)
            if not validated.valid or validated.step is None:
                return self.policy_contract.public_payload({"accepted": False, "error": {"code": "INVALID_ACTION"}})
            if validated.step.type != "environment":
                return self.policy_contract.public_payload(
                    {"accepted": False, "error": {"code": "PLACEMENT_NOT_APPLICABLE"}}
                )
            resolved = self._resolve_dependencies(depends_on, validated.step)
            if isinstance(resolved, dict):
                return resolved
            dependency_steps, dependency_ids = resolved
            self._attempts += 1
            index = self._attempts
            try:
                workspace = materialize_attempt_dir(
                    source_task_dir=self.source_task_dir,
                    episode_root=self.episode_root,
                    attempt_index=index,
                    steps=(*dependency_steps, validated.step),
                    source_manifest=self.source_manifest,
                    candidate_validator=self.candidate_validator,
                    placement_target_index=len(dependency_steps),
                )
                result = await self.runner.run(workspace)
            except Exception:
                result = PlacementRunResult(False)

            # Resource identity is adapter-owned.  Never reconstruct locators
            # in policy-eval: after the fresh-sandbox replay, compare the
            # canonical locators independently proven by the two placement
            # adapters.  This works for any domain using the metadata protocol.
            if result.available and result.valid and dependency_ids:
                dependency_locators = {
                    self._receipts[action_id].locator for action_id in dependency_ids
                }
                if dependency_locators != {result.locator}:
                    return self._dependency_error("DEPENDENCY_RESOURCE_MISMATCH")
            if not result.available:
                return self.policy_contract.public_payload(
                    {"accepted": False, "error": {"code": "EVALUATION_UNAVAILABLE"}}
                )
            action_id = f"act_{secrets.token_urlsafe(24)}"
            tool_name = self._qualified_name(validated.step)
            self._receipts[action_id] = result
            self._receipt_steps[action_id] = canonical_policy_json(validated.step.to_dict())
            self._receipt_step_objects[action_id] = validated.step
            self._receipt_replay_steps[action_id] = (*dependency_steps, validated.step)
            self._receipt_ids_by_tool.setdefault(tool_name, []).append(action_id)
            self._pending_id = action_id
            return self.policy_contract.public_payload(
                {
                    "accepted": True,
                    "tool_name": tool_name,
                    "applied": result.applied,
                    "placement_ready": True,
                    "dependency_count": len(dependency_ids),
                }
            )

    def validate(self, tool_name: Any) -> dict[str, Any]:
        if not isinstance(tool_name, str) or not tool_name or len(tool_name) > 256:
            return self.policy_contract.public_payload(
                {"found": False, "error": {"code": "NO_PENDING_PLACEMENT"}}
            )
        action_id = self._pending_id
        if action_id is not None:
            pending_tool = self._qualified_name(self._receipt_step_objects[action_id])
            if pending_tool != tool_name:
                return self.policy_contract.public_payload(
                    {
                        "found": False,
                        "tool_name": pending_tool,
                        "error": {"code": "TOOL_MISMATCH"},
                    }
                )
            self._validated_ids.add(action_id)
            self._pending_id = None
        else:
            action_id = self._latest_receipt_id(tool_name, require_valid=False)
            if action_id is None:
                return self.policy_contract.public_payload(
                    {"found": False, "error": {"code": "NO_PENDING_PLACEMENT"}}
                )
        result = self._receipts[action_id]
        self._validated_ids.add(action_id)
        payload: dict[str, Any] = {
            "found": True,
            "tool_name": tool_name,
            "valid": result.valid,
            "status": result.status,
        }
        if result.valid:
            payload["validated_placement_locator"] = result.locator
        else:
            payload["requested_placement_locator"] = result.locator
            payload["error"] = {"code": result.code}
            payload["diagnostic"] = {
                "locator_fields": list(result.locator_fields),
                "retryable": result.retryable,
            }
            if result.retryable:
                payload["repair"] = {
                    "fields": list(result.repair_fields),
                    "instruction": "change only the listed fields, then apply the revised action once",
                }
        return self.policy_contract.public_payload(payload)

    def unverified_environment_indices(self, steps: Any) -> tuple[int, ...]:
        """Return submitted environment steps lacking an owned valid read-back.

        A receipt can authorize at most one matching step. This prevents a
        policy from applying one action and duplicating it in the final plan.
        Only receipts explicitly read through validate_placement participate.
        """
        available = [
            self._receipt_steps[action_id] for action_id in self._validated_ids if self._receipts[action_id].valid
        ]
        missing: list[int] = []
        for index, step in enumerate(steps):
            if getattr(step, "type", None) != "environment":
                continue
            canonical = canonical_policy_json(step.to_dict())
            try:
                match = available.index(canonical)
            except ValueError:
                missing.append(index)
            else:
                available.pop(match)
        return tuple(missing)

    @property
    def applied_actions(self) -> int:
        return len(self._receipts)

    @property
    def validated_actions(self) -> int:
        return len(self._validated_ids)

    @property
    def verified_actions(self) -> int:
        return sum(result.valid for result in self._receipts.values())

    @property
    def encountered_unsupported(self) -> bool:
        return any(result.status == "unsupported" for result in self._receipts.values())

    def training_summary(self) -> dict[str, Any]:
        """Return bounded receipt facts without action ids, payloads, or locators."""

        statuses: dict[str, int] = {}
        for result in self._receipts.values():
            statuses[result.status] = statuses.get(result.status, 0) + 1
        return {
            "applied_actions": self.applied_actions,
            "validated_actions": self.validated_actions,
            "verified_actions": self.verified_actions,
            "statuses": statuses,
        }
