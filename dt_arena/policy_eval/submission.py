"""Authoritative M3 submit transaction and policy-safe receipts."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .attempt_runner import AttemptResult
from .audit import AuditEvent
from .candidate_config import CandidateConfigError, materialize_attempt_dir
from .episode_runtime import EpisodeRuntimeState, EpisodeStatus, EpisodeTerminalError
from .integrity import BenchmarkManifest, IntegrityError
from .policy_contract import PolicyContract, PolicyContractViolation
from .security_policy import EvaluationSecurityPolicy, PolicyInputLimitError
from .validation import ValidationContext, validate_attack_plan


class SubmissionErrorCode(str, Enum):
    EMPTY_PLAN = "EMPTY_PLAN"
    YAML_SCHEMA_MISMATCH = "YAML_SCHEMA_MISMATCH"
    EPISODE_TERMINAL = "EPISODE_TERMINAL"
    INFRA_ERROR = "INFRA_ERROR"


_SAFE_AUDIT_CODES = {
    "INVALID_SUBMISSION",
    "POLICY_LIMIT",
    "EPISODE_TERMINAL",
    "EVALUATION_UNAVAILABLE",
    "benchmark_integrity",
    "policy_contract",
    "materialization",
    "runner",
    "evaluation_start",
    "timeout",
    "process_start",
    "judge_result",
    "judge_verdict",
    "scheduler",
    "cancelled",
    "succeeded",
    "exhausted",
    "policy_limit",
    "infra_error",
    "security_abort",
}


class SubmissionPlanError(ValueError):
    def __init__(self, code: str, message: str, path: str):
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.path = path


@dataclass(frozen=True)
class SubmissionPlan:
    steps: tuple[Any, ...]

    @classmethod
    def from_policy(cls, raw: Any) -> SubmissionPlan:
        if not isinstance(raw, Mapping):
            raise SubmissionPlanError("INVALID_SHAPE", "plan must be an object", "$")
        unknown = set(raw) - {"steps"}
        if unknown:
            field = sorted(unknown)[0]
            raise SubmissionPlanError("UNKNOWN_FIELD", "unknown field", field)
        if "steps" not in raw:
            raise SubmissionPlanError("MISSING_FIELD", "required field is missing", "steps")
        if not isinstance(raw["steps"], list):
            raise SubmissionPlanError("INVALID_SHAPE", "steps must be an array", "steps")
        if not raw["steps"]:
            raise SubmissionPlanError(
                SubmissionErrorCode.EMPTY_PLAN.value,
                "steps must not be empty",
                "steps",
            )
        return cls(tuple(copy.deepcopy(raw["steps"])))


class SubmissionCoordinator:
    def __init__(
        self,
        *,
        validation_context: ValidationContext,
        runtime: EpisodeRuntimeState,
        source_task_dir: Path | str,
        episode_root: Path | str,
        runner: Any,
        candidate_validator: Callable[..., Any] | None = None,
        security_policy: EvaluationSecurityPolicy | None = None,
        policy_contract: PolicyContract | None = None,
        source_manifest: BenchmarkManifest | None = None,
        terminal_event: asyncio.Event | None = None,
        audit_sink: Any = None,
        audit_episode_digest: str | None = None,
        placement_coordinator: Any = None,
        feedback_builder: Any = None,
    ) -> None:
        self.validation_context = validation_context
        self.runtime = runtime
        self.source_task_dir = Path(source_task_dir)
        self.episode_root = Path(episode_root)
        self.runner = runner
        self.candidate_validator = candidate_validator
        self.security_policy = security_policy
        self.policy_contract = policy_contract
        self.source_manifest = source_manifest
        self.terminal_event = terminal_event
        self.audit_sink = audit_sink
        self.audit_episode_digest = audit_episode_digest
        self.placement_coordinator = placement_coordinator
        self.feedback_builder = feedback_builder
        if feedback_builder is not None and policy_contract is None:
            raise ValueError("adaptive feedback requires a policy contract")
        if security_policy is not None:
            if runtime.max_submit_calls is None:
                raise ValueError("M4 runtime requires an explicit submit-call budget")
            if runtime.max_submit_calls != security_policy.max_submit_calls:
                raise ValueError("runtime and security-policy submit budgets differ")
        self._submission_lock = asyncio.Lock()
        self._training_attempts: list[dict[str, Any]] = []

    def training_summary(self) -> dict[str, Any]:
        """Trusted, content-free outcome summary for the training record."""

        return {
            "attempts": copy.deepcopy(self._training_attempts),
            "victim_runs_started": self.runtime.victim_runs_started,
            "submit_calls": self.runtime.submit_calls,
        }

    def _state_fields(self) -> dict[str, Any]:
        return {
            "terminal": self.runtime.terminal,
            # Legacy key: this is the count of victim evaluations that started,
            # never the count of calls to submit_attack.
            "submissions_used": self.runtime.victim_runs_started,
            "remaining_submissions": self.runtime.remaining_submissions,
        }

    def _reject(self, code: str, message: str, *, path: str | None = None) -> dict[str, Any]:
        error: dict[str, str] = {"code": code}
        if path is not None:
            error["path"] = path
        error["message"] = message
        return {"accepted": False, **self._state_fields(), "errors": [error]}

    def _terminal_rejection(self) -> dict[str, Any]:
        return self._reject("EPISODE_TERMINAL", "episode is terminal")

    def _signal_terminal(self) -> None:
        if self.runtime.terminal and self.terminal_event is not None:
            self.terminal_event.set()

    def _audit(self, event: str, code: str | None = None) -> None:
        if self.audit_sink is None or self.audit_episode_digest is None:
            return
        safe_code = code if code is None or code in _SAFE_AUDIT_CODES else "other"
        self.audit_sink.emit(
            AuditEvent(
                event,
                self.audit_episode_digest,
                {"h": self.runtime.victim_runs_started, "q": self.runtime.submit_calls},
                safe_code,
            )
        )

    def _policy_reject(self, code: str) -> dict[str, Any]:
        assert self.policy_contract is not None
        self._signal_terminal()
        if code == "EVALUATION_UNAVAILABLE":
            event = "security_abort" if self.runtime.status is EpisodeStatus.SECURITY_ABORT else "infra_abort"
            self._audit(event, self.runtime.infrastructure_stage)
        else:
            self._audit("submit_rejected", code)
        return self.policy_contract.rejected_submit(
            code=code,
            terminal=self.runtime.terminal,
            remaining_submissions=self.runtime.remaining_submissions,
        )

    def _policy_security_abort(self) -> dict[str, Any]:
        if not self.runtime.terminal:
            self.runtime.record_security_failure(stage="policy_contract")
        self._signal_terminal()
        self._audit("security_abort", "policy_contract")
        return {
            "accepted": False,
            "terminal": True,
            "remaining_submissions": self.runtime.remaining_submissions,
            "error": {"code": "EVALUATION_UNAVAILABLE"},
        }

    def _finish_policy_rejection(self) -> None:
        if self.security_policy is not None:
            self.runtime.enforce_submit_call_limit()
            self._signal_terminal()

    async def submit(self, plan: Any) -> dict[str, Any]:
        async with self._submission_lock:
            if self.runtime.terminal:
                if self.policy_contract is not None:
                    code = "POLICY_LIMIT" if self.runtime.status is EpisodeStatus.POLICY_LIMIT else "EPISODE_TERMINAL"
                    return self._policy_reject(code)
                return self._terminal_rejection()
            if self.security_policy is not None:
                try:
                    self.runtime.begin_submit_call()
                    self.security_policy.preflight_plan(plan)
                except EpisodeTerminalError:
                    return self._policy_reject("POLICY_LIMIT")
                except PolicyInputLimitError:
                    self._finish_policy_rejection()
                    return self._policy_reject("POLICY_LIMIT" if self.runtime.terminal else "INVALID_SUBMISSION")
            try:
                parsed_plan = SubmissionPlan.from_policy(plan)
            except SubmissionPlanError as exc:
                if self.policy_contract is not None:
                    self._finish_policy_rejection()
                    return self._policy_reject("POLICY_LIMIT" if self.runtime.terminal else "INVALID_SUBMISSION")
                return self._reject(exc.code, exc.safe_message, path=exc.path)

            validated = validate_attack_plan(parsed_plan.steps, self.validation_context)
            if not validated.valid:
                if self.policy_contract is not None:
                    self._finish_policy_rejection()
                    return self._policy_reject("POLICY_LIMIT" if self.runtime.terminal else "INVALID_SUBMISSION")
                return {
                    "accepted": False,
                    **self._state_fields(),
                    "errors": [error.to_dict() for error in validated.errors],
                }

            if self.placement_coordinator is not None and self.placement_coordinator.unverified_environment_indices(
                validated.steps
            ):
                # M6 final plans may contain only environment actions that this
                # episode applied and then positively read back. The generic
                # response intentionally reveals no environment oracle data.
                self._finish_policy_rejection()
                return self._policy_reject("POLICY_LIMIT" if self.runtime.terminal else "INVALID_SUBMISSION")

            attempt_index = self.runtime.victim_runs_started + 1
            try:
                workspace = materialize_attempt_dir(
                    source_task_dir=self.source_task_dir,
                    episode_root=self.episode_root,
                    attempt_index=attempt_index,
                    steps=validated.steps,
                    candidate_validator=self.candidate_validator,
                    source_manifest=self.source_manifest,
                )
            except IntegrityError:
                self.runtime.record_security_failure(stage="benchmark_integrity")
                self._signal_terminal()
                if self.policy_contract is not None:
                    return self._policy_reject("EVALUATION_UNAVAILABLE")
                return self._reject("INFRA_ERROR", "evaluation unavailable")
            except CandidateConfigError as exc:
                if isinstance(exc.__cause__, IntegrityError):
                    self.runtime.record_security_failure(stage="benchmark_integrity")
                    self._signal_terminal()
                    if self.policy_contract is not None:
                        return self._policy_reject("EVALUATION_UNAVAILABLE")
                    return self._reject("INFRA_ERROR", "evaluation unavailable")
                if self.policy_contract is not None:
                    self._finish_policy_rejection()
                    return self._policy_reject("POLICY_LIMIT" if self.runtime.terminal else "INVALID_SUBMISSION")
                return self._reject("YAML_SCHEMA_MISMATCH", "candidate config failed validation")
            except Exception:
                self.runtime.record_infrastructure_failure(stage="materialization")
                if self.policy_contract is not None:
                    return self._policy_reject("EVALUATION_UNAVAILABLE")
                return self._reject("INFRA_ERROR", "evaluation unavailable")

            try:
                result: AttemptResult = await self.runner.run(workspace)
            except asyncio.CancelledError:
                if self.security_policy is not None and not self.runtime.terminal:
                    self.runtime.record_infrastructure_failure(stage="cancelled")
                    self._signal_terminal()
                raise
            except Exception:
                self.runtime.record_infrastructure_failure(stage="runner")
                if self.policy_contract is not None:
                    return self._policy_reject("EVALUATION_UNAVAILABLE")
                return self._reject("INFRA_ERROR", "evaluation unavailable")

            started_index: int | None = None
            if result.evaluation_started:
                # The sole H-consuming transition. All INVALID_SUBMISSION paths
                # return before the runner, and pre-start failures consume no H.
                started_index = self.runtime.mark_victim_run_started()
                self._audit("evaluation_started")

            if result.is_infrastructure_failure or started_index is None:
                self._training_attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "submitted_config_sha256": hashlib.sha256(workspace.config_path.read_bytes()).hexdigest(),
                        "evaluation_started": bool(result.evaluation_started),
                        "attack_success": None,
                        "task_success": None,
                        "infrastructure_stage": result.infrastructure_stage or "evaluation_start",
                    }
                )
                self.runtime.record_infrastructure_failure(stage=result.infrastructure_stage or "evaluation_start")
                if self.policy_contract is not None:
                    return self._policy_reject("EVALUATION_UNAVAILABLE")
                return self._reject("INFRA_ERROR", "evaluation unavailable")

            assert isinstance(result.attack_success, bool)
            self._training_attempts.append(
                {
                    "attempt_index": started_index,
                    "submitted_config_sha256": hashlib.sha256(workspace.config_path.read_bytes()).hexdigest(),
                    "evaluation_started": True,
                    "attack_success": result.attack_success,
                    "task_success": result.task_success if isinstance(result.task_success, bool) else None,
                    "infrastructure_stage": None,
                }
            )
            self.runtime.record_attack_result(
                attempt_index=started_index,
                attack_success=result.attack_success,
            )
            self.runtime.enforce_submit_call_limit()
            self._signal_terminal()
            self._audit("verdict_accepted")
            if self.runtime.terminal:
                self._audit("episode_terminal", self.runtime.status.value)
            receipt = {
                "accepted": True,
                "submission": started_index,
                "success": result.attack_success,
                "terminal": self.runtime.terminal,
                "remaining_submissions": self.runtime.remaining_submissions,
            }
            if self.feedback_builder is not None:
                try:
                    feedback = await self.feedback_builder.build(
                        attempt_root=workspace.output_root,
                        submitted_steps=validated.steps,
                        submitted_plan={"steps": [step.to_dict() for step in validated.steps]},
                        trajectory_path=result.trajectory_path,
                        mcp_path=result.mcp_events_path,
                        # Artifact identity was fixed by DtapAttemptRunner.  Do
                        # not independently re-discover another same-named file.
                        discover_artifacts=False,
                        redactions=(
                            str(self.source_task_dir.resolve()),
                            str(self.episode_root.resolve()),
                            *(self.policy_contract.guard.secrets if self.policy_contract is not None else ()),
                        ),
                    )
                    if feedback is not None:
                        receipt["feedback"] = feedback
                except Exception:
                    # Optional feedback never changes the already committed H/reward state.
                    pass
            if self.policy_contract is None:
                return receipt
            try:
                return self.policy_contract.from_internal_submit(receipt)
            except PolicyContractViolation:
                # A malformed/leaking optional projection is dropped. The base
                # verdict receipt remains authoritative and policy-safe.
                if "feedback" in receipt:
                    receipt.pop("feedback", None)
                    try:
                        return self.policy_contract.from_internal_submit(receipt)
                    except PolicyContractViolation:
                        pass
                return self._policy_security_abort()


class EpisodeSubmissionRegistry:
    """Thread-safe token-to-coordinator map parallel to the immutable view registry."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._controllers: dict[str, SubmissionCoordinator] = {}

    def register(self, token: str, controller: SubmissionCoordinator) -> None:
        if not isinstance(token, str) or len(token.strip()) < 16:
            raise PermissionError("unauthorized episode")
        with self._lock:
            if token in self._controllers:
                raise ValueError("episode token already registered")
            self._controllers[token] = controller

    def resolve(self, token: str) -> SubmissionCoordinator:
        with self._lock:
            controller = self._controllers.get(token.strip() if isinstance(token, str) else "")
        if controller is None:
            raise PermissionError("unauthorized episode")
        return controller

    def unregister(self, token: str) -> None:
        with self._lock:
            self._controllers.pop(token, None)


@contextmanager
def registered_submission(
    registry: EpisodeSubmissionRegistry,
    *,
    token: str,
    controller: SubmissionCoordinator,
):
    registry.register(token, controller)
    try:
        yield
    finally:
        registry.unregister(token)
