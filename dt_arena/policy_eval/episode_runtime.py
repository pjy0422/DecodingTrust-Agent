"""Pure H-victim-run state machine for one M3 policy trajectory.

The historical public names use ``submission`` for wire compatibility. H is
charged only when DTAP reports that the victim evaluation started; validation
failures and other ``INVALID_SUBMISSION`` receipts do not change it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EpisodeStatus(str, Enum):
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    EXHAUSTED = "exhausted"
    POLICY_LIMIT = "policy_limit"
    INFRA_ERROR = "infra_error"
    SECURITY_ABORT = "security_abort"


class EpisodeTerminalError(RuntimeError):
    """Raised when a trusted caller tries to mutate a terminal episode."""


@dataclass
class EpisodeRuntimeState:
    max_submissions: int
    max_submit_calls: int | None = None
    submissions_used: int = 0
    submit_calls: int = 0
    status: EpisodeStatus = EpisodeStatus.ACTIVE
    successful_attempt: int | None = None
    infrastructure_stage: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_submissions, bool)
            or not isinstance(self.max_submissions, int)
            or self.max_submissions < 1
        ):
            raise ValueError("max_submissions must be a positive integer")
        if self.max_submit_calls is not None:
            if (
                isinstance(self.max_submit_calls, bool)
                or not isinstance(self.max_submit_calls, int)
                or self.max_submit_calls < self.max_submissions
            ):
                raise ValueError("max_submit_calls must be an integer >= max_submissions")

    @property
    def remaining_submissions(self) -> int:
        """Compatibility name for the remaining victim-run budget."""

        return self.remaining_victim_runs

    @property
    def max_victim_runs(self) -> int:
        """Canonical meaning of the legacy ``max_submissions`` field (H)."""

        return self.max_submissions

    @property
    def victim_runs_started(self) -> int:
        """Number of victim evaluations that actually crossed the start gate."""

        return self.submissions_used

    @property
    def remaining_victim_runs(self) -> int:
        return max(0, self.max_victim_runs - self.victim_runs_started)

    @property
    def remaining_submit_calls(self) -> int | None:
        if self.max_submit_calls is None:
            return None
        return max(0, self.max_submit_calls - self.submit_calls)

    @property
    def terminal(self) -> bool:
        return self.status is not EpisodeStatus.ACTIVE

    @property
    def remove_sample(self) -> bool:
        return self.status in {EpisodeStatus.INFRA_ERROR, EpisodeStatus.SECURITY_ABORT}

    @property
    def final_reward(self) -> float | None:
        if self.status is EpisodeStatus.SUCCEEDED:
            return 1.0
        if self.status in {EpisodeStatus.EXHAUSTED, EpisodeStatus.POLICY_LIMIT}:
            return 0.0
        return None

    def _require_active(self) -> None:
        if self.terminal:
            raise EpisodeTerminalError("episode is terminal")

    def mark_victim_run_started(self) -> int:
        """Consume one H slot after DTAP confirms victim evaluation start."""

        self._require_active()
        if self.victim_runs_started >= self.max_victim_runs:
            self.status = EpisodeStatus.EXHAUSTED
            raise EpisodeTerminalError("victim-run budget is exhausted")
        self.submissions_used += 1
        return self.victim_runs_started

    def mark_evaluation_started(self) -> int:
        """Compatibility alias for :meth:`mark_victim_run_started`."""

        return self.mark_victim_run_started()

    def begin_submit_call(self) -> int:
        """Consume one Q slot. Legacy M3 runtimes have no Q budget."""

        self._require_active()
        if self.max_submit_calls is not None and self.submit_calls >= self.max_submit_calls:
            self.status = EpisodeStatus.POLICY_LIMIT
            raise EpisodeTerminalError("submit call budget is exhausted")
        self.submit_calls += 1
        return self.submit_calls

    def enforce_submit_call_limit(self) -> None:
        if (
            self.status is EpisodeStatus.ACTIVE
            and self.max_submit_calls is not None
            and self.submit_calls >= self.max_submit_calls
        ):
            self.status = EpisodeStatus.POLICY_LIMIT

    def record_attack_result(self, *, attempt_index: int, attack_success: bool) -> None:
        self._require_active()
        if attempt_index != self.victim_runs_started or attempt_index < 1:
            raise ValueError("attempt_index is not the current started evaluation")
        if not isinstance(attack_success, bool):
            raise TypeError("attack_success must be a bool")
        if attack_success:
            self.status = EpisodeStatus.SUCCEEDED
            self.successful_attempt = attempt_index
        elif self.victim_runs_started >= self.max_victim_runs:
            self.status = EpisodeStatus.EXHAUSTED

    def record_infrastructure_failure(self, *, stage: str) -> None:
        self._require_active()
        self.status = EpisodeStatus.INFRA_ERROR
        self.infrastructure_stage = str(stage)

    def record_security_failure(self, *, stage: str) -> None:
        self._require_active()
        self.status = EpisodeStatus.SECURITY_ABORT
        self.infrastructure_stage = str(stage)
