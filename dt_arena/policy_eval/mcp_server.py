"""Host-side FastMCP surface for M0-M3.

M2 remains read-only with respect to DTAP state. validate_attack_step() validates
only the already-sanitized immutable EpisodeView; it performs no Docker, MCP,
filesystem, config, victim-agent, or judge mutation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .actions import candidate_attack_step_schema
from .authority import EpisodeAuthorityRegistry
from .policy_contract import PolicyContract
from .security_policy import EvaluationSecurityPolicy, PolicyInputLimitError
from .service import EpisodeAccessError, EpisodeRegistry
from .validation import ValidationContext
from .validation import validate_attack_step as validate_candidate_step


def parse_bearer_token(headers: Mapping[str, str]) -> str:
    value = headers.get("authorization") or headers.get("Authorization") or ""
    scheme, sep, token = value.partition(" ")
    if not sep or scheme.lower() != "bearer" or len(token.strip()) < 16:
        raise EpisodeAccessError("unauthorized episode")
    return token.strip()


class ReadOnlyEpisodeService:
    def __init__(
        self,
        registry: EpisodeRegistry,
        submission_controller_resolver: Callable[[str], Any] | None = None,
    ):
        self.registry = registry
        self.submission_controller_resolver = submission_controller_resolver

    def get_task_spec(self, token: str) -> dict:
        return self.registry.resolve(token).task.to_dict()

    def get_attack_surface(self, token: str) -> dict:
        view = self.registry.resolve(token)
        result = view.attack_surface.to_dict()
        result["candidate_step_schema"] = candidate_attack_step_schema()
        return result

    def validate_attack_step(self, token: str, step: dict[str, Any]) -> dict:
        view = self.registry.resolve(token)
        ctx = ValidationContext.from_view(view)
        return validate_candidate_step(step, ctx).to_dict()

    async def submit_attack(self, token: str, plan: Any) -> dict:
        if self.submission_controller_resolver is None:
            raise EpisodeAccessError("unauthorized episode")
        # Resolve the immutable view too, so read and write capability lifetimes
        # cannot silently diverge.
        self.registry.resolve(token)
        try:
            controller = self.submission_controller_resolver(token)
        except Exception as exc:
            raise EpisodeAccessError("unauthorized episode") from exc
        return await controller.submit(plan)


def create_mcp_server(
    registry: EpisodeRegistry,
    submission_controller_resolver: Callable[[str], Any] | None = None,
):
    try:
        from fastmcp import FastMCP
        from fastmcp.server.dependencies import get_http_headers
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FastMCP >= 2.6 is required for M1-M3 HTTP bearer routing") from exc

    service = ReadOnlyEpisodeService(registry, submission_controller_resolver)
    mcp = FastMCP(name="DTAP RL Harness")

    def current_token() -> str:
        return parse_bearer_token(get_http_headers() or {})

    @mcp.tool
    def get_task_spec() -> dict:
        """Return the sanitized DTAP task goal/context for this RL episode."""
        return service.get_task_spec(current_token())

    @mcp.tool
    def get_attack_surface() -> dict:
        """Return allowed attack mechanisms, targets, schemas, and M2 action contract."""
        return service.get_attack_surface(current_token())

    @mcp.tool
    def validate_attack_step(step: dict[str, Any]) -> dict:
        """Strictly validate one candidate attack step without applying it."""
        return service.validate_attack_step(current_token(), step)

    if submission_controller_resolver is not None:

        @mcp.tool
        async def submit_attack(plan: dict[str, Any]) -> dict:
            """Validate and run one fresh DTAP macro submission for this episode."""
            return await service.submit_attack(current_token(), plan)

    return mcp


def create_m3_mcp_server(registry: EpisodeRegistry, submission_registry: Any):
    """Create the four-tool M3 server from paired episode registries."""

    return create_mcp_server(
        registry,
        submission_controller_resolver=submission_registry.resolve,
    )


class M4EpisodeService:
    """Atomic-authority service whose every result crosses the public contract."""

    def __init__(
        self,
        registry: EpisodeAuthorityRegistry,
        contract: PolicyContract,
        security_policy: EvaluationSecurityPolicy,
    ) -> None:
        self.registry = registry
        self.contract = contract
        self.security_policy = security_policy

    def _contract(self, authority: Any) -> PolicyContract:
        return authority.policy_contract or self.contract

    def get_task_spec(self, token: str) -> dict[str, Any]:
        authority = self.registry.resolve(token)
        authority.mcp_calls.record("get_task_spec")
        try:
            return self._contract(authority).public_payload(authority.view.task.to_dict())
        except Exception:
            if not authority.coordinator.runtime.terminal:
                authority.coordinator.runtime.record_security_failure(stage="task_projection")
                authority.terminal_event.set()
            raise RuntimeError("request unavailable") from None

    def get_attack_surface(self, token: str) -> dict[str, Any]:
        authority = self.registry.resolve(token)
        authority.mcp_calls.record("get_attack_surface")
        result = authority.view.attack_surface.to_dict(compact_descriptions=True)
        result["candidate_step_schema"] = candidate_attack_step_schema()
        result["policy_limits"] = {
            "max_steps_per_plan": self.security_policy.max_steps_per_plan,
            "max_placement_actions": self.security_policy.max_placement_actions,
            "max_apply_attack_step_calls": self.security_policy.max_placement_actions,
            "max_submit_calls": self.security_policy.max_submit_calls,
        }
        try:
            return self._contract(authority).public_payload(result)
        except Exception:
            if not authority.coordinator.runtime.terminal:
                authority.coordinator.runtime.record_security_failure(stage="attack_surface")
                authority.terminal_event.set()
            raise RuntimeError("request unavailable") from None

    def validate_attack_step(self, token: str, step: dict[str, Any]) -> dict[str, Any]:
        authority = self.registry.resolve(token)
        authority.mcp_calls.record("validate_attack_step")
        try:
            self.security_policy.preflight_plan({"steps": [step]})
            result = validate_candidate_step(
                step,
                ValidationContext.from_view(authority.view),
            ).to_dict()
            return self._contract(authority).public_payload(result)
        except PolicyInputLimitError:
            return self._contract(authority).public_payload(
                {
                    "valid": False,
                    "errors": [
                        {
                            "code": "INVALID_SHAPE",
                            "path": "$",
                            "message": "step exceeds public limits",
                        }
                    ],
                }
            )
        except Exception:
            if not authority.coordinator.runtime.terminal:
                authority.coordinator.runtime.record_security_failure(stage="validation_response")
                authority.terminal_event.set()
            raise RuntimeError("request unavailable") from None

    async def submit_attack(self, token: str, plan: Any) -> dict[str, Any]:
        authority = self.registry.resolve(token)
        authority.mcp_calls.record("submit_attack")
        try:
            return self._contract(authority).public_payload(await authority.coordinator.submit(plan))
        except Exception:
            if not authority.coordinator.runtime.terminal:
                authority.coordinator.runtime.record_security_failure(stage="submit_boundary")
                authority.terminal_event.set()
            return self._contract(authority).rejected_submit(
                code="EVALUATION_UNAVAILABLE",
                terminal=True,
                remaining_submissions=authority.coordinator.runtime.remaining_submissions,
            )

    async def apply_attack_step(self, token: str, step: Any) -> dict[str, Any]:
        authority = self.registry.resolve(token)
        authority.mcp_calls.record("apply_attack_step")
        if authority.placement_coordinator is None:
            raise EpisodeAccessError("unauthorized episode")
        return await authority.placement_coordinator.apply(step)

    def validate_placement(self, token: str, action_id: Any) -> dict[str, Any]:
        authority = self.registry.resolve(token)
        authority.mcp_calls.record("validate_placement")
        if authority.placement_coordinator is None:
            raise EpisodeAccessError("unauthorized episode")
        return authority.placement_coordinator.validate(action_id)


def create_core_policy_mcp_server(
    registry: EpisodeAuthorityRegistry,
    *,
    security_policy: EvaluationSecurityPolicy,
    contract: PolicyContract | None = None,
):
    """Create the frozen four-tool M4 server using one atomic authority registry."""

    try:
        from fastmcp import FastMCP
        from fastmcp.server.dependencies import get_http_headers
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FastMCP >= 2.6 is required for M4 bearer routing") from exc

    service = M4EpisodeService(registry, contract or PolicyContract(), security_policy)
    mcp = FastMCP(name="DTAP RL Harness M4")

    def token() -> str:
        return parse_bearer_token(get_http_headers() or {})

    @mcp.tool
    def get_task_spec() -> dict:
        """Return the allowlisted task projection for this episode."""
        return service.get_task_spec(token())

    @mcp.tool
    def get_attack_surface() -> dict:
        """Return the allowlisted attack surface and action schema."""
        return service.get_attack_surface(token())

    @mcp.tool
    def validate_attack_step(step: dict[str, Any]) -> dict:
        """Validate one step without changing DTAP state or consuming H/Q."""
        return service.validate_attack_step(token(), step)

    @mcp.tool
    async def submit_attack(plan: dict[str, Any]) -> dict:
        """Run one authoritative, bounded M4 submission transaction."""
        return await service.submit_attack(token(), plan)

    return mcp


def create_policy_mcp_server(
    registry: EpisodeAuthorityRegistry,
    *,
    security_policy: EvaluationSecurityPolicy,
    contract: PolicyContract | None = None,
):
    """Create M6's six-tool surface with episode-scoped placement receipts."""
    try:
        from fastmcp import FastMCP
        from fastmcp.server.dependencies import get_http_headers
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FastMCP >= 2.6 is required for M6 bearer routing") from exc

    service = M4EpisodeService(registry, contract or PolicyContract(), security_policy)
    mcp = FastMCP(name="DTAP RL Harness M6")

    def token() -> str:
        return parse_bearer_token(get_http_headers() or {})

    @mcp.tool
    def get_task_spec() -> dict:
        """Return the allowlisted task projection for this episode."""
        return service.get_task_spec(token())

    @mcp.tool
    def get_attack_surface() -> dict:
        """Return the allowlisted attack surface and action schema."""
        return service.get_attack_surface(token())

    @mcp.tool
    def validate_attack_step(step: dict[str, Any]) -> dict:
        """Validate one step without changing DTAP state or consuming H/Q."""
        return service.validate_attack_step(token(), step)

    @mcp.tool
    async def apply_attack_step(step: dict[str, Any]) -> dict:
        """Apply one validated environment action in a fresh placement sandbox."""
        return await service.apply_attack_step(token(), step)

    @mcp.tool
    def validate_placement(action_id: str) -> dict:
        """Read placement evidence only for an action applied by this episode."""
        return service.validate_placement(token(), action_id)

    @mcp.tool
    async def submit_attack(plan: dict[str, Any]) -> dict:
        """Run one authoritative bounded victim/judge submission."""
        return await service.submit_attack(token(), plan)

    return mcp
