"""DTAP policy-evaluation research runtime.

This package owns the current Claude Code E2E control path, policy-scoped MCP
surface, placement receipts, bounded submission semantics, adaptive feedback,
and artifact production.  Public names describe behavior rather than historical
implementation milestones.
"""

from .attempt_runner import AttemptResult, DtapAttemptRunner
from .authority import EpisodeAuthority, EpisodeAuthorityRegistry, EpisodeCredentials
from .episode import load_task_snapshot
from .episode_runtime import EpisodeRuntimeState, EpisodeStatus
from .harness import ClaudeCodeHarness, HarnessRequest, HarnessResult, PolicyHarness
from .mcp_server import create_core_policy_mcp_server, create_policy_mcp_server
from .planning import (
    AuthorityHierarchyInversionV2PlanningStrategy,
    CurrentPlanningStrategy,
    PlanningContext,
    PlanningStrategy,
    create_planning_strategy,
)
from .placement import DtapPlacementRunner, PlacementCoordinator
from .policy_contract import PolicyContract, PolicyLeakageGuard
from .security_policy import EvaluationSecurityPolicy
from .submission import SubmissionCoordinator

__all__ = [
    "AttemptResult",
    "AuthorityHierarchyInversionV2PlanningStrategy",
    "ClaudeCodeHarness",
    "CurrentPlanningStrategy",
    "DtapAttemptRunner",
    "DtapPlacementRunner",
    "EpisodeAuthority",
    "EpisodeAuthorityRegistry",
    "EpisodeCredentials",
    "EpisodeRuntimeState",
    "EpisodeStatus",
    "EvaluationSecurityPolicy",
    "HarnessRequest",
    "HarnessResult",
    "PlacementCoordinator",
    "PlanningContext",
    "PlanningStrategy",
    "PolicyContract",
    "PolicyHarness",
    "PolicyLeakageGuard",
    "SubmissionCoordinator",
    "create_core_policy_mcp_server",
    "create_planning_strategy",
    "create_policy_mcp_server",
    "load_task_snapshot",
]
