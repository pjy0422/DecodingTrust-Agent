"""Planning strategies selectable independently from the policy harness."""

from __future__ import annotations

from .authority_inversion_v2 import AuthorityHierarchyInversionV2PlanningStrategy
from .base import PlanningContext, PlanningStrategy
from .current import CurrentPlanningStrategy

_STRATEGIES: dict[str, type[PlanningStrategy]] = {
    CurrentPlanningStrategy.name: CurrentPlanningStrategy,
    AuthorityHierarchyInversionV2PlanningStrategy.name: AuthorityHierarchyInversionV2PlanningStrategy,
}


def planning_strategy_names() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def create_planning_strategy(name: str) -> PlanningStrategy:
    try:
        strategy_type = _STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown planning strategy: {name}") from exc
    return strategy_type()


__all__ = [
    "AuthorityHierarchyInversionV2PlanningStrategy",
    "CurrentPlanningStrategy",
    "PlanningContext",
    "PlanningStrategy",
    "create_planning_strategy",
    "planning_strategy_names",
]
