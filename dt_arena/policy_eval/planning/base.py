"""Planning-strategy boundary layered above a reusable policy harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..harness import HarnessRequest, HarnessResult, PolicyHarness


@dataclass(frozen=True)
class PlanningContext:
    placement_enabled: bool
    max_submissions: int
    feedback_mode: Any
    improvement_wishes: bool = False
    dying_message: bool = False


class PlanningStrategy(ABC):
    """Choose policy reasoning behavior without owning runtime transport.

    Most strategies only need to override ``build_prompt``. Strategies that need
    explicit multi-phase execution may override ``execute`` and invoke the same
    ``PolicyHarness`` one or more times.
    """

    name: str

    @abstractmethod
    def build_prompt(self, context: PlanningContext) -> str:
        raise NotImplementedError

    async def execute(
        self,
        harness: PolicyHarness,
        request: HarnessRequest,
        context: PlanningContext,
    ) -> HarnessResult:
        return await harness.run(request)
