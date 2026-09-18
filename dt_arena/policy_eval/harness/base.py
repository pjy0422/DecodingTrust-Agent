"""Policy-runtime transport boundary independent of planning strategy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol


@dataclass(frozen=True)
class HarnessRequest:
    prompt: str
    cwd: Path
    env: Mapping[str, str]
    model: str
    max_turns: int
    mcp_config: Path
    settings: Path
    allowed_tools: frozenset[str]
    disallowed_tools: frozenset[str]
    timeout_seconds: float


@dataclass(frozen=True)
class HarnessResult:
    returncode: int
    stdout: str
    stderr: str
    command: tuple[str, ...]


class PolicyHarness(Protocol):
    async def run(self, request: HarnessRequest) -> HarnessResult: ...
