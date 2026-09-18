"""Claude Code implementation of the reusable policy harness."""

from __future__ import annotations

import asyncio
from pathlib import Path

from .base import HarnessRequest, HarnessResult


class ClaudeCodeHarness:
    """Run exactly one Claude Code policy session.

    Episode semantics, planning style, MCP authority, victim execution and judge
    behavior intentionally live outside this class. This is only the process /
    transport boundary, so every planning strategy can reuse it unchanged.
    """

    def __init__(self, binary: str | Path = "claude") -> None:
        self.binary = str(binary)

    def command(self, request: HarnessRequest) -> tuple[str, ...]:
        return (
            self.binary,
            "-p",
            request.prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(request.max_turns),
            "--mcp-config",
            str(request.mcp_config),
            "--strict-mcp-config",
            "--settings",
            str(request.settings),
            "--allowedTools",
            ",".join(sorted(request.allowed_tools)),
            "--disallowedTools",
            ",".join(sorted(request.disallowed_tools)),
            "--model",
            request.model,
        )

    async def run(self, request: HarnessRequest) -> HarnessResult:
        command = self.command(request)
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=request.cwd,
            env=dict(request.env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        raw_out, raw_err = await asyncio.wait_for(
            process.communicate(), timeout=request.timeout_seconds
        )
        return HarnessResult(
            returncode=int(process.returncode or 0),
            stdout=raw_out.decode(errors="replace"),
            stderr=raw_err.decode(errors="replace"),
            command=command,
        )
