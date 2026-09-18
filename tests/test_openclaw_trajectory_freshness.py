from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.openclaw.src.agent import OpenClawAgent


@pytest.mark.asyncio
async def test_session_log_freshness_barrier_rejects_previous_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = object.__new__(OpenClawAgent)
    agent._session_id = "session-1"
    agent._profile_dir = str(tmp_path / "profile")
    agent._runtime_trace_dir = str(tmp_path / "runtime")
    agent._last_cli_payload = None
    runtime = Path(agent._runtime_trace_dir)
    runtime.mkdir()
    trace = runtime / "session-1.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    old_ns = trace.stat().st_mtime_ns
    barrier_ns = old_ns + 1_000_000_000

    monkeypatch.setattr(
        "agent.openclaw.src.agent.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=""),
    )
    monkeypatch.setattr(agent, "_build_openclaw_command", lambda *args: ["true"])
    monkeypatch.setattr(agent, "_build_openclaw_env", lambda: {})

    assert await agent._get_session_log_path(
        wait_seconds=0,
        warn=False,
        min_mtime_ns=barrier_ns,
    ) is None

    os.utime(trace, ns=(barrier_ns, barrier_ns))
    assert await agent._get_session_log_path(
        wait_seconds=0,
        warn=False,
        min_mtime_ns=barrier_ns,
    ) == str(trace)


@pytest.mark.asyncio
async def test_trajectory_batch_finalizes_once_at_latest_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(OpenClawAgent)
    agent._defer_trajectory_export = False
    agent._session_started_at = None
    agent._last_invocation_started_ns = 12345
    calls: list[tuple[float, int | None]] = []

    async def generate(duration: float, *, min_mtime_ns: int | None = None) -> None:
        calls.append((duration, min_mtime_ns))

    monkeypatch.setattr(agent, "_generate_trajectory", generate)
    monkeypatch.setattr(agent, "get_result", lambda: "result")

    agent.begin_trajectory_batch()
    assert agent._defer_trajectory_export is True
    assert await agent.finalize_trajectory() == "result"
    assert len(calls) == 1
    assert calls[0][1] == 12345
    assert agent._defer_trajectory_export is False
