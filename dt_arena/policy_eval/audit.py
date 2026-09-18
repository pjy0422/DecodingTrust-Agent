"""Content-free trusted audit events for M4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_EVENTS = {
    "episode_registered",
    "submit_rejected",
    "evaluation_started",
    "verdict_accepted",
    "episode_terminal",
    "infra_abort",
    "security_abort",
    "cleanup_completed",
}


@dataclass(frozen=True)
class AuditEvent:
    event: str
    episode_digest: str
    counters: Mapping[str, int]
    code: str | None = None

    def __post_init__(self) -> None:
        if self.event not in _EVENTS:
            raise ValueError("unknown audit event")
        if not self.episode_digest or len(self.episode_digest) > 64:
            raise ValueError("invalid episode audit digest")
        if any(not isinstance(value, int) or value < 0 for value in self.counters.values()):
            raise ValueError("audit counters must be non-negative integers")


class InMemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        self.events.append(event)
