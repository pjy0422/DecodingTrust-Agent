"""Stable artifact contract shared by the E2E runner and trajectory viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ARTIFACT_SCHEMA = "dtap-agent-rl-episode"  # v1 compatibility identifier
ARTIFACT_SCHEMA_VERSION = 1

POLICY_TRAJECTORY = "policy.jsonl"
POLICY_PROMPT = "policy-prompt.txt"
ORIGINAL_CONFIG = "original-config.yaml"
SUBMITTED_CONFIG = "submitted-config.yaml"
VICTIM_TRAJECTORY = "victim-trajectory.json"
VICTIM_MCP_EVENTS = "victim-mcp-events.jsonl"
JUDGE_RESULT = "judge-result.json"
JUDGE_VERDICT = "judge-verdict.json"
EPISODE_MANIFEST = "episode-manifest.json"
EPISODE_STATE = "episode-state.json"
RESULT = "result.json"
RESEARCH_FEEDBACK = "research-feedback.json"


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path

    @property
    def attempts(self) -> Path:
        return self.root / "attempts"

    def attempt(self, index: int) -> Path:
        if isinstance(index, bool) or index < 1:
            raise ValueError("attempt index must be a positive integer")
        return self.attempts / f"attempt-{index:04d}"

    def path(self, name: str) -> Path:
        return self.root / name
