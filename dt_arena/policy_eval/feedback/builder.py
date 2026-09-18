"""Failure-isolated orchestration of M7 adaptive feedback."""

from __future__ import annotations

import asyncio
import json
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..actions import ValidatedAttackStep
from .deterministic import ParsedMCPTrace, extract_deterministic_feedback, parse_mcp_events
from .digestor import Digestor, DigestorObservation, ReasoningSummarizer, validate_repair_digest
from .schema import FeedbackMode, ReasoningSummaryConfig, deterministic_to_dict
from .victim_trace import build_victim_trace, extract_final_response, load_trajectory, sanitize_trace_value


@dataclass(frozen=True)
class FeedbackBuildLimits:
    max_mcp_bytes: int = 4 * 1024 * 1024
    max_trajectory_bytes: int = 8 * 1024 * 1024
    max_final_chars: int = 16_000
    digest_timeout_seconds: float = 30.0


class FeedbackBuilder:
    def __init__(
        self,
        *,
        mode: FeedbackMode = FeedbackMode.DISABLED,
        digestor: Digestor | None = None,
        reasoning_summarizer: ReasoningSummarizer | None = None,
        reasoning: ReasoningSummaryConfig | None = None,
        limits: FeedbackBuildLimits | None = None,
    ) -> None:
        reasoning = reasoning or ReasoningSummaryConfig()
        limits = limits or FeedbackBuildLimits()
        if reasoning.enabled and mode is not FeedbackMode.FINAL_DETERMINISTIC_DIGESTOR:
            raise ValueError("reasoning summary is available only in Digestor mode")
        if mode is FeedbackMode.FINAL_DETERMINISTIC_DIGESTOR and digestor is None:
            raise ValueError("Digestor mode requires a digestor")
        if reasoning.enabled and reasoning_summarizer is None:
            raise ValueError("enabled reasoning summary requires a summarizer")
        self.mode = mode
        self.digestor = digestor
        self.reasoning_summarizer = reasoning_summarizer
        self.reasoning = reasoning
        self.limits = limits

    @staticmethod
    def _single_regular_artifact(root: Path, name: str, max_bytes: int) -> Path | None:
        try:
            root = root.resolve(strict=True)
        except OSError:
            return None
        matches: list[Path] = []
        for candidate in root.rglob(name):
            try:
                info = candidate.lstat()
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if stat.S_ISREG(info.st_mode) and not candidate.is_symlink() and info.st_size <= max_bytes:
                matches.append(candidate)
        return matches[0] if len(matches) == 1 else None

    @classmethod
    def _single_mcp_artifact(cls, root: Path, max_bytes: int) -> Path | None:
        canonical = cls._single_regular_artifact(root, "victim-mcp-events.jsonl", max_bytes)
        if canonical is not None:
            return canonical
        return cls._single_regular_artifact(root, "*.mcp-events.jsonl", max_bytes)

    @classmethod
    def _single_trajectory_artifact(cls, root: Path, max_bytes: int) -> Path | None:
        canonical = cls._single_regular_artifact(root, "victim-trajectory.json", max_bytes)
        if canonical is not None:
            return canonical
        try:
            resolved_root = root.resolve(strict=True)
        except OSError:
            return None
        matches: list[Path] = []
        for candidate in resolved_root.rglob("*.json"):
            try:
                info = candidate.lstat()
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(resolved_root)
                if not stat.S_ISREG(info.st_mode) or candidate.is_symlink() or info.st_size > max_bytes:
                    continue
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError, UnicodeError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("trajectory"), list):
                matches.append(candidate)
        return matches[0] if len(matches) == 1 else None

    async def build(
        self,
        *,
        attempt_root: Path,
        submitted_steps: Sequence[ValidatedAttackStep],
        submitted_plan: Mapping[str, Any],
        redactions: tuple[str, ...] = (),
    ) -> dict[str, Any] | None:
        if self.mode is FeedbackMode.DISABLED:
            return None
        trajectory_path = self._single_trajectory_artifact(attempt_root, self.limits.max_trajectory_bytes)
        mcp_path = self._single_mcp_artifact(attempt_root, self.limits.max_mcp_bytes)
        trajectory: Mapping[str, Any] = {}
        if trajectory_path is not None:
            try:
                trajectory = load_trajectory(trajectory_path, max_bytes=self.limits.max_trajectory_bytes)
            except Exception:
                trajectory = {}
        mcp = ParsedMCPTrace((), {}, False)
        if mcp_path is not None:
            try:
                mcp = parse_mcp_events(mcp_path, max_bytes=self.limits.max_mcp_bytes)
            except Exception:
                mcp = ParsedMCPTrace((), {}, False)
        deterministic = extract_deterministic_feedback(submitted_steps, mcp)
        effective_redactions = (*redactions, str(attempt_root.resolve()))
        final_response = sanitize_trace_value(
            extract_final_response(trajectory, max_chars=self.limits.max_final_chars),
            redactions=effective_redactions,
            max_chars=self.limits.max_final_chars,
        )
        assert isinstance(final_response, str)
        result: dict[str, Any] = {"schema_version": 2, "final_response": final_response}
        if self.mode is FeedbackMode.FINAL_ONLY:
            return result
        result["deterministic"] = deterministic_to_dict(deterministic)
        if self.mode is not FeedbackMode.FINAL_DETERMINISTIC_DIGESTOR:
            return result

        trace = build_victim_trace(
            trajectory,
            mcp,
            include_reasoning=self.reasoning.enabled,
            redactions=effective_redactions,
        )
        assert self.digestor is not None
        digest_submission = sanitize_trace_value(
            submitted_plan,
            redactions=effective_redactions,
            max_chars=self.limits.max_final_chars,
        )
        assert isinstance(digest_submission, Mapping)
        observation = DigestorObservation(2, digest_submission, deterministic, trace)
        try:
            raw_digest = await asyncio.wait_for(
                self.digestor.digest(observation),
                timeout=self.limits.digest_timeout_seconds,
            )
            result["digest"] = validate_repair_digest(
                raw_digest, submitted_plan, deterministic=deterministic
            ).to_dict()
        except Exception:
            # Optional-analysis timeout/failure cannot alter an already-recorded attempt.
            pass
        if self.reasoning.enabled:
            assert self.reasoning_summarizer is not None
            source = trace.reasoning_source
            if source == "disabled":
                source = "unavailable"
            if source == "unavailable":
                result["reasoning_summary"] = {
                    "source": "unavailable",
                    "summary": "",
                }
                return result
            try:
                summary = await asyncio.wait_for(
                    self.reasoning_summarizer.summarize(trace),
                    timeout=self.reasoning.timeout_seconds,
                )
                if isinstance(summary, str) and summary.strip():
                    result["reasoning_summary"] = {
                        "source": source,
                        "summary": summary[: self.reasoning.max_chars],
                    }
            except Exception:
                pass
        return result
