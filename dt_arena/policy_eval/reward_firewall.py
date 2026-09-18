"""Bounded judge-result parsing for the M4 reward boundary."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .security_policy import EvaluationSecurityPolicy, PolicyInputLimitError, enforce_json_complexity


class VerdictError(ValueError):
    """A verdict could not be trusted. Raw details stay host-side."""


@dataclass(frozen=True)
class TrustedVerdict:
    attack_success: bool


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerdictError("duplicate judge-result key")
        result[key] = value
    return result


class JudgeVerdictReader:
    def __init__(self, policy: EvaluationSecurityPolicy, *, expected_uid: int | None = None) -> None:
        self.policy = policy
        self.expected_uid = os.getuid() if expected_uid is None else expected_uid

    def read(self, path: Path | str, *, result_root: Path | str) -> TrustedVerdict:
        candidate = Path(path)
        root = Path(result_root).resolve()
        if candidate.is_symlink():
            raise VerdictError("judge result may not be a symlink")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise VerdictError("judge result escapes result root")
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise VerdictError("judge result is unavailable") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_size > self.policy.max_judge_bytes:
            raise VerdictError("judge result is not a bounded regular file")
        if info.st_uid != self.expected_uid:
            raise VerdictError("judge result has unexpected ownership")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(candidate, flags)
            try:
                data = os.read(fd, self.policy.max_judge_bytes + 1)
                if os.read(fd, 1):
                    data += b"x"
            finally:
                os.close(fd)
        except OSError as exc:
            raise VerdictError("judge result could not be read safely") from exc
        if len(data) > self.policy.max_judge_bytes:
            raise VerdictError("judge result exceeds byte limit")
        try:
            value = json.loads(data, object_pairs_hook=_reject_duplicates)
            enforce_json_complexity(
                value,
                max_depth=self.policy.max_judge_depth,
                max_nodes=self.policy.max_judge_nodes,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, PolicyInputLimitError) as exc:
            raise VerdictError("judge result is malformed") from exc
        if not isinstance(value, dict) or not isinstance(value.get("attack_success"), bool):
            raise VerdictError("judge result has no trustworthy attack verdict")
        return TrustedVerdict(attack_success=value["attack_success"])


def find_single_judge_result(result_root: Path | str) -> Path:
    root = Path(result_root)
    if root.is_symlink() or not root.is_dir():
        raise VerdictError("result root is invalid")
    matches: list[Path] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            if (current_path / name).is_symlink():
                raise VerdictError("result tree contains a symlink")
        if "judge_result.json" in files:
            candidate = current_path / "judge_result.json"
            if candidate.is_symlink():
                raise VerdictError("judge result may not be a symlink")
            matches.append(candidate)
            if len(matches) > 1:
                break
    if len(matches) != 1:
        raise VerdictError("expected exactly one judge result")
    return matches[0]
