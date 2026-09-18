"""Explicit M4 limits and minimum child-process environment."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class PolicyInputLimitError(ValueError):
    """An authenticated policy request exceeded a public resource limit."""


_FORBIDDEN_CHILD_ENV_NAMES = frozenset(
    {
        "DTAP_EPISODE_TOKEN",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CONFIG_DIR",
        "SLIME_AGENT_CC_EXTRA_ARGS",
        "SLIME_AGENT_CC_EXTRA_ENVS",
    }
)


def enforce_json_complexity(value: Any, *, max_depth: int, max_nodes: int) -> None:
    stack = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            raise PolicyInputLimitError("structured value exceeds configured limits")
        if isinstance(current, Mapping):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise PolicyInputLimitError("object keys must be strings")
                stack.append((item, depth + 1))
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            stack.extend((item, depth + 1) for item in current)
        elif current is not None and not isinstance(current, (str, int, float, bool)):
            raise PolicyInputLimitError("value is not JSON-compatible")


@dataclass(frozen=True)
class EvaluationSecurityPolicy:
    """Trusted rollout policy. Q is intentionally required and has no default."""

    max_submit_calls: int
    max_plan_bytes: int = 64 * 1024
    max_steps_per_plan: int = 32
    max_content_bytes: int = 16 * 1024
    max_json_depth: int = 16
    max_json_nodes: int = 4096
    max_http_body_bytes: int = 128 * 1024
    max_http_header_bytes: int = 16 * 1024
    max_judge_bytes: int = 1024 * 1024
    max_judge_depth: int = 16
    max_judge_nodes: int = 4096
    max_parallel_attempts: int = 1
    max_queued_attempts: int = 32
    max_placement_actions: int = 8
    queue_wait_timeout_seconds: float = 300.0
    inherited_dtap_env_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        integer_fields = (
            "max_submit_calls",
            "max_plan_bytes",
            "max_steps_per_plan",
            "max_content_bytes",
            "max_json_depth",
            "max_json_nodes",
            "max_http_body_bytes",
            "max_http_header_bytes",
            "max_judge_bytes",
            "max_judge_depth",
            "max_judge_nodes",
            "max_parallel_attempts",
            "max_queued_attempts",
            "max_placement_actions",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.queue_wait_timeout_seconds <= 0:
            raise ValueError("queue_wait_timeout_seconds must be positive")
        if len(set(self.inherited_dtap_env_names)) != len(self.inherited_dtap_env_names):
            raise ValueError("inherited DTAP environment names must be unique")
        for name in self.inherited_dtap_env_names:
            if not isinstance(name, str) or not name or "=" in name:
                raise ValueError("invalid inherited DTAP environment name")
            if name in _FORBIDDEN_CHILD_ENV_NAMES:
                raise ValueError("policy/adapter credential may not enter DTAP child")

    def preflight_plan(self, raw: Any) -> None:
        enforce_json_complexity(
            raw,
            max_depth=self.max_json_depth,
            max_nodes=self.max_json_nodes,
        )
        try:
            encoded = json.dumps(
                raw,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise PolicyInputLimitError("plan is not canonical JSON") from exc
        if len(encoded) > self.max_plan_bytes:
            raise PolicyInputLimitError("plan exceeds byte limit")
        if isinstance(raw, Mapping):
            steps = raw.get("steps")
            if isinstance(steps, list) and len(steps) > self.max_steps_per_plan:
                raise PolicyInputLimitError("plan exceeds step limit")
            if isinstance(steps, list):
                for step in steps:
                    if isinstance(step, Mapping) and "content" in step:
                        content = step.get("content")
                        if isinstance(content, str) and len(content.encode("utf-8")) > self.max_content_bytes:
                            raise PolicyInputLimitError("step content exceeds byte limit")

    def build_dtap_child_env(
        self,
        *,
        host_env: Mapping[str, str] | None = None,
        explicit_env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        source = os.environ if host_env is None else host_env
        result = {
            "PATH": source.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": source.get("LANG", "C.UTF-8"),
            "LC_ALL": source.get("LC_ALL", "C.UTF-8"),
            "PYTHONUNBUFFERED": "1",
        }
        for name in self.inherited_dtap_env_names:
            value = source.get(name)
            if value is not None:
                result[name] = value
        for name, value in dict(explicit_env or {}).items():
            if not isinstance(name, str) or not name or "=" in name:
                raise ValueError("invalid explicit child environment name")
            if not isinstance(value, str):
                raise ValueError("child environment values must be strings")
            if name in _FORBIDDEN_CHILD_ENV_NAMES:
                raise ValueError("policy/adapter credential may not enter DTAP child")
            result[name] = value
        return result
