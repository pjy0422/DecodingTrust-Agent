from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DtapApi:
    TaskConfig: Any
    AttackConfig: Any
    AgentConfig: Any
    parse_injection_config: Callable[[dict[str, Any]], dict[str, Any]]


def load_dtap_api() -> DtapApi:
    try:
        from dt_arena.src.types.agent import AgentConfig
        from dt_arena.src.types.task import AttackConfig, TaskConfig

        from utils.injection_mcp_helpers import parse_injection_config
    except ImportError as exc:
        raise RuntimeError(
            "DTAP is not importable. Add the DecodingTrust-Agent repository root "
            "to PYTHONPATH (or install it) before loading a real DTAP task."
        ) from exc

    return DtapApi(
        TaskConfig=TaskConfig,
        AttackConfig=AttackConfig,
        AgentConfig=AgentConfig,
        parse_injection_config=parse_injection_config,
    )
