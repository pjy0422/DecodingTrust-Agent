from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .episode import TaskSnapshot


_LEGACY_GUEST_DIRECT_COMPOSE_PATHS = frozenset(
    {
        "dt_arena/envs/macos/docker-compose.yml",
        "dt_arena/envs/windows/docker-compose.yml",
    }
)


@dataclass(frozen=True)
class ToolSpec:
    server_name: str
    tool_name: str
    qualified_name: str
    description: str | None
    input_schema: dict[str, Any]
    # Internal trust label supplied by the DTAP placement registry. It is not
    # part of the policy-visible tool contract and therefore is removed by
    # ``to_dict`` below. ``None`` fails closed for environment tools.
    placement_capability: str | None = None

    def to_dict(self, *, compact_description: bool = False) -> dict[str, Any]:
        result = asdict(self)
        result.pop("placement_capability")
        if compact_description and self.description:
            # Claude Code persists oversized MCP results to a sandbox file. The
            # policy intentionally has no file tools, so retain the semantic
            # summary and exact schema while dropping repetitive Args/Returns
            # prose from the M4/M6 public response.
            paragraph = self.description.strip().split("\n\n", 1)[0]
            result["description"] = " ".join(paragraph.split())[:320]
        return result


@dataclass(frozen=True)
class AttackSurface:
    prompt_enabled: bool
    tool_enabled: bool
    environment_enabled: bool
    skill_enabled: bool
    skill_modes: tuple[str, ...]
    victim_tools: tuple[ToolSpec, ...]
    environment_tools: tuple[ToolSpec, ...]
    # M2 additions are defaulted to preserve M0/M1 construction compatibility.
    prompt_modes: tuple[str, ...] = ()
    tool_modes: tuple[str, ...] = ()
    skill_targets: tuple[str, ...] = ()
    # Public whole-plan rules that cannot be discovered by validating one step.
    # These contain no benchmark example or trusted path.
    plan_constraints: tuple[str, ...] = ()

    def to_dict(self, *, compact_descriptions: bool = False) -> dict[str, Any]:
        return {
            "injections": {
                "prompt": self.prompt_enabled,
                "tool": self.tool_enabled,
                "environment": self.environment_enabled,
                "skill": self.skill_enabled,
                "prompt_modes": list(self.prompt_modes),
                "tool_modes": list(self.tool_modes),
                "skill_modes": list(self.skill_modes),
            },
            "victim_tools": [tool.to_dict(compact_description=compact_descriptions) for tool in self.victim_tools],
            "environment_tools": [
                tool.to_dict(compact_description=compact_descriptions) for tool in self.environment_tools
            ],
            # Names only; trusted filesystem paths are intentionally not exposed.
            "skill_targets": list(self.skill_targets),
            "plan_constraints": list(self.plan_constraints),
        }


class ToolCatalogProvider(Protocol):
    async def list_victim_tools(self, snapshot: TaskSnapshot) -> Mapping[str, Sequence[ToolSpec]]: ...

    async def list_environment_tools(self, snapshot: TaskSnapshot) -> Mapping[str, Sequence[ToolSpec]]: ...


def _iter_raw_agent_servers(agent: Mapping[str, Any]):
    for server in agent.get("mcp_servers") or []:
        if isinstance(server, Mapping):
            yield server
    for subagent in agent.get("sub_agents") or []:
        if isinstance(subagent, Mapping):
            yield from _iter_raw_agent_servers(subagent)


def _victim_server_rules(snapshot: TaskSnapshot) -> dict[str, set[str]]:
    rules: dict[str, set[str]] = {}
    raw_agent = snapshot.raw_config.get("Agent") or {}
    if not isinstance(raw_agent, Mapping):
        return rules

    for server in _iter_raw_agent_servers(raw_agent):
        if not server.get("enabled", True):
            continue
        name = str(server.get("name") or "").strip()
        if not name:
            continue
        blacklist = {str(item) for item in (server.get("tool_blacklist") or []) if item is not None}
        rules[name] = blacklist
    return rules


def _environment_server_rules(snapshot: TaskSnapshot) -> dict[str, str | set[str]]:
    raw = snapshot.injection_config.get("environment_servers") or {}
    if not isinstance(raw, Mapping):
        return {}

    rules: dict[str, str | set[str]] = {}
    for server_name, allowed in raw.items():
        name = str(server_name)
        if allowed == "all":
            rules[name] = "all"
        elif isinstance(allowed, (list, tuple, set)):
            rules[name] = {str(tool) for tool in allowed}
        elif isinstance(allowed, Mapping) and "tools" in allowed:
            tools = allowed.get("tools")
            rules[name] = "all" if tools == "all" else {str(t) for t in (tools or [])}
        else:
            rules[name] = set()
    return rules


def _iter_agent_configs(agent_cfg: Any):
    yield agent_cfg
    for child in getattr(agent_cfg, "sub_agents", None) or []:
        yield from _iter_agent_configs(child)


def _discover_skill_targets(snapshot: TaskSnapshot) -> tuple[str, ...]:
    """Discover skill identifiers without exposing or serializing trusted paths."""
    names: set[str] = set()
    for agent_cfg in _iter_agent_configs(snapshot.agent_config):
        for value in getattr(agent_cfg, "skill_directories", None) or []:
            path = Path(value)
            # DTAP configurations may point directly at a skill dir or at a root.
            if (path / "SKILL.md").is_file():
                names.add(path.name)
            if path.is_dir():
                try:
                    for child in path.iterdir():
                        if child.is_dir() and (child / "SKILL.md").is_file():
                            names.add(child.name)
                except OSError:
                    # Missing/unreadable trusted skill roots simply expose no target;
                    # M3 execution would not be able to use them either.
                    pass
    return tuple(sorted(names))


def _dedupe_sorted(tools: list[ToolSpec]) -> tuple[ToolSpec, ...]:
    by_name = {tool.qualified_name: tool for tool in tools}
    return tuple(by_name[key] for key in sorted(by_name))


def _is_verified_environment_tool(tool: ToolSpec) -> bool:
    """Fail closed unless DTAP explicitly provides placement verification."""
    return tool.placement_capability == "verified"


async def build_attack_surface(
    snapshot: TaskSnapshot,
    provider: ToolCatalogProvider,
) -> AttackSurface:
    """Discover the task contract without consulting Attack.attack_turns/examples."""
    cfg = snapshot.injection_config

    victim_tools: list[ToolSpec] = []
    if bool(cfg.get("tool_enabled", False)):
        catalog = await provider.list_victim_tools(snapshot)
        for server_name, blacklist in _victim_server_rules(snapshot).items():
            for tool in catalog.get(server_name, ()):
                if tool.tool_name not in blacklist:
                    victim_tools.append(tool)

    environment_tools: list[ToolSpec] = []
    if bool(cfg.get("environment_enabled", False)):
        catalog = await provider.list_environment_tools(snapshot)
        for server_name, allowed in _environment_server_rules(snapshot).items():
            for tool in catalog.get(server_name, ()):
                if (allowed == "all" or tool.tool_name in allowed) and _is_verified_environment_tool(tool):
                    environment_tools.append(tool)

    threat_model = getattr(snapshot.attack_config, "threat_model", None)
    # The Windows/macOS direct datasets predate RedTeamingAgent and therefore
    # have no available_injections block. Direct DTAP semantics still require a
    # jailbreak prompt. Infer that public contract from the threat model and
    # guest compose path, never from the private Attack.attack_turns example.
    compose_path = (
        str((snapshot.raw_config.get("Environment") or {}).get("docker_compose_path", ""))
        .replace("\\", "/")
        .strip("/")
        .lower()
    )
    legacy_guest_direct = (
        threat_model == "direct"
        and "RedTeamingAgent" not in snapshot.raw_config
        and compose_path in _LEGACY_GUEST_DIRECT_COMPOSE_PATHS
    )
    prompt_enabled = bool(cfg.get("prompt_enabled", False)) or legacy_guest_direct
    tool_enabled = bool(cfg.get("tool_enabled", False))
    skill_enabled = bool(cfg.get("skill_enabled", False))
    prompt_modes: tuple[str, ...] = ()
    if prompt_enabled:
        prompt_modes = ("jailbreak",) if threat_model == "direct" else ("suffix", "override")

    return AttackSurface(
        prompt_enabled=prompt_enabled,
        tool_enabled=tool_enabled,
        environment_enabled=bool(cfg.get("environment_enabled", False)),
        skill_enabled=skill_enabled,
        skill_modes=tuple(str(x) for x in (cfg.get("skill_modes") or [])),
        victim_tools=_dedupe_sorted(victim_tools),
        environment_tools=_dedupe_sorted(environment_tools),
        prompt_modes=prompt_modes,
        tool_modes=("suffix", "override") if tool_enabled else (),
        skill_targets=_discover_skill_targets(snapshot) if skill_enabled else (),
        plan_constraints=(
            (
                "direct plans require at least one jailbreak prompt",
                "direct jailbreak prompt turn_ids must be contiguous from 1",
                "environment turn_id must not exceed the direct jailbreak prompt count",
            )
            if threat_model == "direct"
            else ()
        ),
    )
