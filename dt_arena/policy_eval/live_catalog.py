from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping, Sequence
from typing import Any

from .attack_surface import ToolSpec
from .episode import TaskSnapshot


def _iter_agent_configs(agent_cfg: Any):
    yield agent_cfg
    for subagent in getattr(agent_cfg, "sub_agents", None) or []:
        yield from _iter_agent_configs(subagent)


def _environment_placement_capability(server_name: str, tool_name: str) -> str:
    try:
        from dt_arena.src.env_verification import placement_capability
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("DTAP environment catalog requires the slime M6 placement overlay") from exc
    capability = placement_capability(server_name, tool_name)
    return str(getattr(capability, "value", capability))


async def _list_url_tools(
    server_name: str,
    url: str,
    *,
    classify_environment: bool = False,
) -> list[ToolSpec]:
    from fastmcp import Client

    async with Client(url) as client:
        tools = await client.list_tools()

    result: list[ToolSpec] = []
    for tool in tools:
        input_schema = getattr(tool, "input_schema", None)
        if input_schema is None:
            input_schema = getattr(tool, "inputSchema", None)
        if input_schema is None:
            input_schema = {"type": "object", "properties": {}}
        name = str(tool.name)
        result.append(
            ToolSpec(
                server_name=server_name,
                tool_name=name,
                qualified_name=f"{server_name}:{name}",
                description=getattr(tool, "description", None),
                input_schema=dict(input_schema),
                placement_capability=(
                    _environment_placement_capability(server_name, name) if classify_environment else None
                ),
            )
        )
    return result


class LiveDtapCatalogProvider:
    def __init__(self, *, task_runtime_id: str):
        self.task_runtime_id = task_runtime_id

    async def list_victim_tools(self, snapshot: TaskSnapshot) -> Mapping[str, Sequence[ToolSpec]]:
        from utils.mcp_helpers import start_task_mcp_servers
        from utils.resource_manager import ResourceManager

        manager = None
        try:
            manager = start_task_mcp_servers(
                snapshot.agent_config,
                self.task_runtime_id,
                snapshot.task_dir,
                ResourceManager.instance(),
            )
            if manager is None:
                return {}
            result: dict[str, Sequence[ToolSpec]] = {}
            for agent_cfg in _iter_agent_configs(snapshot.agent_config):
                for server in getattr(agent_cfg, "mcp_servers", None) or []:
                    if not getattr(server, "enabled", True):
                        continue
                    url = getattr(server, "url", None)
                    if url:
                        result[server.name] = await _list_url_tools(server.name, url)
            return result
        finally:
            if manager is not None:
                manager.stop_all()

    async def list_environment_tools(self, snapshot: TaskSnapshot) -> Mapping[str, Sequence[ToolSpec]]:
        from utils.injection_mcp_helpers import start_injection_mcp_servers, wait_for_injection_mcp_ready
        from utils.resource_manager import ResourceManager

        if not snapshot.injection_config.get("environment_servers"):
            return {}

        last_error: Exception | None = None
        # DTAP's process manager checks server liveness after a short fixed
        # startup delay. Under a parallel domain run a healthy server can
        # occasionally miss that window. Retry discovery with a newly
        # allocated port, but never project an empty allowlist after a failed
        # startup: that would make the policy guess private target names.
        for attempt in range(3):
            manager = None
            cfg = copy.deepcopy(snapshot.injection_config)
            try:
                manager, updated = start_injection_mcp_servers(
                    cfg,
                    resource_manager=ResourceManager.instance(),
                    task_id=f"{self.task_runtime_id}-catalog-{attempt + 1}",
                )
                if manager is None:
                    raise RuntimeError("DTAP environment catalog server failed to start")
                wait_for_injection_mcp_ready(updated)
                result: dict[str, Sequence[ToolSpec]] = {}
                for server_name, server_info in (updated.get("environment_servers") or {}).items():
                    if isinstance(server_info, dict) and server_info.get("url"):
                        result[server_name] = await _list_url_tools(
                            server_name,
                            server_info["url"],
                            classify_environment=True,
                        )
                return result
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.25 * (attempt + 1))
            finally:
                if manager is not None:
                    manager.stop_all()
        raise RuntimeError("DTAP environment tool discovery failed after 3 attempts") from last_error
