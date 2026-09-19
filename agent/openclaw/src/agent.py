import os
import sys
import json
import asyncio
import uuid
import subprocess
import shutil
import time
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import aiohttp

# Add parent directory to path to import types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from dt_arena.src.types.agent import Agent, AgentConfig, RuntimeConfig, MCPServerConfig, AgentResult
from dt_arena.src.types.trajectory import Trajectory
from dt_arena.src.feedback_observation import (
    build_environment_feedback_probe,
    build_static_feedback_probes,
)

from .mcp_proxy import MCPProxyManager, _tool_timeout_seconds
from .utils import OpenClawTrajectoryConverter
from .plugin_generator import StaticPluginGenerator, MCPServerTools, MCPTool
from .helpers.auth_helpers import populate_openclaw_profile_auth

from utils.skill_helpers import (
    create_injected_skills_directory,
    cleanup_temp_directory,
)


# Default timeout for OpenClaw operations (increased from 120s for complex multi-tool tasks)
OPENCLAW_TIMEOUT_SECONDS: float = float(os.getenv("OPENCLAW_TIMEOUT_SECONDS", "1000"))
OPENCLAW_TRAJECTORY_CONVERSION_TIMEOUT_SECONDS: float = float(
    os.getenv("OPENCLAW_TRAJECTORY_CONVERSION_TIMEOUT_SECONDS", "15")
)

# OpenClaw config path
OPENCLAW_CONFIG_PATH = Path.home() / ".openclaw" / "openclaw.json"


def _openclaw_token_usage(payload: Any) -> Optional[Dict[str, Any]]:
    """Project OpenClaw's provider receipt without estimating missing fields."""
    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta")
    agent_meta = meta.get("agentMeta") if isinstance(meta, dict) else None
    candidates = (
        agent_meta.get("diagnosticUsage") if isinstance(agent_meta, dict) else None,
        agent_meta.get("usage") if isinstance(agent_meta, dict) else None,
        agent_meta.get("lastCallUsage") if isinstance(agent_meta, dict) else None,
        meta.get("usage") if isinstance(meta, dict) else None,
        payload.get("usage"),
    )
    usage = next((item for item in candidates if isinstance(item, dict)), None)
    if usage is None:
        return None

    def count(*names: str) -> Optional[int]:
        for name in names:
            value = usage.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0, int(value))
        return None

    input_tokens = count("input", "inputTokens", "input_tokens")
    output_tokens = count("output", "outputTokens", "output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    cache_read = count("cacheRead", "cache_read_input_tokens") or 0
    cache_write = count("cacheWrite", "cache_creation_input_tokens") or 0
    reasoning = count("reasoningTokens", "reasoning_tokens")
    prompt_tokens = (input_tokens or 0) + cache_read + cache_write
    output_tokens = output_tokens or 0
    return {
        "input_tokens": input_tokens or 0,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning,
        "reasoning_source": "provider" if reasoning is not None else "unavailable",
        "tokens_with_reasoning": prompt_tokens + output_tokens,
        "tokens_without_reasoning": (
            prompt_tokens + max(0, output_tokens - reasoning)
            if reasoning is not None else None
        ),
    }


class OpenClawAgent(Agent):
    """
    This agent integrates with the OpenClaw CLI to execute instructions and interact with MCP servers.
    """

    # Valid thinking levels for OpenClaw CLI
    VALID_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high")

    def __init__(
        self,
        agent_config: AgentConfig,
        runtime_config: Optional[RuntimeConfig] = None,
    ):
        """
        Initialize OpenClaw Agent.

        Args:
            agent_config: Agent configuration (system_prompt and mcp_servers)
            runtime_config: Runtime configuration (model, temperature, max_turns, output_dir)
        """
        super().__init__(agent_config, runtime_config)

        # Setup output directories
        output_dir = self.runtime_config.output_dir or os.path.join(os.getcwd(), "results")
        self.output_dir = output_dir
        self.traces_dir = os.path.join(self.output_dir, "traces")
        self._runtime_trace_dir = os.path.join(self.traces_dir, "openclaw_runtime")
        self.trajectories_dir = self.output_dir
        os.makedirs(self.traces_dir, exist_ok=True)
        os.makedirs(self._runtime_trace_dir, exist_ok=True)
        os.makedirs(self.trajectories_dir, exist_ok=True)

        # Initialize timestamp and trajectory converter
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Initialize trajectory converter
        self.trajectory_converter = OpenClawTrajectoryConverter(
            self.trajectories_dir,
            self.timestamp
        )

        # Session management
        self._session_id: str = f"dt-arena-{uuid.uuid4().hex[:8]}"
        self._mcp_event_log: str = os.path.join(
            self._runtime_trace_dir, f"{self._session_id}.mcp-events.jsonl"
        )
        self._feedback_probe_file = os.path.join(
            self._runtime_trace_dir, f"{self._session_id}.feedback-probes.json"
        )
        self._provider_observation_file = os.path.join(
            self._runtime_trace_dir, f"{self._session_id}.provider-feedback.jsonl"
        )
        self._conversation_history: List[Dict[str, Any]] = []

        # Unique profile name for isolation (prevents plugin conflicts in parallel execution)
        self._profile_name: str = f"task-{uuid.uuid4().hex[:8]}"
        self._profile_dir: str = os.path.expanduser(f"~/.openclaw-{self._profile_name}")
        self._profile_extensions_dir: str = os.path.join(self._profile_dir, "extensions")

        # Execution state
        self._turn_count: int = 0
        self._trace_metadata: Optional[Dict[str, Any]] = None
        self._current_trajectory: Optional[Trajectory] = None
        self._last_cli_payload: Optional[Dict[str, Any]] = None
        self._token_usage: Optional[Dict[str, Any]] = None
        self._session_started_at: Optional[datetime] = None
        self._last_invocation_started_ns: Optional[int] = None
        self._defer_trajectory_export: bool = False

        # MCP Proxy Manager
        self._proxy_manager: Optional[MCPProxyManager] = None
        self._proxy_urls: Dict[str, str] = {}

        # Temporary config file path (cleaned up on exit)
        self._openclaw_config_path: Optional[str] = None

        # Skill injection temp directory
        self._skill_temp_dir: Optional[str] = None

        # Static plugin generator for MCP tools
        self._plugin_generator: Optional[StaticPluginGenerator] = None
        self._generated_plugin_id: Optional[str] = None

        # Discovered MCP tool lists per server (for trajectory "List MCP Tools" steps)
        self._mcp_tool_lists: Dict[str, List[Dict[str, Any]]] = {}

        # Debug mode
        self._debug = runtime_config.debug if runtime_config else False

        # Thinking level
        self._thinking = self._resolve_thinking_level()

    def _get_thinking_level(self) -> str:
        """Get valid thinking level for OpenClaw CLI."""
        return self._thinking

    def _resolve_thinking_level(self) -> str:
        """
        Resolve thinking level from agent_kwargs.

        Agent kwargs can specify:
            thinking_level: "off" | "minimal" | "low" | "medium" | "high"

        Returns:
            Valid thinking level string, defaults to "medium" if not specified
        """
        agent_kwargs = self.runtime_config.agent_kwargs or {}
        thinking_level = agent_kwargs.get("thinking_level", "medium")

        if thinking_level not in self.VALID_THINKING_LEVELS:
            raise ValueError(
                f"Invalid thinking_level '{thinking_level}'. "
                f"Must be one of: {', '.join(self.VALID_THINKING_LEVELS)}"
            )
        return thinking_level

    def _load_openclaw_config(self) -> Dict[str, Any]:
        """Load current OpenClaw configuration."""
        if OPENCLAW_CONFIG_PATH.exists():
            try:
                return json.loads(OPENCLAW_CONFIG_PATH.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _build_openclaw_env(self) -> Dict[str, str]:
        """Build the environment used for all OpenClaw subprocesses."""
        env = os.environ.copy()
        node_bin = os.getenv("OPENCLAW_NODE_BIN")
        if node_bin:
            node_dir = str(Path(node_bin).parent)
            current_path = env.get("PATH", "")
            if current_path:
                path_parts = current_path.split(os.pathsep)
                if path_parts[0] != node_dir:
                    env["PATH"] = os.pathsep.join([node_dir, *path_parts])
            else:
                env["PATH"] = node_dir
        return env

    def _build_openclaw_command(self, *args: str) -> List[str]:
        """Build a consistent OpenClaw CLI command for init and execution."""
        cli_override = os.getenv("OPENCLAW_CLI_BIN")
        if cli_override:
            if cli_override.endswith(".mjs"):
                node_bin = os.getenv("OPENCLAW_NODE_BIN") or shutil.which("node")
                if not node_bin:
                    raise RuntimeError("Compatible Node.js runtime not found for OpenClaw CLI")
                return [node_bin, cli_override, *args]
            return [cli_override, *args]

        cli_bin = shutil.which("openclaw")
        if cli_bin:
            return [cli_bin, *args]
        raise FileNotFoundError("openclaw")

    def _configure_openclaw_with_proxies(self) -> None:
        """
        Configure OpenClaw with proxy-backed MCP servers for embedded runs.

        Creates a temporary OpenClaw config file with:
        1. Model from runtime_config
        2. Proxy-backed MCP servers under mcp.servers for bundle-mcp
        3. Generated static plugin enabled for non-embedded/plugin tooling
        4. Skill directories if configured

        Environment variables are set when running OpenClaw CLI:
        - OPENCLAW_CONFIG_PATH: path to temp openclaw config
        """
        # Evaluation profiles must be isolated from the operator's persistent
        # OpenClaw configuration.  Copying ~/.openclaw/openclaw.json here can
        # leak unrelated providers/plugins into a benchmark and, after an
        # OpenClaw upgrade, can make every task fail on stale config keys.
        # Materialize only the settings required by this DTAP attempt.
        openclaw_config: Dict[str, Any] = {}

        # Set model from runtime_config
        model = self.runtime_config.model
        if "agents" not in openclaw_config:
            openclaw_config["agents"] = {}
        if "defaults" not in openclaw_config["agents"]:
            openclaw_config["agents"]["defaults"] = {}
        if "model" not in openclaw_config["agents"]["defaults"]:
            openclaw_config["agents"]["defaults"]["model"] = {}

        openclaw_config["agents"]["defaults"]["model"]["primary"] = model
        # DTAP supplies the victim's complete evaluation prompt through the
        # workspace AGENTS.md below.  OpenClaw otherwise seeds BOOTSTRAP.md in
        # every fresh isolated profile and asks the victim to choose a name and
        # identity.  That onboarding text is unrelated to the benchmark and can
        # leak into the victim's final response.  The supported skipBootstrap
        # setting prevents default profile files from being created while
        # preserving injection of our existing AGENTS.md.
        openclaw_config["agents"]["defaults"]["skipBootstrap"] = True
        # A hardened slime DTAP child maps its trusted victim provider aliases
        # onto the standard Anthropic environment variables before constructing
        # the agent. OpenClaw otherwise consults only its user's persistent
        # provider catalog, which makes an unqualified evaluation model resolve
        # against unrelated local configuration. Materialize an isolated custom
        # provider instead; the key remains an environment reference and is
        # never written to the generated profile.
        dtap_anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
        if dtap_anthropic_base_url and os.environ.get("ANTHROPIC_API_KEY", "").strip():
            provider_name = "dtap-anthropic"
            model_id = model.split("/", 1)[1] if model.startswith(
                provider_name + "/"
            ) else model
            openclaw_config["agents"]["defaults"]["model"]["primary"] = (
                f"{provider_name}/{model_id}"
            )
            openclaw_config["models"] = {
                "providers": {
                    provider_name: {
                        "baseUrl": dtap_anthropic_base_url,
                        "apiKey": {
                            "source": "env",
                            "provider": "default",
                            "id": "ANTHROPIC_API_KEY",
                        },
                        "api": "anthropic-messages",
                        # Ollama's hosted Anthropic-compatible endpoint expects
                        # the shared token as Authorization: Bearer rather than
                        # Anthropic's native x-api-key header.
                        "authHeader": True,
                        "models": [
                            {
                                "id": model_id,
                                "name": model_id,
                                "reasoning": True,
                                "input": ["text", "image"],
                                "contextWindow": 200000,
                                "maxTokens": 8192,
                            }
                        ],
                    }
                }
            }
        # litellm support
        elif model.startswith("litellm/"):
            openclaw_config["models"] = {
                "providers": {
                    "litellm": {
                        "baseUrl": "${LITELLM_BASE_URL}",
                        "apiKey": "${LITELLM_API_KEY}",
                        "api": "anthropic-messages",
                        "models": [
                            {
                                "id": "claude-opus-4-6",
                                "name": "Claude Opus 4.6",
                                "reasoning": True,
                                "input": ["text", "image"],
                                "contextWindow": 200000,
                                "maxTokens": 64000,
                            }
                        ],
                    }
                }
            }
        # Llama API (Meta internal, OpenAI-compatible). Pass --model llama/<id>
        # e.g. llama/quiet_sand. Key is read from LLAMA_API_KEY at runtime.
        elif model.startswith("llama/"):
            llama_model_id = model.split("/", 1)[1]
            llama_base_url = os.environ.get("LLAMA_API_BASE_URL")
            openclaw_config["models"] = {
                "providers": {
                    "llama": {
                        "baseUrl": llama_base_url,
                        "apiKey": {
                            "source": "env",
                            "provider": "default",
                            "id": "LLAMA_API_KEY",
                        },
                        "auth": "api-key",
                        "api": "openai-completions",
                        "authHeader": True,
                        # quiet_sand and other Meta models emit large reasoning
                        # traces; raise per-request timeout above OpenClaw's
                        # default to avoid idle-timeouts under heavy tool loads.
                        "timeoutSeconds": 600,
                        "models": [
                            {
                                "id": llama_model_id,
                                "name": llama_model_id,
                                "input": ["text", "image"],
                                "contextWindow": 128000,
                                "maxTokens": 8192,
                            }
                        ],
                    }
                }
            }
        elif model.startswith("openai/"):
            # OpenClaw supports the normal OpenAI API-key route directly from the
            # process environment. Avoid generating a partial
            # models.providers.openai block here, because explicit provider config
            # is validated against the full provider schema.
            pass
        elif model.startswith("openrouter/"):
            if "auth" not in openclaw_config:
                openclaw_config["auth"] = {}
            if "profiles" not in openclaw_config["auth"]:
                openclaw_config["auth"]["profiles"] = {}
            openclaw_config["auth"]["profiles"].setdefault(
                "openrouter:default",
                {
                    "provider": "openrouter",
                    "mode": "api_key",
                },
            )

        # Create workspace directory and write system prompt to AGENTS.md
        # OpenClaw injects bootstrap files (AGENTS.md, SOUL.md, etc.) into the system prompt
        workspace_dir = os.path.join(self._profile_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)

        if self.config and self.config.system_prompt:
            agents_md_path = os.path.join(workspace_dir, "AGENTS.md")
            try:
                with open(agents_md_path, 'w') as f:
                    f.write(self.config.system_prompt)
                if self._debug:
                    print(f"[OpenClaw] Wrote system prompt to: {agents_md_path}")
            except Exception as e:
                print(f"[OpenClaw] Warning: Failed to write AGENTS.md: {e}")

        # Set the workspace directory in config
        openclaw_config["agents"]["defaults"]["workspace"] = workspace_dir

        # Embedded `agent --local` materializes MCP tools from mcp.servers via
        # the built-in bundle-mcp runtime; custom plugin tools are not merged
        # into the embedded attempt tool list.
        if self._proxy_urls:
            if "mcp" not in openclaw_config:
                openclaw_config["mcp"] = {}

            configured_servers = dict(openclaw_config["mcp"].get("servers", {}))
            for server_name, proxy_url in self._proxy_urls.items():
                configured_servers[server_name] = {
                    "transport": "streamable-http",
                    "url": proxy_url,
                    "requestTimeoutMs": int(_tool_timeout_seconds() * 1000),
                }
            openclaw_config["mcp"]["servers"] = configured_servers

            if self._debug:
                print(
                    f"[OpenClaw] Configured {len(self._proxy_urls)} proxy-backed MCP servers for bundle-mcp"
                )

        # Configure the generated static plugin
        if "plugins" not in openclaw_config:
            openclaw_config["plugins"] = {}
        if "entries" not in openclaw_config["plugins"]:
            openclaw_config["plugins"]["entries"] = {}

        # Enable the generated static plugin if we have one
        if self._generated_plugin_id:
            openclaw_config["plugins"]["entries"][self._generated_plugin_id] = {
                "enabled": True,
                "hooks": {"allowConversationAccess": True},
            }

            # Add to plugins.allow list (required for plugin to load)
            if "allow" not in openclaw_config["plugins"]:
                openclaw_config["plugins"]["allow"] = []
            if self._generated_plugin_id not in openclaw_config["plugins"]["allow"]:
                openclaw_config["plugins"]["allow"].append(self._generated_plugin_id)

            if self._debug:
                print(f"[OpenClaw] Enabled static plugin: {self._generated_plugin_id}")

        if self._proxy_urls or self._generated_plugin_id:
            if "agents" not in openclaw_config:
                openclaw_config["agents"] = {}
            if "entries" not in openclaw_config["agents"]:
                openclaw_config["agents"]["entries"] = {}

            main_agent = openclaw_config["agents"]["entries"].setdefault("main", {})

            if "tools" not in main_agent:
                main_agent["tools"] = {}
            if "allow" not in main_agent["tools"]:
                main_agent["tools"]["allow"] = []

            allowlist = main_agent["tools"]["allow"]
            for tool_name in [
                "group:plugins",
                "bundle-mcp",
                "memory_search",
                "memory_get",
            ]:
                if tool_name not in allowlist:
                    allowlist.append(tool_name)
            if self._generated_plugin_id and self._generated_plugin_id not in allowlist:
                allowlist.append(self._generated_plugin_id)

        if "tools" not in openclaw_config:
            openclaw_config["tools"] = {}
        if "web" not in openclaw_config["tools"]:
            openclaw_config["tools"]["web"] = {}
        if "search" not in openclaw_config["tools"]["web"]:
            openclaw_config["tools"]["web"]["search"] = {}
        if "fetch" not in openclaw_config["tools"]["web"]:
            openclaw_config["tools"]["web"]["fetch"] = {}
        openclaw_config["tools"]["web"]["search"]["enabled"] = False
        openclaw_config["tools"]["web"]["fetch"]["enabled"] = False

        if "browser" not in openclaw_config:
            openclaw_config["browser"] = {}
        openclaw_config["browser"]["enabled"] = False

        # Add skill directory if skills are configured
        if self._skill_temp_dir:
            if "skills" not in openclaw_config:
                openclaw_config["skills"] = {}
            if "load" not in openclaw_config["skills"]:
                openclaw_config["skills"]["load"] = {}
            if "extraDirs" not in openclaw_config["skills"]["load"]:
                openclaw_config["skills"]["load"]["extraDirs"] = []
            openclaw_config["skills"]["load"]["extraDirs"].append(self._skill_temp_dir)
            if self._debug:
                print(f"[OpenClaw] Added skill directory to config: {self._skill_temp_dir}")

        # Apply tools.deny from agent_kwargs (disable native tools)
        agent_kwargs = self.runtime_config.agent_kwargs or {}
        denied_tools = agent_kwargs.get("disallowed_tools", [])
        if denied_tools:
            if "tools" not in openclaw_config:
                openclaw_config["tools"] = {}
            existing_deny = openclaw_config["tools"].get("deny", [])
            openclaw_config["tools"]["deny"] = list(set(existing_deny + denied_tools))
            if self._debug:
                print(f"[OpenClaw] Denied tools: {openclaw_config['tools']['deny']}")

        # Write config to profile directory (--profile uses profile's config, not OPENCLAW_CONFIG_PATH)
        profile_config_path = os.path.join(self._profile_dir, "openclaw.json")
        try:
            with open(profile_config_path, 'w') as f:
                json.dump(openclaw_config, f, indent=2)
            self._openclaw_config_path = profile_config_path
        except Exception as e:
            raise RuntimeError(f"Failed to create OpenClaw config: {e}")

        if self._debug:
            print(f"[OpenClaw] Wrote config to profile: {profile_config_path}")
            print(f"[OpenClaw] Model: {model}")
            print(f"[OpenClaw] Workspace: {workspace_dir}")
            if self.config and self.config.system_prompt:
                prompt_preview = self.config.system_prompt[:100] + "..." if len(self.config.system_prompt) > 100 else self.config.system_prompt
                print(f"[OpenClaw] System prompt (AGENTS.md): {prompt_preview}")

        populate_openclaw_profile_auth(self._profile_dir, debug=self._debug)

    def _cleanup_temp_resources(self) -> None:
        """Clean up temporary resources outside the profile directory."""
        # Clean up skill temp directory (lives in output_dir, not profile_dir)
        if self._skill_temp_dir:
            cleanup_temp_directory(self._skill_temp_dir)
            self._skill_temp_dir = None

    async def _discover_and_generate_plugin(self) -> None:
        """
        Discover tools from proxy servers and generate a static OpenClaw plugin.
        """
        if not self._proxy_urls:
            return

        servers: List[MCPServerTools] = []

        async with aiohttp.ClientSession() as session:
            for server_name, proxy_url in self._proxy_urls.items():
                try:
                    # Query the proxy for available tools
                    request_body = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {}
                    }

                    async with session.post(proxy_url, json=request_body) as response:
                        result = await response.json()

                        if "error" in result:
                            print(f"[OpenClaw] Warning: Failed to list tools from {server_name}: {result['error']}")
                            continue

                        tools_data = result.get("result", {}).get("tools", [])
                        tools = [
                            MCPTool(
                                name=t["name"],
                                description=t.get("description", f"Tool from {server_name}"),
                                input_schema=t.get("inputSchema", {"type": "object", "properties": {}})
                            )
                            for t in tools_data
                        ]

                        servers.append(MCPServerTools(
                            name=server_name,
                            url=proxy_url,
                            tools=tools
                        ))

                        # Save tool list for records
                        self._mcp_tool_lists[server_name] = [
                            {"name": t["name"], "description": t.get("description", "")}
                            for t in tools_data
                        ]

                        if self._debug:
                            print(f"[OpenClaw] Discovered {len(tools)} tools from {server_name}")

                except Exception as e:
                    print(f"[OpenClaw] Warning: Failed to discover tools from {server_name}: {e}")

        if not servers:
            # The generated plugin also owns the provider-input observation
            # hook, so prompt/skill presentation remains measurable even when
            # this victim exposes no MCP tools.
            print("[OpenClaw] No MCP tools discovered; generating observation-only plugin")

        # Generate unique plugin ID for this task
        self._generated_plugin_id = f"mcp-tools-{uuid.uuid4().hex[:8]}"

        # Initialize plugin generator with profile-specific extensions directory
        # This ensures parallel tasks don't load each other's plugins
        self._plugin_generator = StaticPluginGenerator(
            extensions_dir=self._profile_extensions_dir,
            debug=self._debug
        )

        # Generate and install the static plugin
        try:
            self._plugin_generator.generate_plugin(
                servers=servers,
                plugin_id=self._generated_plugin_id,
                feedback_probe_path=self._feedback_probe_file,
                provider_observation_path=self._provider_observation_file,
            )

            total_tools = sum(len(s.tools) for s in servers)
            print(f"[OpenClaw] Generated static plugin '{self._generated_plugin_id}' with {total_tools} tools")

        except Exception as e:
            print(f"[OpenClaw] Error generating plugin: {e}")
            self._generated_plugin_id = None
            raise

    async def _setup_skills(self) -> None:
        """Setup skill directories and apply any skill injections."""
        skill_directories = self.config.skill_directories if self.config else []
        skill_injection = self.runtime_config.skill_injection

        # Check if we have any skills to process
        has_create_mode = skill_injection and any(
            any(inj.mode == "create" for inj in injs)
            for injs in skill_injection.values()
        )

        if not skill_directories and not has_create_mode:
            return

        # Create temp directory with skills (and injections applied)
        # OpenClaw expects skills in <dir>/<skill_name>/SKILL.md (no subpath)
        self._skill_temp_dir = create_injected_skills_directory(
            source_skill_dirs=skill_directories,
            skill_injection=skill_injection,
            skill_subpath="",  # OpenClaw loads skills from root of directory
            temp_prefix="openclaw_skills_",
            base_dir=self.output_dir,
        )

        if self._skill_temp_dir:
            print(f"[OpenClaw] Created skill temp directory: {self._skill_temp_dir}")

    def _create_mcp_server(self, server_config: MCPServerConfig) -> Any:
        """
        Create MCP server configuration placeholder.

        For OpenClaw, actual MCP handling is done via proxy servers.
        """
        return {
            "name": server_config.name,
            "transport": server_config.transport,
            "url": server_config.url,
            "command": server_config.command,
            "args": server_config.args,
            "env": server_config.env,
        }

    async def initialize(self) -> None:
        """
        Initialize agent:
        1. Check OpenClaw CLI is available
        2. Create MCP proxy servers with tool injections
        3. Configure OpenClaw to use proxy servers
        """
        if not self.config:
            raise ValueError("Agent config is required")

        # Check if openclaw CLI is available
        try:
            cli_env = self._build_openclaw_env()
            result = subprocess.run(
                self._build_openclaw_command("--version"),
                capture_output=True,
                text=True,
                timeout=10,
                env=cli_env,
            )
            if result.returncode != 0:
                raise RuntimeError("openclaw CLI not working properly")
            if self._debug:
                version_line = result.stdout.strip().split('\n')[0] if result.stdout else "unknown"
                print(f"[OpenClaw] CLI version: {version_line}")
        except FileNotFoundError:
            raise RuntimeError(
                "openclaw CLI not found. Install with: npm install -g openclaw"
            )

        # Initialize isolated profile for this task (prevents plugin conflicts in parallel execution)
        try:
            setup_result = subprocess.run(
                self._build_openclaw_command("--profile", self._profile_name, "setup"),
                capture_output=True,
                text=True,
                timeout=30,
                env=cli_env,
            )
            if setup_result.returncode != 0:
                print(f"[OpenClaw] Warning: Profile setup returned non-zero: {setup_result.stderr}")
            elif self._debug:
                print(f"[OpenClaw] Initialized isolated profile: {self._profile_name}")
        except Exception as e:
            print(f"[OpenClaw] Warning: Could not initialize profile: {e}")

        # Setup skills if configured
        await self._setup_skills()

        # Create MCP proxy servers
        self._proxy_manager = MCPProxyManager(
            debug=self._debug,
            hook_manager=self.hook_manager,
            event_log_path=self._mcp_event_log,
            episode_id=os.environ.get("DTAP_EVALUATION_EPISODE_ID", self._session_id),
        )

        # Get tool injections from runtime config
        tool_injections = self.runtime_config.mcp_injection or {}

        try:
            self._proxy_urls = await self._proxy_manager.create_proxies(
                mcp_configs=self.config.mcp_servers,
                tool_injections=tool_injections,
            )
            print(f"[OpenClaw] Created {len(self._proxy_urls)} MCP proxy servers")
        except Exception as e:
            raise RuntimeError(f"Failed to create MCP proxy servers: {e}")

        # Discover tools from proxies and generate static plugin
        await self._discover_and_generate_plugin()

        # Configure OpenClaw to use the generated static plugin
        self._configure_openclaw_with_proxies()

        # Load MCP server configs (for reference)
        await self.load_mcp_servers()

        print(f"[OpenClaw] Initialized with {len(self.mcp_servers)} MCP servers via proxy")

    async def _run_openclaw_cli(self, message: str) -> Dict[str, Any]:
        """
        Run OpenClaw CLI with a message.

        Args:
            message: User message

        Returns:
            Dict with 'success', 'content', 'error' keys
        """
        cmd = self._build_openclaw_command(
            "--profile", self._profile_name,  # Isolate from other parallel tasks
            "agent",
            "--local",  # Run in embedded mode (no gateway required)
            "--message", message,
            "--thinking", self._get_thinking_level(),
            "--session-id", self._session_id,
            "--json",
        )

        if self._debug:
            print(f"[OpenClaw CLI] Running: {' '.join(cmd[:6])}...")

        # Use inherited environment (--profile handles config isolation)
        env = self._build_openclaw_env()
        env["OPENCLAW_TRAJECTORY"] = "1"
        env["OPENCLAW_TRAJECTORY_DIR"] = self._runtime_trace_dir
        if self._debug and self._openclaw_config_path:
            print(f"[OpenClaw CLI] Using openclaw config: {self._openclaw_config_path}")
            print(f"[OpenClaw CLI] Runtime trace dir: {self._runtime_trace_dir}")

        proc = None
        # Freshness barrier anchor for the transcript that this invocation is
        # expected to update.  A previous turn's same-session JSONL must not be
        # accepted merely because it already exists.
        self._last_invocation_started_ns = time.time_ns()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=OPENCLAW_TIMEOUT_SECONDS,
            )

            output = stdout.decode().strip()
            error_output = stderr.decode().strip() if stderr else ""

            if self._debug:
                print(f"[OpenClaw CLI] Exit code: {proc.returncode}")
                if output:
                    # Truncate for display
                    display_output = output[:500] + "..." if len(output) > 500 else output
                    print(f"[OpenClaw CLI] Output: {display_output}")
                if error_output:
                    print("[OpenClaw CLI] Stderr:")
                    print(error_output)

            if proc.returncode != 0:
                return {
                    "success": False,
                    "content": "",
                    "error": error_output or f"Exit code {proc.returncode}",
                }

            # Current OpenClaw emits a stable JSON envelope after optional
            # migration/plugin status lines. Decode the largest complete JSON
            # object rather than assuming stdout starts with ``{``.
            decoder = json.JSONDecoder()
            decoded: List[tuple[int, Dict[str, Any]]] = []
            for index, character in enumerate(output):
                if character != "{":
                    continue
                try:
                    value, end = decoder.raw_decode(output[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    decoded.append((end, value))
            if decoded:
                result = max(decoded, key=lambda item: item[0])[1]
                self._last_cli_payload = result
                self._token_usage = _openclaw_token_usage(result)
                payload_texts = [
                    item.get("text", "")
                    for item in result.get("payloads", [])
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                ]
                meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
                content = "\n".join(payload_texts).strip()
                if not content:
                    content = str(meta.get("finalAssistantVisibleText", "")).strip()
                return {"success": True, "content": content or output, "error": None}

            return {
                "success": True,
                "content": output,
                "error": None,
            }

        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
            return {
                "success": False,
                "content": "",
                "error": f"Command timed out after {OPENCLAW_TIMEOUT_SECONDS}s",
            }
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": str(e),
            }

    async def run(
        self,
        user_input: Union[str, List[str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Run the agent with given input. Supports multi-turn conversations.

        Args:
            user_input: User instruction/query. Can be:
                - str: Single query (backward compatible)
                - List[str]: Multiple queries processed sequentially
            metadata: Optional metadata (task_id, domain, category, instruction)

        Returns:
            AgentResult with final output, turn count, trace ID, and trajectory
        """
        # Normalize input to list
        if isinstance(user_input, str):
            user_inputs = [user_input]
        else:
            user_inputs = user_input

        # Initialize trace metadata
        if self._trace_metadata is None:
            self._trace_metadata = metadata or {}

        # Set instruction in metadata
        if "instruction" not in self._trace_metadata:
            instr = " | ".join(user_inputs) if len(user_inputs) > 1 else (user_inputs[0] or "")
            if len(instr) > 500:
                instr = instr[:500]
            self._trace_metadata["instruction"] = instr

        if self._session_started_at is None:
            self._session_started_at = datetime.now()
        final_output = ""

        for query in user_inputs:
            # Add to conversation history
            self._conversation_history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat(),
            })

            # Run OpenClaw CLI
            response = await self._run_openclaw_cli(query)

            if response["success"]:
                final_output = response["content"]

                self._conversation_history.append({
                    "role": "assistant",
                    "content": final_output,
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                error_msg = f"Error: {response.get('error', 'Unknown error')}"
                final_output = error_msg

            self._turn_count += 1

        # Standalone OpenClaw callers retain the old behavior. The benchmark
        # runner defers export across task turns and calls finalize_trajectory
        # once after the complete multi-turn session.
        if not self._defer_trajectory_export:
            duration = (datetime.now() - self._session_started_at).total_seconds()
            await self._generate_trajectory(
                duration,
                min_mtime_ns=self._last_invocation_started_ns,
            )

        return self.get_result()

    def begin_trajectory_batch(self) -> None:
        """Defer trajectory conversion until all task turns have completed."""
        self._defer_trajectory_export = True

    async def finalize_trajectory(self) -> AgentResult:
        """Export one fresh, complete session trajectory for the current task."""
        if self._session_started_at is None:
            self._session_started_at = datetime.now()
        duration = (datetime.now() - self._session_started_at).total_seconds()
        await self._generate_trajectory(
            duration,
            min_mtime_ns=self._last_invocation_started_ns,
        )
        self._defer_trajectory_export = False
        return self.get_result()

    async def _get_session_log_path(
        self,
        wait_seconds: float = 0.0,
        warn: bool = True,
        min_mtime_ns: Optional[int] = None,
    ) -> Optional[str]:
        """Find a session trace updated by the current OpenClaw invocation."""
        expected_names = (
            f"{self._session_id}.jsonl",
            f"{self._session_id}.trajectory.jsonl",
        )
        profile_dir = Path(self._profile_dir)
        runtime_trace_dir = Path(self._runtime_trace_dir)
        deadline = time.monotonic() + max(wait_seconds, 0.0)
        tried_materialize = False

        def _is_fresh(path: Path) -> bool:
            try:
                info = path.stat()
            except OSError:
                return False
            if not path.is_file():
                return False
            if min_mtime_ns is None:
                return True
            return info.st_mtime_ns >= min_mtime_ns

        def _resolve_trajectory_pointer(pointer_path: Path) -> Optional[str]:
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

            runtime_file = pointer.get("runtimeFile")
            if runtime_file and os.path.exists(runtime_file):
                return runtime_file
            return None

        def _existing_jsonl_from_output(output: str) -> Optional[str]:
            """Find a transcript materialized by current OpenClaw releases."""
            candidates: List[str] = []
            for line in reversed(output.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    candidates.append(line)
                    continue

                def collect(item: Any) -> None:
                    if isinstance(item, str):
                        candidates.append(item)
                    elif isinstance(item, dict):
                        for child in item.values():
                            collect(child)
                    elif isinstance(item, list):
                        for child in item:
                            collect(child)

                collect(value)
            for candidate in candidates:
                path = Path(candidate).expanduser()
                if path.suffix == ".jsonl" and _is_fresh(path):
                    return str(path)
            return None

        while True:
            for expected_name in expected_names:
                runtime_trace = runtime_trace_dir / expected_name
                if _is_fresh(runtime_trace):
                    return str(runtime_trace)

            session_dir = profile_dir / "agents" / "main" / "sessions"
            for expected_name in expected_names:
                session_file = session_dir / expected_name
                if _is_fresh(session_file):
                    return str(session_file)

            pointer_file = session_dir / f"{self._session_id}.trajectory-path.json"
            if pointer_file.exists():
                resolved = _resolve_trajectory_pointer(pointer_file)
                if resolved and _is_fresh(Path(resolved)):
                    return resolved

            # OpenClaw 2026.9 stores embedded-agent transcripts in SQLite, so
            # there may be no profile-local JSONL until the supported
            # `transcripts path --transcript` command materializes one.
            if not tried_materialize:
                tried_materialize = True
                try:
                    result = await asyncio.to_thread(
                        subprocess.run,
                        self._build_openclaw_command(
                            "--profile", self._profile_name,
                            "transcripts", "path", self._session_id,
                            "--transcript", "--json",
                        ),
                        capture_output=True,
                        text=True,
                        timeout=15,
                        env=self._build_openclaw_env(),
                    )
                    materialized = _existing_jsonl_from_output(result.stdout)
                    if materialized:
                        return materialized
                except Exception:
                    pass

            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.25)

        if warn:
            print(
                f"[WARNING] OpenClaw session log not found for session_id="
                f"{self._session_id} under {self._profile_dir}"
            )
        return None

    def register_environment_feedback(
        self,
        injections: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> None:
        """Register trusted probes after injection IDs have been returned."""
        if self._proxy_manager is None:
            return
        remaining = list(results)
        for injection in injections:
            result = next((
                item for item in remaining
                if item.get("server_name") == injection.get("server_name")
                and item.get("tool_name") == injection.get("tool_name")
                and item.get("turn_id") == injection.get("turn_id")
                and item.get("feedback_step_index") == injection.get("feedback_step_index")
            ), None)
            if result is None:
                result = {
                    "feedback_step_index": injection.get("feedback_step_index"),
                    "result": None,
                }
            else:
                remaining.remove(result)
            self._proxy_manager.register_feedback_probe(
                build_environment_feedback_probe(injection, result)
            )
        self._proxy_manager.write_provider_probe_registry(self._feedback_probe_file)

    def register_attack_feedback(self, attack_config: Any) -> None:
        """Register non-environment attack probes after the proxy exists."""
        if self._proxy_manager is None:
            return
        for probe in build_static_feedback_probes(attack_config):
            self._proxy_manager.register_feedback_probe(probe)
        self._proxy_manager.write_provider_probe_registry(self._feedback_probe_file)

    def _observe_feedback_messages(self, session_path: str) -> None:
        if self._proxy_manager is None:
            return
        try:
            path = Path(session_path)
            if path.stat().st_size <= 32 * 1024 * 1024:
                for line in path.read_text(encoding="utf-8").splitlines():
                    value = json.loads(line)
                    self._proxy_manager.observe_runtime_event(value)

                    def visit(item: Any) -> None:
                        if isinstance(item, dict):
                            snapshot = item.get("messagesSnapshot")
                            if isinstance(snapshot, list):
                                self._proxy_manager.observe_messages_snapshot(snapshot)
                            for child in item.values():
                                visit(child)
                        elif isinstance(item, list):
                            for child in item:
                                visit(child)

                    visit(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        try:
            provider_path = Path(self._provider_observation_file)
            if provider_path.is_file() and provider_path.stat().st_size <= 4 * 1024 * 1024:
                for line in provider_path.read_text(encoding="utf-8").splitlines():
                    self._proxy_manager.observe_provider_input(json.loads(line))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass

    async def _generate_trajectory(
        self,
        duration: float = 0.0,
        *,
        min_mtime_ns: Optional[int] = None,
    ) -> None:
        """Generate trajectory from OpenClaw session JSONL."""
        session_path = await self._get_session_log_path(
            wait_seconds=5.0,
            min_mtime_ns=min_mtime_ns,
        )
        if not session_path and self._last_cli_payload:
            # Embedded OpenClaw 2026.9 runs may intentionally be detached from
            # the transcript catalog. Preserve a standards-compatible dialogue
            # trajectory from the stable CLI envelope instead of dropping the
            # victim artifact entirely. Tool names remain available through the
            # separately captured MCP catalog; arguments/results cannot be
            # reconstructed from this fallback envelope.
            meta = self._last_cli_payload.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            prompt = meta.get("finalPromptText")
            assistant = meta.get("finalAssistantVisibleText")
            entries = []
            common = {
                "traceSchema": "openclaw-trajectory", "schemaVersion": 1,
                "traceId": self._session_id, "sessionId": self._session_id,
            }
            if isinstance(prompt, str) and prompt:
                entries.append({**common, "type": "prompt.submitted", "data": {"prompt": prompt}})
            if isinstance(assistant, str) and assistant:
                entries.append({
                    **common, "type": "model.completed",
                    "data": {
                        "assistantTexts": [assistant],
                        **({"usage": self._token_usage} if self._token_usage else {}),
                    },
                })
            entries.append({
                **common, "type": "session.ended",
                "data": {"status": meta.get("stopReason", "completed")},
            })
            fallback_path = Path(self._runtime_trace_dir) / f"{self._session_id}.jsonl"
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            fallback_path.write_text(
                "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
                encoding="utf-8",
            )
            session_path = str(fallback_path)
        if not session_path:
            self._current_trajectory = None
            print(
                "[ERROR] OpenClaw runtime trace missing; no trajectory generated "
                f"for session_id={self._session_id}"
            )
            return
        self._observe_feedback_messages(session_path)
        trace_metadata = dict(self._trace_metadata or {})
        if self._token_usage is not None:
            trace_metadata["token_usage"] = self._token_usage
        if self._last_cli_payload:
            cli_meta = self._last_cli_payload.get("meta")
            agent_meta = (
                cli_meta.get("agentMeta") if isinstance(cli_meta, dict) else None
            )
            execution_trace = (
                cli_meta.get("executionTrace")
                if isinstance(cli_meta, dict) else None
            )
            model = (
                agent_meta.get("model") if isinstance(agent_meta, dict) else None
            ) or (
                execution_trace.get("winnerModel")
                if isinstance(execution_trace, dict) else None
            )
            if isinstance(model, str) and model.strip():
                trace_metadata["model"] = model.strip()
        self._trace_metadata = trace_metadata
        try:
            convert_call = partial(
                self.trajectory_converter.convert_session_log,
                session_log_path=session_path,
                metadata=self._trace_metadata,
                task_id=(self._trace_metadata or {}).get("task_id"),
                duration=duration,
                mcp_tool_lists=self._mcp_tool_lists or None,
            )
            traj = await asyncio.wait_for(
                asyncio.to_thread(convert_call),
                timeout=OPENCLAW_TRAJECTORY_CONVERSION_TIMEOUT_SECONDS,
            )
            if traj:
                self._current_trajectory = traj
            else:
                self._current_trajectory = None
                print(
                    "[ERROR] OpenClaw runtime trace conversion produced no "
                    f"trajectory for session_id={self._session_id}"
                )
        except asyncio.TimeoutError:
            print(
                "[WARNING] Timed out converting OpenClaw runtime trace; "
                f"session_id={self._session_id}"
            )
            self._current_trajectory = None
        except Exception as e:
            print(f"[WARNING] Failed to generate trajectory: {e}")
            self._current_trajectory = None

    def get_result(self) -> AgentResult:
        """Get the current execution result."""
        final_output = None
        for msg in reversed(self._conversation_history):
            if msg.get("role") == "assistant":
                final_output = msg.get("content")
                break

        return AgentResult(
            final_output=final_output,
            turn_count=self._turn_count,
            trajectory=self._current_trajectory,
            trace_id=self._session_id,
        )

    def reset_conversation(self) -> None:
        """Reset conversation for a new session."""
        self._conversation_history = []
        self._turn_count = 0
        self._session_id = f"dt-arena-{uuid.uuid4().hex[:8]}"
        self._mcp_event_log = os.path.join(
            self._runtime_trace_dir, f"{self._session_id}.mcp-events.jsonl"
        )
        self._feedback_probe_file = os.path.join(
            self._runtime_trace_dir, f"{self._session_id}.feedback-probes.json"
        )
        self._provider_observation_file = os.path.join(
            self._runtime_trace_dir, f"{self._session_id}.provider-feedback.jsonl"
        )
        self._trace_metadata = None
        self._current_trajectory = None
        self._last_cli_payload = None
        self._token_usage = None
        self._session_started_at = None
        self._last_invocation_started_ns = None
        self._defer_trajectory_export = False

    async def cleanup(self) -> None:
        """Clean up resources: stop proxies, restore config."""
        probe_path = Path(self._feedback_probe_file)
        self.reset_conversation()

        # Stop all proxy servers
        if self._proxy_manager:
            await self._proxy_manager.stop_all()
            self._proxy_manager = None
            self._proxy_urls = {}

        # The provider hook needs raw values transiently for exact matching,
        # but only its raw-free boolean observations may survive the run.
        try:
            probe_path.unlink(missing_ok=True)
        except OSError:
            pass

        # Clean up temp resources outside profile dir
        self._cleanup_temp_resources()

        # Clean up profile directory (isolated extensions for this task)
        if hasattr(self, '_profile_dir') and os.path.exists(self._profile_dir):
            try:
                shutil.rmtree(self._profile_dir)
                if self._debug:
                    print(f"[OpenClaw] Cleaned up profile directory: {self._profile_dir}")
            except Exception as e:
                print(f"[OpenClaw] Warning: Failed to clean up profile directory: {e}")
