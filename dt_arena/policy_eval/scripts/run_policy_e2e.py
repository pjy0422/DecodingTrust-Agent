"""Run the current Claude Code policy through the complete DTAP policy-evaluation path.

The prompt contains no candidate attack step. The policy discovers the task and
live attack surface, authors and validates a plan, applies/verifies placement when
needed, submits the plan, and waits for the real victim/judge receipt.
"""

from __future__ import annotations

from dt_arena.policy_eval.harness import ClaudeCodeHarness, HarnessRequest
from dt_arena.policy_eval.planning import PlanningContext, create_planning_strategy, planning_strategy_names
from dt_arena.policy_eval.planning.current import (
    CORE_POLICY_PROMPT,
    CORE_POLICY_PROMPT_TEMPLATE,
    PLACEMENT_POLICY_PROMPT,
    PLACEMENT_POLICY_PROMPT_TEMPLATE,
    build_policy_prompt,
)

import argparse
import asyncio
import json
import os
import shutil
import socket
import tempfile
from contextlib import suppress
from pathlib import Path

import yaml
from dt_arena.policy_eval.attempt_runner import DtapAttemptRunner
from dt_arena.policy_eval.authority import (
    EpisodeAuthority,
    EpisodeAuthorityRegistry,
    EpisodeCredentials,
    registered_authority,
)
from dt_arena.policy_eval.episode import load_task_snapshot
from dt_arena.policy_eval.episode_runtime import EpisodeRuntimeState, EpisodeStatus
from dt_arena.policy_eval.feedback import (
    AnthropicMessagesJSONCompleter,
    FeedbackBuilder,
    FeedbackBuildLimits,
    FeedbackMode,
    PromptedReasoningSummarizer,
    ReasoningSummaryConfig,
)
from dt_arena.policy_eval.feedback.digestor import PromptedLLMDigestor
from dt_arena.policy_eval.integrity import BenchmarkIntegrityGuard
from dt_arena.policy_eval.mcp_server import create_core_policy_mcp_server, create_policy_mcp_server
from dt_arena.policy_eval.placement import DtapPlacementRunner, PlacementCoordinator
from dt_arena.policy_eval.policy_contract import PolicyContract, PolicyLeakageGuard
from dt_arena.policy_eval.scheduler import AttemptScheduler
from dt_arena.policy_eval.scripts.discover_live_view import live_view
from dt_arena.policy_eval.security_policy import EvaluationSecurityPolicy
from dt_arena.policy_eval.submission import SubmissionCoordinator
from dt_arena.policy_eval.validation import ValidationContext

TOOLS = {
    "mcp__dtap__get_task_spec",
    "mcp__dtap__get_attack_surface",
    "mcp__dtap__validate_attack_step",
    "mcp__dtap__submit_attack",
}
PLACEMENT_POLICY_TOOLS = TOOLS | {
    "mcp__dtap__apply_attack_step",
    "mcp__dtap__validate_placement",
}
DISALLOWED_NATIVE_TOOLS = {
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "NotebookEdit",
}




POLICY_BASE_URL_ENV = "DTAP_POLICY_ANTHROPIC_BASE_URL"
POLICY_AUTH_FROM_API_KEY_ENV = "DTAP_POLICY_USE_API_KEY_AS_AUTH_TOKEN"


class RecordingRunner:
    """Record only the policy-authored candidate, then delegate to real DTAP."""

    policy_hardened = True

    def __init__(
        self,
        delegate: DtapAttemptRunner,
        *,
        artifacts_dir: Path | None = None,
    ) -> None:
        self.delegate = delegate
        self.plans: list[list[dict]] = []
        self.artifacts_dir = artifacts_dir
        self.exported_victim_traces = 0
        self.exported_victim_mcp_events = 0
        self.exported_judge_artifacts = 0
        self.delegate_completed = False

    def _export_artifacts(self, workspace) -> None:
        if self.artifacts_dir is None:
            return
        attempt_index = int(getattr(workspace, "attempt_index", len(self.plans)))
        attempt_dir = self.artifacts_dir / "attempts" / f"attempt-{attempt_index:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        for name in ("judge_result.json", ".m4-verdict.json"):
            candidates = sorted(workspace.output_root.rglob(name))
            if candidates:
                output_name = "judge-result.json" if name == "judge_result.json" else "judge-verdict.json"
                shutil.copy2(candidates[0], attempt_dir / output_name)
                shutil.copy2(candidates[0], self.artifacts_dir / output_name)
                self.exported_judge_artifacts += 1
        for candidate in sorted(workspace.output_root.rglob("*.json")):
            if candidate.name in {"judge_result.json", ".m4-verdict.json"}:
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("trajectory"), list):
                shutil.copy2(candidate, attempt_dir / "victim-trajectory.json")
                shutil.copy2(candidate, self.artifacts_dir / "victim-trajectory.json")
                self.exported_victim_traces = 1
                break
        event_logs = sorted(
            workspace.output_root.rglob("*.mcp-events.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if event_logs:
            shutil.copy2(event_logs[0], attempt_dir / "victim-mcp-events.jsonl")
            shutil.copy2(event_logs[0], self.artifacts_dir / "victim-mcp-events.jsonl")
            self.exported_victim_mcp_events = 1
        diagnostic = workspace.output_root / ".dtap-stderr.log"
        if diagnostic.is_file():
            shutil.copy2(diagnostic, attempt_dir / "dtap-stderr.log")
            shutil.copy2(diagnostic, self.artifacts_dir / "dtap-stderr.log")
        (self.artifacts_dir / "episode-state.json").write_text(
            json.dumps(
                {
                    "schema": "dtap-agent-rl-episode-state",
                    "schema_version": 1,
                    "plan_generated": bool(self.plans),
                    "submissions_retained": len(self.plans),
                    "attempt_indices": list(range(1, len(self.plans) + 1)),
                    "victim_trajectory_retained": bool(self.exported_victim_traces),
                    "victim_mcp_log_retained": bool(self.exported_victim_mcp_events),
                    "judge_artifacts_retained": self.exported_judge_artifacts,
                    "evaluation_delegate_completed": self.delegate_completed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    async def run(self, workspace):
        config = yaml.safe_load(workspace.config_path.read_text(encoding="utf-8"))
        turns = config["Attack"]["attack_turns"]
        self.plans.append(turns)
        if self.artifacts_dir is not None:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            attempt_index = int(getattr(workspace, "attempt_index", len(self.plans)))
            attempt_dir = self.artifacts_dir / "attempts" / f"attempt-{attempt_index:04d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(workspace.config_path, attempt_dir / "submitted-config.yaml")
            shutil.copy2(workspace.config_path, self.artifacts_dir / "submitted-config.yaml")
        try:
            result = await self.delegate.run(workspace)
            self.delegate_completed = True
            return result
        finally:
            self._export_artifacts(workspace)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_port(port: int) -> None:
    for _ in range(200):
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.025)
    raise RuntimeError("M5 policy MCP server did not start")


def _tool_names(stdout: str) -> set[str]:
    names: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            name = value.get("name")
            if isinstance(name, str):
                names.add(name)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for line in stdout.splitlines():
        try:
            walk(json.loads(line))
        except Exception:
            continue
    return names


def _environment_step_count(plan: list[dict]) -> int:
    return sum(step.get("type") == "environment" for turn in plan for step in turn.get("attack_steps", []))


def _environment_tools(plan: list[dict]) -> list[str]:
    return [
        str(step.get("injection_mcp_tool"))
        for turn in plan
        for step in turn.get("attack_steps", [])
        if step.get("type") == "environment" and step.get("injection_mcp_tool")
    ]


def _victim_artifacts_complete(runner: RecordingRunner, victim_agent_type: str) -> bool:
    """OpenClaw's proxy event stream is its deterministic headless contract."""
    if victim_agent_type == "openclaw":
        return runner.exported_victim_mcp_events >= 1
    return runner.exported_victim_traces >= 1


async def _main(args) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("export ANTHROPIC_API_KEY")
    claude = shutil.which(args.claude_bin)
    if claude is None:
        raise RuntimeError("Claude Code CLI not found")
    harness = ClaudeCodeHarness(claude)

    snapshot = load_task_snapshot(args.task_dir)
    feedback_mode = FeedbackMode(args.feedback_mode)
    planning_context = PlanningContext(
        placement_enabled=args.placement_enabled,
        max_submissions=args.max_submissions,
        feedback_mode=feedback_mode,
    )
    planning_strategy = create_planning_strategy(args.planning_strategy)
    policy_prompt_text = planning_strategy.build_prompt(planning_context)
    digest_completer = None
    feedback_builder = None
    if feedback_mode is not FeedbackMode.DISABLED:
        digestor = None
        summarizer = None
        reasoning = ReasoningSummaryConfig(
            enabled=args.reasoning_summary,
            timeout_seconds=args.digestor_timeout,
        )
        if feedback_mode is FeedbackMode.FINAL_DETERMINISTIC_DIGESTOR:
            base_url = (
                os.environ.get("DTAP_DIGESTOR_ANTHROPIC_BASE_URL", "").strip()
                or os.environ.get(POLICY_BASE_URL_ENV, "").strip()
                or os.environ.get("ANTHROPIC_BASE_URL", "").strip()
            )
            api_key = (
                os.environ.get("DTAP_DIGESTOR_API_KEY", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
            )
            if not base_url or not api_key:
                raise RuntimeError("Digestor mode requires a base URL and API key")
            digest_completer = AnthropicMessagesJSONCompleter(
                base_url=base_url,
                api_key=api_key,
                model=args.digestor_model,
                timeout_seconds=args.digestor_timeout,
            )
            digestor = PromptedLLMDigestor(digest_completer)
            if args.reasoning_summary:
                summarizer = PromptedReasoningSummarizer(digest_completer)
        feedback_builder = FeedbackBuilder(
            mode=feedback_mode,
            digestor=digestor,
            reasoning_summarizer=summarizer,
            reasoning=reasoning,
            limits=FeedbackBuildLimits(
                # PromptedLLMDigestor permits one schema-only retry. Keep each
                # provider request bounded while allowing that retry to finish.
                digest_timeout_seconds=2 * args.digestor_timeout
                + 1.0,
            ),
        )
    artifacts_dir = args.artifacts_dir.expanduser().resolve() if args.artifacts_dir else None
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot.task_dir / "config.yaml", artifacts_dir / "original-config.yaml")
        (artifacts_dir / "policy-prompt.txt").write_text(
            policy_prompt_text,
            encoding="utf-8",
        )
    view = await live_view(snapshot)
    source_turns = (
        yaml.safe_load((snapshot.task_dir / "config.yaml").read_text(encoding="utf-8"))
        .get("Attack", {})
        .get("attack_turns", [])
    )

    policy = EvaluationSecurityPolicy(
        # Q remains a separate abuse bound, but includes repair headroom so an
        # invalid plan does not crowd out the intended H victim executions.
        max_submit_calls=max(6, args.max_submissions * 3),
        max_parallel_attempts=1,
        max_queued_attempts=1,
        queue_wait_timeout_seconds=args.timeout,
        inherited_dtap_env_names=(
            "PYTHONPATH",
            "ANTHROPIC_API_KEY",
            "DTAP_VICTIM_ANTHROPIC_BASE_URL",
            "DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN",
            "DTAP_ENV_VERIFICATION",
            "DTAP_ENV_VERIFICATION_STRICT",
            "WINDOWS_DATA_DIR",
            "MACOS_DATA_DIR",
        ),
    )
    scheduler = AttemptScheduler(max_parallel=1, max_queued=1, wait_timeout=args.timeout)
    real_runner = DtapAttemptRunner(
        agent_type=args.victim_agent_type,
        model=args.victim_model,
        max_turns=args.victim_max_turns,
        timeout_seconds=args.timeout,
        python_executable=args.python,
        dtap_root=args.dtap_root,
        security_policy=policy,
        scheduler=scheduler,
        port_range_start=args.port_range_start,
        extra_env={},
    )
    runner = RecordingRunner(real_runner, artifacts_dir=artifacts_dir)
    placement = None
    credentials = EpisodeCredentials.issue("glm-e2e-adapter-session-0123456789")
    real_runner.extra_env["DTAP_EVALUATION_EPISODE_ID"] = credentials.public_episode_id
    if artifacts_dir is not None:
        (artifacts_dir / "episode-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "dtap-agent-rl-episode",
                    "schema_version": 1,
                    "episode_id": credentials.public_episode_id,
                    "policy_model": args.policy_model,
                    "victim_model": args.victim_model,
                    "victim_agent_type": args.victim_agent_type,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    runtime = EpisodeRuntimeState(
        max_submissions=args.max_submissions,
        max_submit_calls=policy.max_submit_calls,
    )
    contract = PolicyContract(
        PolicyLeakageGuard(
            secrets=(credentials.adapter_session_id, credentials.mcp_bearer_token),
            forbidden_fragments=(str(snapshot.task_dir),),
        )
    )
    registry = EpisodeAuthorityRegistry()
    terminal_event = asyncio.Event()

    with tempfile.TemporaryDirectory(prefix="slime-m5-glm-e2e-") as temp_dir:
        root = Path(temp_dir)
        if args.placement_enabled:
            placement_runner = DtapPlacementRunner(
                dtap_root=args.dtap_root,
                security_policy=policy,
                scheduler=scheduler,
                python_executable=args.python,
                timeout_seconds=args.timeout,
                extra_env={
                    "DT_DISABLE_DEFAULT_PORTS": "1",
                    "DT_PORT_RANGE_START": str(args.port_range_start),
                    "DT_PORT_RANGE_END": str(args.port_range_start + 511),
                },
            )
            placement = PlacementCoordinator(
                validation_context=ValidationContext.from_view(view),
                source_task_dir=snapshot.task_dir,
                source_manifest=snapshot.benchmark_manifest,
                episode_root=root / "placements",
                runner=placement_runner,
                security_policy=policy,
                policy_contract=contract,
                max_actions=policy.max_placement_actions,
            )
        controller = SubmissionCoordinator(
            validation_context=ValidationContext.from_view(view),
            runtime=runtime,
            source_task_dir=snapshot.task_dir,
            source_manifest=snapshot.benchmark_manifest,
            episode_root=root / "attempts" / credentials.public_episode_id,
            runner=runner,
            security_policy=policy,
            policy_contract=contract,
            terminal_event=terminal_event,
            placement_coordinator=placement,
            feedback_builder=feedback_builder,
        )
        authority = EpisodeAuthority(
            view,
            controller,
            terminal_event,
            contract,
            placement_coordinator=placement,
        )
        port = _free_port()
        server = (
            create_policy_mcp_server(registry, security_policy=policy)
            if args.placement_enabled
            else create_core_policy_mcp_server(registry, security_policy=policy)
        )
        server_task = asyncio.create_task(
            server.run_async(
                transport="http",
                host="127.0.0.1",
                port=port,
                stateless_http=True,
                show_banner=False,
            )
        )
        try:
            await _wait_port(port)
            with registered_authority(registry, credentials=credentials, authority=authority):
                mcp_config = root / "mcp.json"
                mcp_config.write_text(
                    json.dumps(
                        {
                            "mcpServers": {
                                "dtap": {
                                    "type": "http",
                                    "url": "${DTAP_HARNESS_URL}",
                                    "headers": {"Authorization": "Bearer ${DTAP_EPISODE_TOKEN}"},
                                    "timeout": (args.timeout + 60) * 1000,
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                settings = root / "settings.json"
                settings.write_text(json.dumps({"disableAllHooks": True}), encoding="utf-8")
                config_home = root / "claude-home"
                config_home.mkdir()
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if key not in {"ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"}
                }
                env.update(
                    DTAP_HARNESS_URL=f"http://127.0.0.1:{port}/mcp/",
                    DTAP_EPISODE_TOKEN=credentials.mcp_bearer_token,
                    CLAUDE_CONFIG_DIR=str(config_home),
                    # submit_attack performs the real victim + judge run. Some
                    # domains legitimately exceed Claude Code's 300 s MCP tool
                    # default, so keep its client deadline aligned with this
                    # smoke's explicit evaluation deadline.
                    MCP_TOOL_TIMEOUT=str((args.timeout + 60) * 1000),
                    MCP_TIMEOUT=str((args.timeout + 60) * 1000),
                    CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=str((args.timeout + 60) * 1000),
                )
                provider_url = os.environ.get(POLICY_BASE_URL_ENV, "").strip()
                if provider_url:
                    env["ANTHROPIC_BASE_URL"] = provider_url
                if os.environ.get(POLICY_AUTH_FROM_API_KEY_ENV) == "1":
                    env["ANTHROPIC_AUTH_TOKEN"] = env["ANTHROPIC_API_KEY"]
                allowed_tools = PLACEMENT_POLICY_TOOLS if args.placement_enabled else TOOLS
                request = HarnessRequest(
                    prompt=policy_prompt_text,
                    cwd=root,
                    env=env,
                    model=args.policy_model,
                    max_turns=args.policy_max_turns,
                    mcp_config=mcp_config,
                    settings=settings,
                    allowed_tools=frozenset(allowed_tools),
                    disallowed_tools=frozenset(DISALLOWED_NATIVE_TOOLS),
                    timeout_seconds=args.timeout + 60,
                )
                execution = await planning_strategy.execute(
                    harness, request, planning_context
                )
                stdout = execution.stdout
                stderr = execution.stderr
                if artifacts_dir is not None:
                    # Preserve failed policy runs too; these traces are needed
                    # to distinguish a model/tool-contract defect from victim
                    # or infrastructure failures in the domain matrix.
                    (artifacts_dir / "policy.jsonl").write_text(stdout, encoding="utf-8")
                    (artifacts_dir / "policy.stderr.log").write_text(stderr, encoding="utf-8")
                if execution.returncode:
                    detail = stderr[-2000:] or stdout[-2000:]
                    raise RuntimeError(f"policy exited {execution.returncode}: {detail}")
                # Placement tools are conditional: a prompt/tool/skill-only
                # plan has nothing to apply or read back. The four core calls
                # remain mandatory for every authored E2E plan.
                missing = TOOLS - _tool_names(stdout)
                if missing:
                    raise RuntimeError(f"GLM policy missed required tools: {sorted(missing)}")
                for secret in (credentials.mcp_bearer_token, str(snapshot.task_dir)):
                    if secret in stdout or secret in stderr:
                        raise RuntimeError("privileged episode value leaked")

            if not runner.plans or any(not plan for plan in runner.plans):
                raise RuntimeError("GLM did not produce a non-empty accepted plan")
            # Equality is useful diagnostic metadata, not evidence of template
            # access: a policy can independently reconstruct the same minimal
            # plan from the public task and tool schemas. Non-disclosure is
            # enforced above by the prompt/provenance and leakage checks.
            final_plan = runner.plans[-1]
            matches_source_template = final_plan == source_turns
            if runtime.status not in {EpisodeStatus.SUCCEEDED, EpisodeStatus.EXHAUSTED}:
                raise RuntimeError(f"real DTAP evaluation did not terminate cleanly: {runtime.status.value}")
            environment_steps = _environment_step_count(final_plan)
            environment_tools = _environment_tools(final_plan)
            if (
                args.placement_enabled
                and environment_steps
                and (
                    placement is None
                    or placement.verified_actions < environment_steps
                    or placement.validated_actions < environment_steps
                )
            ):
                raise RuntimeError("GLM did not complete a verified M6 placement receipt")
            if artifacts_dir is not None and not _victim_artifacts_complete(runner, args.victim_agent_type):
                kind = (
                    "OpenClaw victim MCP event log"
                    if args.victim_agent_type == "openclaw"
                    else "DTAP victim trajectory"
                )
                raise RuntimeError(f"artifact export found no {kind}")
            BenchmarkIntegrityGuard.verify(snapshot.task_dir, snapshot.benchmark_manifest)
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "evaluation_completed": True,
                        "failure_class": None,
                        "episode_id": credentials.public_episode_id,
                        "plan_generated": True,
                        "policy_model": args.policy_model,
                        "victim_model": args.victim_model,
                        "generated_plan": final_plan,
                        "generated_plans": runner.plans,
                        "matches_source_template": matches_source_template,
                        "attack_success": runtime.status is EpisodeStatus.SUCCEEDED,
                        "episode_status": runtime.status.value,
                        "submissions": runtime.victim_runs_started,
                        "victim_runs": runtime.victim_runs_started,
                        "placement_actions": placement.applied_actions if placement else 0,
                        "placements_verified": placement.verified_actions if placement else 0,
                        "environment_steps": environment_steps,
                        "action_applied": bool(
                            environment_steps == 0
                            or not args.placement_enabled
                            or (placement is not None and placement.applied_actions >= environment_steps)
                        ),
                        "environment_tools": environment_tools,
                        "placement_applicable": environment_steps > 0,
                        "placement_covered": bool(
                            environment_steps > 0
                            and placement is not None
                            and placement.verified_actions >= environment_steps
                        ),
                        "placement_verified": bool(
                            environment_steps > 0
                            and placement is not None
                            and placement.verified_actions >= environment_steps
                        ),
                        "victim_completed": True,
                        "judge_completed": True,
                        "victim_agent_type": args.victim_agent_type,
                        "victim_mcp_events": runner.exported_victim_mcp_events,
                        "judge_artifacts": runner.exported_judge_artifacts,
                        "feedback_mode": feedback_mode.value,
                        "reasoning_summary_enabled": args.reasoning_summary,
                        "digestor_usage": (digest_completer.usage.to_dict() if digest_completer else None),
                        "artifacts_dir": str(artifacts_dir) if artifacts_dir else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        finally:
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--dtap-root", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--policy-model", default="glm-5.2")
    parser.add_argument("--victim-model", default="glm-5.2")
    parser.add_argument("--victim-agent-type", default="claudesdk")
    parser.add_argument("--policy-max-turns", type=int, default=16)
    parser.add_argument("--victim-max-turns", type=int, default=80)
    parser.add_argument("--max-submissions", type=int, default=2)
    parser.add_argument(
        "--planning-strategy",
        choices=planning_strategy_names(),
        default="current",
        help="planning algorithm; all strategies reuse the same policy harness",
    )
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--placement", dest="placement_enabled", action="store_true")
    parser.add_argument(
        "--feedback-mode",
        choices=[mode.value for mode in FeedbackMode],
        default=FeedbackMode.DISABLED.value,
    )
    parser.add_argument("--digestor-model", default="glm-5.2")
    parser.add_argument(
        "--digestor-timeout",
        type=float,
        default=30.0,
        help="timeout per hosted Digestor or reasoning-summary request",
    )
    parser.add_argument("--reasoning-summary", action="store_true")
    parser.add_argument("--port-range-start", type=int, default=20_000)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help=(
            "opt-in trusted output directory for policy.jsonl, original/submitted "
            "configs, and victim traces consumable by tools/dtap-trajectory-viewer"
        ),
    )
    args = parser.parse_args()
    args.task_dir = args.task_dir.expanduser().resolve()
    args.dtap_root = args.dtap_root.expanduser().resolve()
    if args.max_submissions < 1:
        parser.error("--max-submissions must be positive")
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
