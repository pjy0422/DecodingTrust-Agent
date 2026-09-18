from __future__ import annotations

import ast
from pathlib import Path

import dt_arena.policy_eval as policy_eval
from dt_arena.policy_eval.artifact_contract import (
    ARTIFACT_SCHEMA,
    ARTIFACT_SCHEMA_VERSION,
    ArtifactLayout,
)


def test_public_api_is_semantic():
    assert policy_eval.EvaluationSecurityPolicy.__name__ == "EvaluationSecurityPolicy"
    assert callable(policy_eval.create_policy_mcp_server)
    assert not hasattr(policy_eval, "M4SecurityPolicy")
    assert not hasattr(policy_eval, "create_m6_mcp_server")


def test_artifact_v1_compatibility(tmp_path: Path):
    assert ARTIFACT_SCHEMA == "dtap-agent-rl-episode"
    assert ARTIFACT_SCHEMA_VERSION == 1
    layout = ArtifactLayout(tmp_path)
    assert layout.attempt(1).name == "attempt-0001"


def _string_assignment(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
    raise AssertionError(name)


def _feedback_suffix(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_policy_prompt":
            values = [
                child.value for child in ast.walk(node)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and "After a genuine failed victim execution" in child.value
            ]
            assert len(values) == 1
            return values[0]
    raise AssertionError("build_policy_prompt")


def _env_keys(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "update"
                and isinstance(func.value, ast.Name)
                and func.value.id == "env"
            ):
                keys.update(kw.arg for kw in node.keywords if kw.arg is not None)
                for arg in node.args:
                    if isinstance(arg, ast.Dict):
                        keys.update(
                            key.value for key in arg.keys
                            if isinstance(key, ast.Constant) and isinstance(key.value, str)
                        )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "env"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    keys.add(target.slice.value)
    return keys


def test_latest_holdout_prompt_is_byte_frozen():
    import hashlib
    current = Path(policy_eval.__file__).resolve().parent / "planning/current.py"
    prompt = _string_assignment(current, "PLACEMENT_POLICY_PROMPT_TEMPLATE").format(max_submissions=2)
    prompt += _feedback_suffix(current)
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == "94422599f87769747e90f184022f25bd123dd227177929fa701a05afd2580380"


def test_harness_and_planning_are_separate():
    root = Path(policy_eval.__file__).resolve().parent
    runner = (root / "scripts/run_policy_e2e.py").read_text(encoding="utf-8")
    harness = (root / "harness/claude_code.py").read_text(encoding="utf-8")
    current = (root / "planning/current.py").read_text(encoding="utf-8")
    assert "asyncio.create_subprocess_exec" not in runner
    assert "asyncio.create_subprocess_exec" in harness
    assert "class CurrentPlanningStrategy" in current
    assert "ClaudeCodeHarness" in runner
    assert "planning_strategy.execute" in runner


def test_current_claude_launch_contract_literals_are_present():
    root = Path(policy_eval.__file__).resolve().parent
    harness = (root / "harness/claude_code.py").read_text(encoding="utf-8")
    runner = (root / "scripts/run_policy_e2e.py").read_text(encoding="utf-8")
    launch_required = (
        '"--output-format"', '"stream-json"', '"--verbose"', '"--max-turns"',
        '"--mcp-config"', '"--strict-mcp-config"', '"--settings"',
        '"--allowedTools"', '"--disallowedTools"', '"--model"',
    )
    env_required = (
        "DTAP_HARNESS_URL", "DTAP_EPISODE_TOKEN", "CLAUDE_CONFIG_DIR",
        "MCP_TOOL_TIMEOUT", "MCP_TIMEOUT",
        "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT",
    )
    provider_required = (
        '"DTAP_POLICY_ANTHROPIC_BASE_URL"',
        '"DTAP_POLICY_USE_API_KEY_AS_AUTH_TOKEN"',
    )
    assert not [item for item in launch_required if item not in harness]
    assert not [item for item in env_required if item not in _env_keys(root / "scripts/run_policy_e2e.py")]
    assert not [item for item in provider_required if item not in runner]


def test_no_runtime_slime_imports():
    root = Path(policy_eval.__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "slime" or name.startswith("slime.") or name.startswith("examples.dtap_agent_rl") for name in names):
                offenders.append((path, names))
    assert not offenders
