from __future__ import annotations

from types import SimpleNamespace

import pytest

from dt_arena.policy_eval.attack_surface import AttackSurface, ToolSpec
from dt_arena.policy_eval.authority import (
    EpisodeAuthority,
    EpisodeAuthorityRegistry,
    EpisodeCredentials,
)
from dt_arena.policy_eval.mcp_server import M4EpisodeService, create_policy_mcp_server
from dt_arena.policy_eval.policy_contract import PolicyContract
from dt_arena.policy_eval.protocol import HARNESS_PROTOCOL_LAZY_SCHEMA_V2
from dt_arena.policy_eval.security_policy import EvaluationSecurityPolicy
from dt_arena.policy_eval.service import EpisodeView


class _Runtime:
    terminal = False

    def record_security_failure(self, *, stage: str) -> None:  # pragma: no cover
        raise AssertionError(stage)


class _Event:
    def set(self) -> None:  # pragma: no cover
        raise AssertionError("unexpected terminal event")


def _surface(*, oversized: bool = False) -> tuple[AttackSurface, ToolSpec, ToolSpec]:
    victim = ToolSpec(
        server_name="browser",
        tool_name="click",
        qualified_name="browser:click",
        description="Click a browser element.\n\nArgs:\n    selector: target selector",
        input_schema={"type": "object", "properties": {"selector": {"type": "string"}}},
    )
    environment = ToolSpec(
        server_name="travel-injection",
        tool_name="inject_review",
        qualified_name="travel-injection:inject_review",
        description="Inject one listing review.\n\nArgs:\n    content: review text",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["content"],
            "properties": {
                "content": {"type": "string", **({"description": "x" * 70_000} if oversized else {})}
            },
        },
        placement_capability="verified",
    )
    return (
        AttackSurface(
            prompt_enabled=True,
            tool_enabled=True,
            environment_enabled=True,
            skill_enabled=True,
            skill_modes=("append",),
            victim_tools=(victim,),
            environment_tools=(environment,),
            prompt_modes=("suffix", "override"),
            tool_modes=("suffix", "override"),
            skill_targets=("browser-skill",),
        ),
        victim,
        environment,
    )


def _registered_service(*, oversized: bool = False, protocol: str = HARNESS_PROTOCOL_LAZY_SCHEMA_V2):
    surface, victim, environment = _surface(oversized=oversized)
    view = EpisodeView(
        task=SimpleNamespace(task_instruction=["book a room"], threat_model="indirect"),
        attack_surface=surface,
    )
    authority = EpisodeAuthority(
        view=view,
        coordinator=SimpleNamespace(runtime=_Runtime()),
        terminal_event=_Event(),
        policy_contract=PolicyContract(),
    )
    credentials = EpisodeCredentials.issue("lazy-schema-test-session")
    registry = EpisodeAuthorityRegistry(digest_key=b"z" * 32)
    registry.register(credentials, authority)
    service = M4EpisodeService(
        registry,
        PolicyContract(),
        EvaluationSecurityPolicy(max_submit_calls=3),
        protocol,
    )
    return service, credentials.mcp_bearer_token, authority, victim, environment


def test_v1_surface_remains_eager_and_unchanged() -> None:
    service, token, _, _, _ = _registered_service(protocol="v1")

    surface = service.get_attack_surface(token)

    assert "candidate_step_schema" in surface
    assert "input_schema" in surface["victim_tools"][0]
    assert "server_name" in surface["environment_tools"][0]


def test_v2_surface_omits_schemas_and_preserves_existing_descriptions() -> None:
    service, token, _, victim, environment = _registered_service()

    surface = service.get_attack_surface(token)

    assert "candidate_step_schema" not in surface
    assert "input_schema" not in str(surface)
    assert surface["channels"]["tool"]["modes"] == ["suffix", "override"]
    assert surface["channels"]["environment"]["required_fields"][-1] == "kwargs"
    assert surface["victim_tools"] == [
        victim.to_summary_dict(compact_description=True)
    ]
    assert surface["environment_tools"] == [
        environment.to_summary_dict(compact_description=True)
    ]
    assert set(surface["victim_tools"][0]) == {"qualified_name", "description"}


def test_schema_lookup_is_exact_idempotent_and_target_bound() -> None:
    service, token, authority, _, environment = _registered_service()

    first = service.get_tool_schema(token, environment.qualified_name)
    second = service.get_tool_schema(token, environment.qualified_name)
    rejected = service.get_tool_schema(token, "travel-injection:missing")

    assert first == second
    assert first["usage"] == "environment"
    assert first["description"] == environment.to_summary_dict(compact_description=True)["description"]
    assert first["input_schema"] == environment.input_schema
    assert first["candidate_step_schema"]["properties"]["injection_mcp_tool"] == {
        "const": environment.qualified_name
    }
    assert first["candidate_step_schema"]["properties"]["kwargs"] == environment.input_schema
    assert rejected == {"found": False, "error": {"code": "TOOL_NOT_ALLOWED"}}
    assert authority.schema_access.summary() == (environment.qualified_name,)
    assert authority.mcp_calls.summary()["tool_counts"]["get_tool_schema"] == 3


def test_target_schema_must_be_loaded_before_validation() -> None:
    service, token, _, victim, environment = _registered_service()
    environment_step = {
        "type": "environment",
        "turn_id": 1,
        "injection_mcp_tool": environment.qualified_name,
        "kwargs": {"content": "plausible listing text"},
    }
    tool_step = {
        "type": "tool",
        "mode": "suffix",
        "content": "additional result text",
        "injected_tool": victim.qualified_name,
    }

    missing = service.validate_attack_step(token, environment_step)
    assert missing["errors"][0]["code"] == "SCHEMA_NOT_LOADED"

    service.get_tool_schema(token, environment.qualified_name)
    assert service.validate_attack_step(token, environment_step)["valid"] is True
    assert service.validate_attack_step(token, tool_step)["errors"][0]["code"] == "SCHEMA_NOT_LOADED"

    service.get_tool_schema(token, victim.qualified_name)
    assert service.validate_attack_step(token, tool_step)["valid"] is True


def test_prompt_and_skill_validation_need_no_schema_lookup() -> None:
    service, token, authority, _, _ = _registered_service()

    prompt = service.validate_attack_step(
        token,
        {"type": "prompt", "turn_id": 1, "mode": "suffix", "content": "test"},
    )
    skill = service.validate_attack_step(
        token,
        {
            "type": "skill",
            "mode": "append",
            "content": "test",
            "skill_name": "browser-skill",
            "row": -1,
        },
    )

    assert prompt["valid"] is True
    assert skill["valid"] is True
    assert authority.schema_access.summary() == ()


def test_oversized_schema_is_absent_and_fails_closed() -> None:
    service, token, _, _, environment = _registered_service(oversized=True)

    surface = service.get_attack_surface(token)
    lookup = service.get_tool_schema(token, environment.qualified_name)
    validation = service.validate_attack_step(
        token,
        {
            "type": "environment",
            "turn_id": 1,
            "injection_mcp_tool": environment.qualified_name,
            "kwargs": {"content": "test"},
        },
    )

    assert surface["environment_tools"] == []
    assert lookup["error"]["code"] == "TOOL_NOT_ALLOWED"
    assert validation["errors"][0]["code"] == "TARGET_NOT_ALLOWED"


def test_schema_access_is_episode_scoped() -> None:
    service_a, token_a, _, _, environment = _registered_service()
    service_b, token_b, _, _, _ = _registered_service()
    step = {
        "type": "environment",
        "turn_id": 1,
        "injection_mcp_tool": environment.qualified_name,
        "kwargs": {"content": "test"},
    }

    service_a.get_tool_schema(token_a, environment.qualified_name)

    assert service_a.validate_attack_step(token_a, step)["valid"] is True
    assert service_b.validate_attack_step(token_b, step)["errors"][0]["code"] == "SCHEMA_NOT_LOADED"


@pytest.mark.asyncio
async def test_v1_has_six_tools_and_v2_has_seven() -> None:
    policy = EvaluationSecurityPolicy(max_submit_calls=1)
    v1 = create_policy_mcp_server(EpisodeAuthorityRegistry(), security_policy=policy)
    v2 = create_policy_mcp_server(
        EpisodeAuthorityRegistry(),
        security_policy=policy,
        harness_protocol=HARNESS_PROTOCOL_LAZY_SCHEMA_V2,
    )

    v1_tools = await v1.get_tools()
    v2_tools = await v2.get_tools()

    assert len(v1_tools) == 6
    assert "get_tool_schema" not in v1_tools
    assert len(v2_tools) == 7
    assert "get_tool_schema" in v2_tools
