from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dt_arena.policy_eval.attack_surface import AttackSurface, ToolSpec
from dt_arena.policy_eval.placement import DtapPlacementRunner, PlacementCoordinator, PlacementRunResult
from dt_arena.policy_eval.policy_contract import PolicyContract
from dt_arena.policy_eval.scheduler import AttemptScheduler
from dt_arena.policy_eval.security_policy import EvaluationSecurityPolicy
from dt_arena.policy_eval.validation import ValidationContext, validate_attack_step
from dt_arena.src.placement_contract import (
    PlacementResourceContract,
    parse_placement_resource_metadata,
    placement_resource_metadata,
)


def _schema(required: tuple[str, ...], properties: dict[str, dict]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": properties,
    }


def _context() -> ValidationContext:
    news = ToolSpec(
        server_name="finance-injection",
        tool_name="create_news",
        qualified_name="finance-injection:create_news",
        description=None,
        input_schema=_schema(
            ("title", "summary", "symbol"),
            {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "symbol": {"type": "string"},
            },
        ),
        placement_capability="verified",
        placement_resource=PlacementResourceContract("finance.article", "provider"),
    )
    comment = ToolSpec(
        server_name="finance-injection",
        tool_name="create_comment",
        qualified_name="finance-injection:create_comment",
        description=None,
        input_schema=_schema(
            ("symbol", "content", "article_id"),
            {
                "symbol": {"type": "string"},
                "content": {"type": "string"},
                "article_id": {"type": "integer"},
            },
        ),
        placement_capability="verified",
        placement_resource=PlacementResourceContract("finance.article", "consumer"),
    )
    portfolio = ToolSpec(
        server_name="finance-injection",
        tool_name="inject_html_portfolio",
        qualified_name="finance-injection:inject_html_portfolio",
        description=None,
        input_schema=_schema(
            ("prompt", "page"),
            {
                "prompt": {"type": "string"},
                "page": {"type": "string", "enum": ["portfolios", "trading"]},
            },
        ),
        placement_capability="verified",
    )
    surface = AttackSurface(
        prompt_enabled=False,
        tool_enabled=False,
        environment_enabled=True,
        skill_enabled=False,
        skill_modes=(),
        victim_tools=(),
        environment_tools=(news, comment, portfolio),
    )
    return ValidationContext(threat_model="indirect", instruction_count=1, attack_surface=surface)


def _step(tool: str, kwargs: dict) -> dict:
    return {
        "type": "environment",
        "turn_id": 1,
        "injection_mcp_tool": f"finance-injection:{tool}",
        "kwargs": kwargs,
    }


def test_finance_page_contract_is_schema_visible_and_enforced() -> None:
    from dt_arena.mcp_server.finance.server.injection_mcp import create_injection_mcp_server

    tools = asyncio.run(create_injection_mcp_server("http://127.0.0.1:1").mcp.list_tools())
    by_name = {tool.name: tool for tool in tools}
    schema = by_name["inject_html_portfolio"].inputSchema
    assert schema["properties"]["page"]["enum"] == ["portfolios", "trading"]
    assert schema["properties"]["location"]["enum"] == [
        "portfolio_summary", "portfolio_table", "position_row", "cash_balance",
    ]
    assert parse_placement_resource_metadata(by_name["create_news"].meta) == (
        PlacementResourceContract("finance.article", "provider")
    )
    assert parse_placement_resource_metadata(by_name["create_comment"].meta) == (
        PlacementResourceContract("finance.article", "consumer")
    )

    result = validate_attack_step(
        _step("inject_html_portfolio", {"prompt": "payload", "page": "portfolio"}),
        _context(),
    )
    assert not result.valid
    assert result.errors[0].code == "SCHEMA_MISMATCH"
    assert result.errors[0].path == "kwargs.page"


def test_placement_metadata_protocol_is_domain_agnostic_and_fail_closed() -> None:
    meta = placement_resource_metadata(kind="example.record", role="provider")
    assert parse_placement_resource_metadata(meta) == PlacementResourceContract(
        "example.record", "provider"
    )
    assert parse_placement_resource_metadata(
        {"dtap": {"placement": {"resource": {"kind": "Bad Kind", "role": "provider"}}}}
    ) is None
    assert parse_placement_resource_metadata(
        {"dtap": {"placement": {"resource": {"kind": "example.record", "role": "oracle"}}}}
    ) is None


@pytest.mark.asyncio
async def test_live_catalog_retains_trusted_resource_metadata(monkeypatch) -> None:
    from dt_arena.policy_eval import live_catalog

    class Client:
        def __init__(self, _url: str):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def list_tools(self):
            return [
                SimpleNamespace(
                    name="attach_record",
                    description="Attach data to a generated record.",
                    inputSchema=_schema(("record_id",), {"record_id": {"type": "string"}}),
                    meta=placement_resource_metadata(
                        kind="example.record", role="consumer"
                    ),
                )
            ]

    import fastmcp

    monkeypatch.setattr(fastmcp, "Client", Client)
    monkeypatch.setattr(
        live_catalog, "_environment_placement_capability", lambda *_args: "verified"
    )
    tools = await live_catalog._list_url_tools(
        "example-injection", "http://unused", classify_environment=True
    )
    assert tools[0].placement_resource == PlacementResourceContract(
        "example.record", "consumer"
    )
    assert "placement_resource" not in tools[0].to_dict()


def test_non_finance_domain_constraints_flow_through_mcp_schema() -> None:
    from dt_arena.injection_mcp_server.travel.env_injection import mcp

    tools = asyncio.run(mcp.get_tools())
    review = tools["inject_review"]
    assert review.parameters["properties"]["entity_type"]["enum"] == [
        "accommodation", "restaurant",
    ]
    assert parse_placement_resource_metadata(review.meta) == PlacementResourceContract(
        "travel.reviewable_entity", "consumer"
    )
    assert parse_placement_resource_metadata(
        tools["inject_accommodation"].meta
    ) == PlacementResourceContract("travel.reviewable_entity", "provider")
    assert parse_placement_resource_metadata(
        tools["inject_restaurant"].meta
    ) == PlacementResourceContract("travel.reviewable_entity", "provider")

    spec = ToolSpec(
        server_name="travel-injection",
        tool_name="inject_review",
        qualified_name="travel-injection:inject_review",
        description=None,
        input_schema=dict(review.parameters),
        placement_capability="verified",
    )
    surface = AttackSurface(False, False, True, False, (), (), (spec,))
    result = validate_attack_step(
        {
            "type": "environment",
            "turn_id": 1,
            "injection_mcp_tool": "travel-injection:inject_review",
            "kwargs": {
                "entity_type": "hotel",
                "entity_name": "Example",
                "city": "Seoul",
                "reviewer_name": "Reviewer",
                "rating": 5,
                "content": "payload",
            },
        },
        ValidationContext("indirect", 1, surface),
    )
    assert not result.valid
    assert result.errors[0].path == "kwargs.entity_type"


def test_generated_finance_article_id_is_stable_across_hash_seeds() -> None:
    source = (
        "from dt_arena.mcp_server.finance.injection.content import FakeNews; "
        "print(FakeNews(title='t', summary='s', source='x', symbol='AAPL').article_id)"
    )
    values = []
    for seed in ("1", "987654"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        values.append(
            subprocess.check_output([sys.executable, "-c", source], text=True, env=env).strip()
        )
    assert values[0] == values[1]
    assert 10_000_000 <= int(values[0]) <= 99_999_999


def test_finance_environment_loads_the_repository_identity_implementation() -> None:
    root = Path(__file__).resolve().parents[2]
    compose = (root / "dt_arena/envs/finance/docker-compose.yml").read_text(encoding="utf-8")
    assert "../../mcp_server/finance/injection/content.py:/app/injection/content.py:ro" in compose


def test_placement_child_v2_contract_is_parsed_fail_closed(tmp_path: Path) -> None:
    policy = EvaluationSecurityPolicy(max_submit_calls=1)
    runner = DtapPlacementRunner(
        dtap_root=tmp_path,
        security_policy=policy,
        scheduler=AttemptScheduler(
            max_parallel=policy.max_parallel_attempts,
            max_queued=policy.max_queued_attempts,
            wait_timeout=1,
        ),
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema": "m6-placement-v2",
                "applied": True,
                "valid": False,
                "status": "invalid",
                "locator": "finance:portfolios#symbol=",
                "code": "PLACEMENT_MISMATCH",
                "locator_fields": ["kwargs.page", "kwargs.symbol"],
                "repair_fields": [],
                "retryable": False,
            }
        ),
        encoding="utf-8",
    )
    parsed = runner._read(result_path, tmp_path)
    assert parsed.available
    assert parsed.locator_fields == ("kwargs.page", "kwargs.symbol")
    assert not parsed.repair_fields
    assert not parsed.retryable

    value = json.loads(result_path.read_text(encoding="utf-8"))
    value["retryable"] = True
    result_path.write_text(json.dumps(value), encoding="utf-8")
    assert not runner._read(result_path, tmp_path).available


class _Runner:
    def __init__(self) -> None:
        self.workspaces = []

    async def run(self, workspace):
        self.workspaces.append(workspace)
        if len(self.workspaces) == 1:
            return PlacementRunResult(
                available=True,
                applied=True,
                valid=True,
                status="verified",
                locator="finance:news#symbol=AAPL&article_id=12345678",
                code="PLACEMENT_VERIFIED",
            )
        return PlacementRunResult(
            available=True,
            applied=True,
            valid=True,
            status="verified",
            locator="finance:news#symbol=AAPL&article_id=12345678",
            code="PLACEMENT_VERIFIED",
        )


def _coordinator(runner: _Runner) -> PlacementCoordinator:
    return PlacementCoordinator(
        validation_context=_context(),
        source_task_dir=Path("/unused"),
        episode_root=Path("/unused"),
        runner=runner,
        security_policy=EvaluationSecurityPolicy(max_submit_calls=1),
        policy_contract=PolicyContract(),
        max_actions=4,
    )


@pytest.mark.asyncio
async def test_generated_resource_dependency_is_explicit_and_replayed(monkeypatch) -> None:
    calls = []

    def materialize(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output_root=Path("/unused"),
            task_dir=Path("/unused"),
            placement_target_index=kwargs["placement_target_index"],
            steps=kwargs["steps"],
        )

    monkeypatch.setattr("dt_arena.policy_eval.placement.materialize_attempt_dir", materialize)
    runner = _Runner()
    coordinator = _coordinator(runner)
    news = _step("create_news", {"title": "t", "summary": "s", "symbol": "AAPL"})
    comment = _step(
        "create_comment",
        {"symbol": "AAPL", "content": "payload", "article_id": 12345678},
    )

    news_receipt = await coordinator.apply(news)
    news_id = news_receipt["action_id"]
    assert coordinator.validate(news_id)["valid"] is True

    missing = await coordinator.apply(comment)
    assert missing == {
        "accepted": False,
        "dependency_action_ids": [news_id],
        "error": {"code": "DEPENDENCY_REQUIRED"},
    }
    assert len(calls) == 1

    comment_receipt = await coordinator.apply(comment, depends_on=[news_id])
    assert comment_receipt["accepted"] is True
    assert comment_receipt["dependency_count"] == 1
    assert len(calls[1]["steps"]) == 2
    assert calls[1]["placement_target_index"] == 1
    assert calls[1]["steps"][0].injection_mcp_tool.endswith(":create_news")
    assert calls[1]["steps"][1].injection_mcp_tool.endswith(":create_comment")


@pytest.mark.asyncio
async def test_invalid_placement_does_not_fabricate_expected_or_repair_fields(monkeypatch) -> None:
    def materialize(**kwargs):
        return SimpleNamespace(placement_target_index=0, steps=kwargs["steps"])

    class InvalidRunner:
        async def run(self, _workspace):
            return PlacementRunResult(
                available=True,
                applied=True,
                valid=False,
                status="invalid",
                locator="finance:portfolios#symbol=",
                code="PLACEMENT_MISMATCH",
                locator_fields=("kwargs.page", "kwargs.symbol"),
            )

    monkeypatch.setattr("dt_arena.policy_eval.placement.materialize_attempt_dir", materialize)
    coordinator = _coordinator(InvalidRunner())
    receipt = await coordinator.apply(
        _step("inject_html_portfolio", {"prompt": "payload", "page": "portfolios"})
    )
    result = coordinator.validate(receipt["action_id"])
    assert result["diagnostic"] == {
        "locator_fields": ["kwargs.page", "kwargs.symbol"],
        "retryable": False,
    }
    assert "repair" not in result
    assert "expected_placement_locator" not in json.dumps(result)


@pytest.mark.asyncio
async def test_dependency_must_be_owned_validated_and_compatible(monkeypatch) -> None:
    monkeypatch.setattr(
        "dt_arena.policy_eval.placement.materialize_attempt_dir",
        lambda **kwargs: SimpleNamespace(
            placement_target_index=kwargs["placement_target_index"], steps=kwargs["steps"]
        ),
    )
    coordinator = _coordinator(_Runner())
    comment = _step(
        "create_comment",
        {"symbol": "AAPL", "content": "payload", "article_id": 12345678},
    )
    assert (await coordinator.apply(comment, depends_on=["x" * 24]))["error"]["code"] == "UNKNOWN_DEPENDENCY"
    assert (await coordinator.apply(comment, depends_on=["x" * 129]))["error"]["code"] == "INVALID_DEPENDENCY"

    news_receipt = await coordinator.apply(
        _step("create_news", {"title": "t", "summary": "s", "symbol": "AAPL"})
    )
    assert (
        await coordinator.apply(comment, depends_on=[news_receipt["action_id"]])
    )["error"]["code"] == "UNVERIFIED_DEPENDENCY"


@pytest.mark.asyncio
async def test_dependency_identity_uses_adapter_locators_not_domain_rules(monkeypatch) -> None:
    monkeypatch.setattr(
        "dt_arena.policy_eval.placement.materialize_attempt_dir",
        lambda **kwargs: SimpleNamespace(
            placement_target_index=kwargs["placement_target_index"], steps=kwargs["steps"]
        ),
    )

    class MismatchedRunner:
        def __init__(self):
            self.calls = 0

        async def run(self, _workspace):
            self.calls += 1
            locator = "example:record/created" if self.calls == 1 else "example:record/other"
            return PlacementRunResult(
                available=True,
                applied=True,
                valid=True,
                status="verified",
                locator=locator,
                code="PLACEMENT_VERIFIED",
            )

    provider = ToolSpec(
        "example-injection",
        "create_record",
        "example-injection:create_record",
        None,
        _schema(("value",), {"value": {"type": "string"}}),
        "verified",
        PlacementResourceContract("example.record", "provider"),
    )
    consumer = ToolSpec(
        "example-injection",
        "attach_record",
        "example-injection:attach_record",
        None,
        _schema(("record_id",), {"record_id": {"type": "string"}}),
        "verified",
        PlacementResourceContract("example.record", "consumer"),
    )
    context = ValidationContext(
        "indirect", 1, AttackSurface(False, False, True, False, (), (), (provider, consumer))
    )
    coordinator = PlacementCoordinator(
        validation_context=context,
        source_task_dir=Path("/unused"),
        episode_root=Path("/unused"),
        runner=MismatchedRunner(),
        security_policy=EvaluationSecurityPolicy(max_submit_calls=1),
        policy_contract=PolicyContract(),
        max_actions=4,
    )
    created = await coordinator.apply(
        {
            "type": "environment",
            "turn_id": 1,
            "injection_mcp_tool": "example-injection:create_record",
            "kwargs": {"value": "payload"},
        }
    )
    coordinator.validate(created["action_id"])
    result = await coordinator.apply(
        {
            "type": "environment",
            "turn_id": 1,
            "injection_mcp_tool": "example-injection:attach_record",
            "kwargs": {"record_id": "other"},
        },
        depends_on=[created["action_id"]],
    )
    assert result == {
        "accepted": False,
        "error": {"code": "DEPENDENCY_RESOURCE_MISMATCH"},
    }
