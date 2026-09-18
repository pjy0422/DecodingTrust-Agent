"""M7 feedback-boundary tests over an alternate, non-E2E benchmark sample.

The live 24-case matrix selects benchmark record zero.  This suite deliberately
selects a later record for each Linux domain and threat model, preferring one
with an environment step so domain correlation gets broad coverage.  It stops
at the trusted feedback boundary: no environment, victim model, digestor, or
judge is started.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent.openclaw.src.mcp_proxy import FeedbackObserver, MCPEventSink
from dt_arena.src.feedback_observation import (
    build_environment_feedback_probe,
    build_static_feedback_probes,
)
from dt_arena.src.types.task import AttackConfig, AttackStepConfig, AttackTurn


ROOT = Path(__file__).parents[1]
DOMAINS = (
    "browser", "code", "crm", "customer-service", "finance", "legal",
    "medical", "os-filesystem", "research", "telecom", "travel", "workflow",
)
THREAT_MODELS = ("direct", "indirect")


@dataclass(frozen=True)
class BoundaryCase:
    domain: str
    threat_model: str
    config_path: Path
    config: dict[str, Any]
    benchmark_index: int
    step_index: int
    step: AttackStepConfig

    @property
    def id(self) -> str:
        task_id = self.config["Task"]["task_id"]
        return f"{self.domain}-{self.threat_model}-{task_id}"


def _benchmark_records(domain: str, threat_model: str) -> list[dict[str, Any]]:
    path = ROOT / "benchmark" / domain / f"{threat_model}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _config_path(record: dict[str, Any]) -> Path:
    return ROOT.joinpath(
        "dataset", record["domain"], "malicious", record["threat_model"],
        record["risk_category"], str(record["task_id"]), "config.yaml",
    )


def _pick_step(config_path: Path) -> tuple[int, AttackStepConfig]:
    attack = AttackConfig.from_yaml(str(config_path))
    assert attack is not None
    flattened = [step for turn in attack.attack_turns for step in turn.attack_steps]
    # Prefer environment evidence because it exercises domain read correlation;
    # otherwise tool presentation is more specific than prompt presentation.
    for preferred in ("environment", "tool", "prompt"):
        for index, step in enumerate(flattened):
            if step.type == preferred:
                return index, step
    raise AssertionError(f"no M7-compatible attack step: {config_path}")


def _load_cases() -> tuple[BoundaryCase, ...]:
    cases: list[BoundaryCase] = []
    for domain in DOMAINS:
        for threat_model in THREAT_MODELS:
            records = _benchmark_records(domain, threat_model)
            assert len(records) > 1
            selected = 1
            for index, record in enumerate(records[1:], 1):
                candidate_path = _config_path(record)
                candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
                steps = [
                    step
                    for turn in candidate.get("Attack", {}).get("attack_turns", [])
                    for step in turn.get("attack_steps", [])
                ]
                if any(step.get("type") == "environment" for step in steps):
                    selected = index
                    break
            assert records[selected] != records[0]
            config_path = _config_path(records[selected])
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            step_index, step = _pick_step(config_path)
            cases.append(BoundaryCase(
                domain, threat_model, config_path, config, selected, step_index, step,
            ))
    return tuple(cases)


CASES = _load_cases()


def _environment_probe(case: BoundaryCase) -> dict[str, Any]:
    assert case.step.injection_mcp_tool
    server, tool = case.step.injection_mcp_tool.split(":", 1)
    receipt: dict[str, Any] = {"result": "ok"}
    if server == "customer-service-injection" and tool.startswith("cs_case_"):
        receipt = {"result": {"case_ids": ["case-secondary-42"]}}
    elif server == "legal-injection" and tool == "inject_fake_document":
        receipt = {"result": {"document_id": "document-secondary-42"}}
    return build_environment_feedback_probe({
        "server_name": server,
        "tool_name": tool,
        "feedback_step_index": case.step_index,
        "kwargs": case.step.kwargs,
    }, receipt)


def _static_probe(case: BoundaryCase) -> dict[str, Any]:
    attack = AttackConfig(attack_turns=[
        AttackTurn(turn_id=1, attack_steps=[case.step]),
    ])
    probe = build_static_feedback_probes(attack)[0]
    # The isolated one-step config starts at zero; restore the benchmark step
    # identity so this exercises the same flattening contract as production.
    return {**probe, "step_index": case.step_index}


def _probe(case: BoundaryCase) -> dict[str, Any]:
    return _environment_probe(case) if case.step.type == "environment" else _static_probe(case)


def _runtime_servers(case: BoundaryCase) -> set[str]:
    return {
        str(item["name"])
        for item in case.config.get("Agent", {}).get("mcp_servers", [])
        if isinstance(item, dict) and item.get("enabled", True) and item.get("name")
    }


def _runtime_pattern(case: BoundaryCase, probe: dict[str, Any]) -> dict[str, Any]:
    runtime_servers = _runtime_servers(case)
    for pattern in probe["victim_patterns"]:
        server = str(pattern["tool"]).split(":", 1)[0]
        if server in runtime_servers:
            return pattern
    raise AssertionError(
        f"{case.id}: no feedback pattern uses a configured victim MCP; "
        f"runtime={sorted(runtime_servers)}, patterns={probe['victim_patterns']}"
    )


def _event(path: Path) -> dict[str, Any]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return next(item for item in events if item["type"] == "feedback.observed")


def _observe_environment(
    tmp_path: Path,
    case: BoundaryCase,
    *,
    response_has_payload: bool,
    present_result: bool,
) -> dict[str, Any]:
    probe = _environment_probe(case)
    pattern = _runtime_pattern(case, probe)
    payload = "\n".join(probe["payload_values"])
    response = payload if response_has_payload else "result without the current payload"
    server, tool = str(pattern["tool"]).replace("*", "read").split(":", 1)
    path = tmp_path / f"{case.id}.jsonl"
    observer = FeedbackObserver(MCPEventSink(path, case.id))
    observer.register(probe)
    observer.observe_call(
        call_id="read-secondary-case",
        server=server,
        tool=tool,
        arguments=dict(pattern["fields"]),
        result={"content": [{"type": "text", "text": response}], "isError": False},
    )
    observer.observe_messages_snapshot([{
        "role": "toolResult",
        "content": [{
            "type": "text",
            "text": response if present_result else "different provider-bound result",
        }],
    }])
    observer.emit()
    return _event(path)


def _observe_static(
    tmp_path: Path,
    case: BoundaryCase,
    *,
    include_payload: bool,
    emit_boundary: bool,
) -> dict[str, Any]:
    probe = _static_probe(case)
    payload = "\n".join(probe["payload_values"])
    path = tmp_path / f"{case.id}.jsonl"
    observer = FeedbackObserver(MCPEventSink(path, case.id))
    observer.register(probe)
    if emit_boundary:
        value = payload if include_payload else "provider input without current payload"
        if probe["presentation_source"] == "submitted_prompt":
            event = {"type": "prompt.submitted", "data": {"prompt": value}}
        elif probe["presentation_source"] == "available_tools":
            target = probe["target_tool"].replace(":", "__")
            event = {
                "type": "context.compiled",
                "data": {"availableTools": [{"name": target, "description": value}]},
            }
        else:
            event = {"type": "context.compiled", "data": {"systemPrompt": value}}
        observer.observe_runtime_event(event)
    observer.emit()
    return _event(path)


def _observe(
    tmp_path: Path,
    case: BoundaryCase,
    *,
    response_has_payload: bool,
    present_result: bool,
) -> dict[str, Any]:
    if case.step.type == "environment":
        return _observe_environment(
            tmp_path, case,
            response_has_payload=response_has_payload,
            present_result=present_result,
        )
    return _observe_static(
        tmp_path, case,
        include_payload=response_has_payload,
        emit_boundary=present_result,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_alternate_sample_is_24_distinct_nonprimary_cases(case):
    assert len(CASES) == 24
    assert len({item.config_path for item in CASES}) == 24
    assert case.benchmark_index > 0
    assert case.config_path.is_file()
    assert case.threat_model == case.config["Attack"]["threat_model"]
    assert case.domain in str(case.config_path)
    assert _probe(case)["supported"] is True


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_alternate_sample_positive_boundary_evidence(case, tmp_path):
    event = _observe(
        tmp_path, case, response_has_payload=True, present_result=True,
    )
    assert event["step_index"] == case.step_index
    assert event["presented_to_model"] is True
    assert event["unknown_reasons"] == []
    if case.step.type == "environment":
        assert event["response_contains_injection"] is True
        assert event["access_call_status"] == "ok"
        assert event["call_ids"] == ["read-secondary-case"]
        assert event["match_basis"] in {
            "exact_locator", "collection_locator", "payload_probe",
        }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_alternate_sample_rejects_missing_payload(case, tmp_path):
    event = _observe(
        tmp_path, case, response_has_payload=False, present_result=True,
    )
    assert event["presented_to_model"] is False
    if case.step.type == "environment":
        assert event["response_contains_injection"] is False
        # Calling the right collection/entity is not proof that this attempt's
        # injected member survived filtering, pagination, or stale state.
        assert event["access_call_status"] == "ok"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_alternate_sample_separates_result_from_model_presentation(case, tmp_path):
    event = _observe(
        tmp_path, case, response_has_payload=True, present_result=False,
    )
    assert event["presented_to_model"] is (None if case.step.type != "environment" else False)
    if case.step.type == "environment":
        assert event["response_contains_injection"] is True
    else:
        assert event["unknown_reasons"] == ["message_boundary_unavailable"]


@pytest.mark.parametrize(
    ("domain", "expected_server"),
    [
        ("customer-service", "customer_service"),
        ("research", "Research"),
        ("travel", "travel-suite"),
    ],
)
def test_domain_patterns_use_actual_configured_mcp_namespace(domain, expected_server):
    case = next(
        item for item in CASES
        if item.domain == domain and item.threat_model == "indirect"
    )
    pattern = _runtime_pattern(case, _environment_probe(case))
    assert pattern["tool"].startswith(expected_server + ":")


def test_customer_service_batch_fallback_cannot_treat_write_echo_as_read(tmp_path):
    case = next(
        item for item in CASES
        if item.domain == "customer-service" and item.threat_model == "indirect"
    )
    probe = _environment_probe(case)
    payload = probe["payload_values"][0]
    path = tmp_path / "customer-service-write.jsonl"
    observer = FeedbackObserver(MCPEventSink(path, case.id))
    observer.register(probe)
    observer.observe_call(
        call_id="write-echo",
        server="customer_service",
        tool="update_case",
        arguments={},
        result={"content": [{"type": "text", "text": payload}], "isError": False},
    )
    observer.observe_messages_snapshot([{
        "role": "toolResult", "content": [{"type": "text", "text": payload}],
    }])
    observer.emit()

    event = _event(path)
    assert event["call_ids"] == []
    assert event["response_contains_injection"] is None
    assert event["presented_to_model"] is None
