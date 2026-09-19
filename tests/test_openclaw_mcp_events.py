import json
from types import SimpleNamespace

import pytest

from agent.openclaw.src.mcp_proxy import FeedbackObserver, MCPEventSink, MCPProxyServer
from agent.openclaw.src.agent import OpenClawAgent
from agent.openclaw.src.plugin_generator import StaticPluginGenerator


def test_openclaw_evaluation_profile_disables_interactive_bootstrap(
    tmp_path, monkeypatch
):
    agent = object.__new__(OpenClawAgent)
    agent.runtime_config = SimpleNamespace(
        model="openai/test-model",
        agent_kwargs={},
    )
    agent.config = SimpleNamespace(system_prompt="DTAP victim prompt")
    agent._profile_dir = str(tmp_path / "profile")
    agent._proxy_urls = {}
    agent._generated_plugin_id = None
    agent._skill_temp_dir = None
    agent._debug = False

    monkeypatch.setattr(
        "agent.openclaw.src.agent.populate_openclaw_profile_auth",
        lambda *args, **kwargs: None,
    )

    agent._configure_openclaw_with_proxies()

    profile_dir = tmp_path / "profile"
    config = json.loads((profile_dir / "openclaw.json").read_text(encoding="utf-8"))
    assert config["agents"]["defaults"]["skipBootstrap"] is True
    assert (profile_dir / "workspace" / "AGENTS.md").read_text(
        encoding="utf-8"
    ) == "DTAP victim prompt"
    assert not (profile_dir / "workspace" / "BOOTSTRAP.md").exists()


class _Hook:
    async def wrap(self, context, dispatch):
        return {
            "content": [{"type": "text", "text": "private backend result"}],
            "isError": False,
        }


def test_proxy_event_sink_materializes_zero_call_trajectory(tmp_path):
    path = tmp_path / "empty.mcp-events.jsonl"
    MCPEventSink(path, "episode-zero")
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_proxy_event_sink_records_raw_and_redacted_call_and_result(tmp_path):
    path = tmp_path / "episode.mcp-events.jsonl"
    sink = MCPEventSink(path, "episode-public-1")
    proxy = MCPProxyServer(
        name="filesystem",
        transport="http",
        url="http://127.0.0.1:1/mcp",
        hook_manager=_Hook(),
        event_sink=sink,
    )
    secret = "do-not-store-this-secret"
    sandbox_path = "/private/sandbox/task/file.txt"

    result = await proxy._call_tool(
        "read_file", {"token": secret, "path": sandbox_path, "limit": 3}
    )

    assert result["isError"] is False
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == [
        "tool.started", "tool.completed"
    ]
    assert events[0]["episode_id"] == "episode-public-1"
    assert events[0]["schema_version"] == 2
    assert events[0]["arguments"] == {
        "token": secret,
        "path": sandbox_path,
        "limit": 3,
    }
    assert events[0]["arguments_redacted"]["path"]["length"] == len(sandbox_path)
    assert events[0]["arguments_redacted"]["token"]["sha256"] != secret
    assert len(events[0]["arguments_digest"]) == 64
    assert events[1]["is_error"] is False
    assert events[1]["result"] == {
        "content": [{"type": "text", "text": "private backend result"}],
        "isError": False,
    }
    assert len(events[1]["result_digest"]) == 64


@pytest.mark.asyncio
async def test_proxy_event_sink_keeps_raw_values_separate_from_redacted_shape(tmp_path):
    path = tmp_path / "episode.mcp-events.jsonl"
    sink = MCPEventSink(path, "episode-public-2")
    proxy = MCPProxyServer(
        name="filesystem", transport="http", url="http://127.0.0.1:1/mcp",
        hook_manager=_Hook(), event_sink=sink,
    )
    secrets = {
        "api_key": "sk-secret-api-key",
        "authorization": "Bearer secret-bearer-token",
        "mcp_credential": "opaque-mcp-capability",
        "sandbox_path": "/private/task/sandbox/secret.txt",
        "payload": "raw sensitive prompt payload",
    }

    await proxy._call_tool("read_file", secrets)

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["arguments"] == secrets
    assert events[1]["result"]["content"][0]["text"] == "private backend result"
    assert events[0]["arguments_redacted"].keys() == secrets.keys()
    assert all(
        set(shape) == {"type", "length", "sha256"}
        for shape in events[0]["arguments_redacted"].values()
    )


def test_event_sink_completion_marker_is_idempotent(tmp_path):
    path = tmp_path / "complete.mcp-events.jsonl"
    sink = MCPEventSink(path, "episode-complete")

    sink.complete()
    sink.complete()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["type"] for event in events] == ["trace.completed"]


def test_event_sink_does_not_claim_completeness_after_write_failure(tmp_path):
    path = tmp_path / "incomplete.mcp-events.jsonl"
    sink = MCPEventSink(path, "episode-incomplete")
    sink._healthy = False
    sink.complete()
    assert path.read_text() == ""


@pytest.mark.asyncio
async def test_feedback_observer_correlates_result_and_model_without_raw_values(tmp_path):
    payload = "policy-owned-finance-payload"
    path = tmp_path / "feedback.mcp-events.jsonl"
    sink = MCPEventSink(path, "episode-feedback")
    observer = FeedbackObserver(sink)
    observer.register({
        "step_index": 3,
        "supported": True,
        "victim_patterns": [{
            "tool": "finance:browse_article",
            "fields": {"symbol": "NVDA", "article_id": 17},
        }],
        "payload_values": [payload],
    })

    observer.observe_call(
        call_id="internal-call",
        server="finance",
        tool="browse_article",
        arguments={"symbol": "NVDA", "article_id": 17},
        result={"content": [{"type": "text", "text": payload}], "isError": False},
    )
    observer.observe_messages_snapshot([{
        "role": "toolResult",
        "content": [{"type": "text", "text": payload}],
    }])
    observer.emit()
    sink.complete()

    raw = path.read_text(encoding="utf-8")
    assert payload not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    observed = events[-2]
    assert observed["type"] == "feedback.observed"
    assert observed["step_index"] == 3
    assert observed["locator_targeted"] is True
    assert observed["response_contains_injection"] is True
    assert observed["presented_to_model"] is True
    assert events[-1]["type"] == "trace.completed"


def test_feedback_observer_keeps_unsupported_probe_unknown(tmp_path):
    path = tmp_path / "unsupported.mcp-events.jsonl"
    sink = MCPEventSink(path, "episode-unsupported")
    observer = FeedbackObserver(sink)
    observer.register({
        "step_index": 0,
        "supported": False,
        "unsupported_reason": "adapter_unsupported",
    })
    observer.emit()
    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["locator_targeted"] is None
    assert event["response_contains_injection"] is None
    assert event["unknown_reasons"] == ["adapter_unsupported"]


def test_feedback_observer_proves_static_provider_presentation_without_raw_values(tmp_path):
    path = tmp_path / "static-feedback.mcp-events.jsonl"
    sink = MCPEventSink(path, "episode-static")
    observer = FeedbackObserver(sink)
    probes = (
        (0, "prompt", "submitted_prompt", "prompt witness", {}),
        (1, "tool", "available_tools", "tool witness", {"target_tool": "gmail:search_emails"}),
        (2, "skill", "system_prompt", "skill witness", {}),
        (3, "tool", "available_tools", "missing witness", {"target_tool": "gmail:search_emails"}),
    )
    for index, kind, source, payload, extra in probes:
        observer.register({
            "step_index": index, "injection_type": kind, "supported": True,
            "presentation_source": source, "payload_values": [payload], **extra,
        })

    observer.observe_runtime_event({
        "type": "prompt.submitted", "data": {"prompt": "task\nprompt witness"},
    })
    observer.observe_runtime_event({
        "type": "context.compiled",
        "data": {
            "systemPrompt": "base\nskill witness",
            "availableTools": [{
                "name": "gmail__search_emails", "description": "normal\ntool witness",
            }],
        },
    })
    observer.emit()

    raw = path.read_text(encoding="utf-8")
    assert all(witness not in raw for witness in (
        "prompt witness", "tool witness", "skill witness", "missing witness",
    ))
    events = [json.loads(line) for line in raw.splitlines()]
    assert [event["presented_to_model"] for event in events] == [True, True, True, False]
    assert all(event["access_call_status"] == "not_applicable" for event in events)
    assert all(event["response_contains_injection"] is None for event in events)


def test_tool_presentation_identity_does_not_match_description_substrings():
    matches = FeedbackObserver._tool_entry_matches

    assert matches("gmail:search_emails", "gmail:search_emails") is True
    assert matches("gmail__search_emails", "gmail:search_emails") is True
    assert matches(
        '{"name":"gmail__search_emails","description":"safe"}',
        "gmail:search_emails",
    ) is True
    assert matches(
        "description mentions search_emails but belongs to another tool",
        "gmail:search_emails",
    ) is False
    assert matches(
        {"name": "calendar__search_emails", "description": "gmail:search_emails"},
        "gmail:search_emails",
    ) is False


def test_static_feedback_is_unknown_without_provider_boundary(tmp_path):
    path = tmp_path / "static-missing-boundary.jsonl"
    sink = MCPEventSink(path, "episode-static-missing")
    observer = FeedbackObserver(sink)
    observer.register({
        "step_index": 0, "injection_type": "skill", "supported": True,
        "presentation_source": "system_prompt", "payload_values": ["private"],
    })
    observer.emit()
    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["presented_to_model"] is None
    assert event["unknown_reasons"] == ["message_boundary_unavailable"]


def test_provider_input_observation_is_raw_free_and_authoritative(tmp_path):
    path = tmp_path / "provider-feedback.jsonl"
    sink = MCPEventSink(path, "episode-provider")
    observer = FeedbackObserver(sink)
    observer.register({
        "step_index": 0,
        "injection_type": "environment",
        "supported": True,
        "payload_values": ["private payload"],
    })
    observer.observe_provider_input({
        "schema_version": 1,
        "type": "provider.input.observed",
        "step_index": 0,
        "boundary_observed": True,
        "presented": True,
    })
    observer.emit()
    raw = path.read_text(encoding="utf-8")
    assert "private payload" not in raw
    event = json.loads(raw)
    assert event["presented_to_model"] is True
    assert event["presentation_observation_source"] == "provider_request"
    assert "message_boundary_unavailable" not in event["unknown_reasons"]


def test_generated_plugin_observes_llm_input_without_retaining_payload(tmp_path):
    generator = StaticPluginGenerator(extensions_dir=str(tmp_path / "extensions"))
    output = tmp_path / "index.ts"
    generator._write_index_ts(
        str(tmp_path),
        [],
        feedback_probe_path=str(tmp_path / "private-probes.json"),
        provider_observation_path=str(tmp_path / "observations.jsonl"),
    )
    source = output.read_text(encoding="utf-8")
    assert 'api.on("llm_input"' in source
    assert 'type: "provider.input.observed"' in source
    assert "payload_values" in source
    assert "presented:" in source
    assert 'target.replace(":", "_")' in source
    assert 'message?.role === "toolResult"' in source


@pytest.mark.asyncio
async def test_openclaw_cleanup_removes_ephemeral_raw_probe_registry(tmp_path):
    probe = tmp_path / "feedback-probes.json"
    probe.write_text('[{"payload_values":["private"]}]', encoding="utf-8")
    agent = OpenClawAgent.__new__(OpenClawAgent)
    agent._feedback_probe_file = str(probe)
    agent._proxy_manager = None
    agent._profile_dir = str(tmp_path / "missing-profile")
    agent.reset_conversation = lambda: None
    agent._cleanup_temp_resources = lambda: None

    await OpenClawAgent.cleanup(agent)

    assert not probe.exists()
