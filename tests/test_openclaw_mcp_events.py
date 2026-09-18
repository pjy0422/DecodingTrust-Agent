import json

import pytest

from agent.openclaw.src.mcp_proxy import FeedbackObserver, MCPEventSink, MCPProxyServer


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
async def test_proxy_event_sink_records_redacted_call_and_result(tmp_path):
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
    raw = path.read_text(encoding="utf-8")
    assert secret not in raw
    assert sandbox_path not in raw
    assert "private backend result" not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    assert [event["type"] for event in events] == [
        "tool.started", "tool.completed"
    ]
    assert events[0]["episode_id"] == "episode-public-1"
    assert events[0]["arguments"]["path"]["length"] == len(sandbox_path)
    assert len(events[0]["arguments_digest"]) == 64
    assert events[1]["is_error"] is False
    assert len(events[1]["result_digest"]) == 64


@pytest.mark.asyncio
async def test_proxy_event_sink_never_persists_secret_classes(tmp_path):
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

    raw = path.read_text(encoding="utf-8")
    assert all(secret not in raw for secret in secrets.values())
    assert "private backend result" not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    assert events[0]["arguments"].keys() == secrets.keys()
    assert all(
        set(shape) == {"type", "length", "sha256"}
        for shape in events[0]["arguments"].values()
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
