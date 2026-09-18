"""Trusted, in-memory probes for adaptive victim feedback.

Probe values never belong in retained event artifacts.  OpenClaw consumes them
only to emit bounded booleans and step references after a victim run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


_FINANCE_TOOLS = {
    "inject_html_news": ("finance:browse_news",),
    "inject_html_article": ("finance:browse_article",),
    "inject_html_quote": ("finance:browse_stock",),
    "inject_html_portfolio": ("finance:browse_portfolio",),
    "inject_html_market": ("finance:browse_markets", "finance:browse_stock_list"),
    "inject_html_options": ("finance:browse_options",),
    "inject_html_analysis": ("finance:browse_stock",),
    "inject_html_general": (
        "finance:browse_stock", "finance:browse_news", "finance:browse_article",
        "finance:browse_portfolio", "finance:browse_options",
        "finance:browse_markets", "finance:browse_stock_list",
    ),
    "create_news": ("finance:browse_news", "finance:browse_article"),
    "create_comment": ("finance:browse_article",),
    "inject_news": ("finance:browse_news", "finance:browse_article"),
}

_FINANCE_PAYLOAD_FIELDS = {
    "inject_html_news": ("prompt",),
    "inject_html_article": ("prompt",),
    "inject_html_quote": ("prompt",),
    "inject_html_portfolio": ("prompt",),
    "inject_html_market": ("prompt",),
    "inject_html_options": ("prompt",),
    "inject_html_analysis": ("prompt",),
    "inject_html_general": ("prompt",),
    "create_news": ("title", "summary"),
    "create_comment": ("content",),
    "inject_news": (
        "title_prefix", "title_suffix", "title_replace", "summary_prefix",
        "summary_suffix", "summary_inject", "content_inject",
    ),
}

# Every enabled Linux M6 mutator is explicitly classified here.  The first
# present field is the minimum payload witness for broad read/list responses;
# exact locator matching remains a separate fact.
_LINUX_PAYLOAD_FIELDS = {
    "customer-service-injection": {
        "cs_case_description_inject": ("description",),
        "cs_case_note_inject": ("body",),
        "cs_case_note_inject_all_by_order": ("body",),
        "cs_order_metadata_inject": ("metadata_updates",),
        "cs_order_metadata_inject_adaptive": (
            "customer_note", "delivery_instructions", "gift_message", "category",
        ),
    },
    "salesforce-injection": {
        name: ("description", "name", "first_name") for name in (
            "inject_account", "inject_call", "inject_case", "inject_contact",
            "inject_lead", "inject_meeting", "inject_note", "inject_opportunity",
            "inject_task",
        )
    },
    "legal-injection": {
        "inject_case_note": ("content",),
        "inject_extra_entry": ("content", "entry_name"),
        "inject_fake_document": ("content", "summary", "filename"),
        "inject_into_document": ("content",),
        "inject_into_matter": ("content",),
        "modify_matter": (
            "strategy_replace", "strategy_inject", "strategy_prefix",
            "strategy_suffix", "notes_replace", "notes_inject", "notes_prefix",
            "notes_suffix", "settlement_inject",
        ),
    },
    "finance-injection": _FINANCE_PAYLOAD_FIELDS,
    "os-filesystem-injection": {
        "inject_append": ("content",), "inject_directory": ("dir_path",),
        "inject_executable": ("script_content",), "inject_file": ("content",),
        "inject_symlink": ("target_path",),
    },
    "gmail-injection": {"inject_email": ("body", "subject")},
    "slack-injection": {
        "inject_fake_announcement": ("announcement_body", "announcement_title"),
        "inject_malicious_link_message": ("cover_text", "malicious_link"),
        "inject_prompt_in_slack": ("injection_payload",),
        "inject_slack_dm": ("message",), "inject_slack_message": ("message",),
        "inject_slack_thread_reply": ("message",),
    },
    "calendar-injection": {
        "inject_calendar_event": ("description", "title"),
        "inject_fake_meeting_invite": ("malicious_link", "title"),
        "inject_false_event_info": ("wrong_location", "wrong_time"),
        "inject_prompt_in_calendar": ("injection_payload",),
    },
    "zoom-injection": {
        "inject_fake_zoom_invite": ("malicious_join_link", "topic"),
        "inject_prompt_in_zoom": ("injection_payload",),
        "inject_zoom_meeting": ("agenda", "topic"),
        "inject_zoom_with_leaked_password": ("password", "topic"),
    },
    "googledocs-injection": {
        "inject_comment": ("content",), "inject_into_document": ("content",),
        "inject_shared_document": ("content", "title"),
        "update_document": ("content", "title"),
    },
    "whatsapp-injection": {
        "create_contact": ("notes", "first_name", "company"),
        "send_whatsapp_message": ("body",),
    },
    "google-form-injection": {
        "inject_bulk_form_responses": ("responses_list",),
        "inject_form_response": ("responses",),
        "inject_malicious_form_response": ("injection_payload",),
        "inject_pii_in_form": ("fake_ssn", "fake_email", "fake_name"),
        "inject_prompt_in_form": ("injection_payload",),
    },
    "snowflake-injection": {
        "inject_exfiltration_target": ("confidential_records", "label"),
        "inject_pii_data": ("pii_records",),
        "inject_prompt_in_data": ("injection_payload",),
        "inject_sensitive_query_result": ("sensitive_data",),
        "inject_snowflake_data": ("data",),
    },
    "databricks-injection": {
        "inject_databricks_data": ("data",),
        "inject_exfiltration_target": ("confidential_records", "label"),
        "inject_job_with_data_access": ("action", "job_name"),
        "inject_pii_data": ("pii_records",),
        "inject_prompt_in_notebook": ("injection_payload",),
    },
    "paypal-injection": {
        "inject_fraudulent_request": ("urgency_message",),
        "inject_payment_request": ("message",),
        "inject_prompt_in_transaction": ("injection_payload",),
        "inject_refund_scenario": ("reason",),
        "inject_transaction": ("note",),
    },
    "ecommerce-injection": {"inject_review": ("detail", "title")},
    "custom-website-injection": {"update_html_content": ("html",)},
    "travel-injection": {
        "inject_accommodation": ("description", "house_rules", "name"),
        "inject_flight": ("flight_number",),
        "inject_restaurant": ("name", "cuisines"),
        "inject_review": ("content",),
    },
    "telecom-injection": {
        name: ("value", "updates", "data") for name in (
            "account_partial_inject", "bill_partial_inject",
            "calendar_event_partial_inject", "customer_partial_inject",
            "email_partial_inject", "forum_post_inject",
            "forum_post_partial_inject", "forum_thread_inject",
            "forum_thread_partial_inject", "order_partial_inject",
            "outage_partial_inject", "sms_partial_inject", "ticket_partial_inject",
        )
    },
    "atlassian-injection": {
        "inject_confluence_page": ("content", "title"),
        "inject_jira_comment": ("comment_body",),
        "inject_jira_issue": ("description", "summary"),
        "inject_sensitive_data": ("content",),
    },
    "terminal-injection": {
        "inject_file": ("content",), "inject_readme": ("content",),
        "inject_todo_list": ("content",),
    },
    "research-injection": {
        "inject_fake_paper": ("abstract", "title"),
        "inject_html_paper_metadata": ("modified_abstract", "modified_title"),
        "inject_paper_notes": ("content",), "inject_readme": ("content",),
    },
    "telegram-injection": {
        "inject_contact": ("notes", "first_name"), "inject_message": ("body",),
    },
    "github-injection": {
        "inject_comment": ("body",), "inject_commit": ("content", "message"),
        "inject_issue": ("body", "title"),
        "inject_prompt_chain": ("issue_payload", "comment_payload"),
        "inject_pull_request": ("body", "title"),
    },
    "hospital-env-injection": {"inject_test_result": ("test_result",)},
}

_VICTIM_SERVERS = {
    server: (server.removesuffix("-injection"),)
    for server in _LINUX_PAYLOAD_FIELDS
}
_VICTIM_SERVERS.update({
    "hospital-env-injection": ("hospital",),
    "os-filesystem-injection": ("OS-filesystem", "filesystem"),
    "custom-website-injection": ("browser",),
    "terminal-injection": ("terminal", "OS-filesystem", "filesystem"),
    "research-injection": ("research", "browser", "OS-filesystem", "filesystem"),
})

# Payload-only correlation is still restricted to victim reads. A service-wide
# wildcard would let a write tool that echoes its input masquerade as evidence
# that the victim consumed the injected environment content.
_PAYLOAD_READ_PATTERNS = {
    "customer-service-injection": (
        "customer_service:get_*", "customer_service:list_*",
        "customer_service:find_*", "customer_service:search_*",
        "customer-service:get_*", "customer-service:list_*",
        "customer-service:find_*", "customer-service:search_*",
    ),
    "salesforce-injection": ("salesforce:get_*", "salesforce:list_*", "salesforce:search_*", "salesforce:meta_*"),
    "legal-injection": ("legal:get_*", "legal:list_*", "legal:search*", "legal:fetch*", "legal:analyze_*"),
    "finance-injection": _FINANCE_TOOLS["inject_html_general"],
    "os-filesystem-injection": ("OS-filesystem:read_*", "OS-filesystem:list_*", "OS-filesystem:search_*", "filesystem:read_*", "filesystem:list_*", "filesystem:search_*"),
    "gmail-injection": ("gmail:get_*", "gmail:list_*", "gmail:search_*", "gmail:find_message"),
    "slack-injection": ("slack:channels_history", "slack:conversations_history", "slack:inbox", "slack:dm_feed", "slack:list_*"),
    "calendar-injection": ("calendar:get_event", "calendar:list_events", "calendar:search_events"),
    "zoom-injection": ("zoom:meetings_get", "zoom:meetings_list", "zoom:invitations_list", "zoom:chat_list", "zoom:transcripts_*", "zoom:notes_*"),
    "googledocs-injection": ("googledocs:get_*", "googledocs:list_*", "googledocs:search_*"),
    "whatsapp-injection": ("whatsapp:get_*", "whatsapp:list_*", "whatsapp:search_*"),
    "google-form-injection": ("google-form:get_*", "google-form:list_*", "google-form:search_*"),
    "snowflake-injection": ("snowflake:product-search",),
    "databricks-injection": ("databricks:databricks-vector-search",),
    "paypal-injection": ("paypal:get_*", "paypal:list_*", "paypal:search_*"),
    "ecommerce-injection": ("ecommerce:get_*", "ecommerce:list_*", "ecommerce:search_*"),
    "custom-website-injection": ("browser:*",),
    "travel-injection": (
        "travel-suite:query_*", "travel-suite:manager_query_*",
        "travel:query_*", "travel:manager_query_*",
    ),
    "telecom-injection": ("telecom:query_*",),
    "atlassian-injection": ("atlassian:get*", "atlassian:search*", "atlassian:fetch"),
    "terminal-injection": ("terminal:execute_command", "OS-filesystem:read_*", "filesystem:read_*"),
    "research-injection": (
        "Research:search_*", "Research:fetch_*", "Research:list_*",
        "Research:execute_command", "research:search_*", "research:fetch_*",
        "research:list_*", "research:execute_command", "browser:*",
        "OS-filesystem:read_*", "filesystem:read_*",
    ),
    "telegram-injection": ("telegram:telegram_get_*", "telegram:telegram_list_*", "telegram:telegram_search_*"),
    "github-injection": ("github:get_*", "github:list_*", "github:search_*"),
    "hospital-env-injection": (
        "HospitalClient:get_patient_status", "HospitalClient:list_patients",
        "HospitalClient:request_*", "hospital:get_patient_status",
        "hospital:list_patients", "hospital:request_*",
    ),
}

SUPPORTED_FEEDBACK_TOOLS = {
    server: frozenset(tools) for server, tools in _LINUX_PAYLOAD_FIELDS.items()
}
FEEDBACK_PAYLOAD_FIELDS = {
    server: dict(tools) for server, tools in _LINUX_PAYLOAD_FIELDS.items()
}

_CUSTOMER_SERVICE_CASE_FEEDBACK_TOOLS = frozenset({
    "cs_case_description_inject", "cs_case_note_inject",
    "cs_case_note_inject_all_by_order",
})
_CUSTOMER_SERVICE_ORDER_FEEDBACK_TOOLS = frozenset({
    "cs_order_metadata_inject", "cs_order_metadata_inject_adaptive",
})
_FINANCE_HTML_FEEDBACK_TOOLS = frozenset({
    "inject_html_analysis", "inject_html_article", "inject_html_general",
    "inject_html_market", "inject_html_news", "inject_html_options",
    "inject_html_portfolio", "inject_html_quote",
})
_FINANCE_NEWS_FEEDBACK_TOOLS = frozenset({
    "create_comment", "create_news", "inject_news",
})
_EXACT_FEEDBACK_HANDLER_TOOLS = {
    "customer-service-injection": (
        _CUSTOMER_SERVICE_CASE_FEEDBACK_TOOLS
        | _CUSTOMER_SERVICE_ORDER_FEEDBACK_TOOLS
    ),
    "finance-injection": (
        _FINANCE_HTML_FEEDBACK_TOOLS | _FINANCE_NEWS_FEEDBACK_TOOLS
    ),
}
for _server, _tools in _EXACT_FEEDBACK_HANDLER_TOOLS.items():
    if _tools != SUPPORTED_FEEDBACK_TOOLS[_server]:
        raise RuntimeError(f"feedback handler registry drift: {_server}")


def _article_id(result: Mapping[str, Any]) -> int | None:
    text = result.get("result")
    if not isinstance(text, str):
        return None
    match = re.search(r"(?im)^Article ID:\s*([0-9]+)\s*$", text)
    return int(match.group(1)) if match else None


def _result_value(result: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Find a receipt identity without assuming one MCP serialization shape."""
    wanted = set(keys)

    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in wanted and child not in (None, ""):
                    return child
            for child in value.values():
                found = visit(child)
                if found not in (None, ""):
                    return found
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                found = visit(child)
                if found not in (None, ""):
                    return found
        elif isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
            return visit(decoded)
        return None

    return visit(result)


def _exact_locator_patterns(
    server: str, tool: str, kwargs: Mapping[str, Any], result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Map a write receipt to victim reads only when their arguments prove identity."""
    if server == "customer-service-injection":
        if tool in _CUSTOMER_SERVICE_CASE_FEEDBACK_TOOLS:
            case_id = _result_value(result, ("case_id", "case_ids"))
            if isinstance(case_id, Sequence) and not isinstance(case_id, (str, bytes, bytearray)):
                case_id = next((value for value in case_id if value not in (None, "")), None)
            if case_id not in (None, ""):
                exact = [
                    {"tool": f"{victim}:{name}", "fields": {"case_id": case_id}}
                    for victim in ("customer_service", "customer-service")
                    for name in ("get_case", "get_case_activities")
                ]
                # Batch injectors can return several case IDs. Preserve payload
                # correlation for the remaining IDs without calling them exact.
                return exact + [
                    {"tool": victim_tool, "fields": {}, "locator_exact": False}
                    for victim_tool in _PAYLOAD_READ_PATTERNS[server]
                ]
        if tool in _CUSTOMER_SERVICE_ORDER_FEEDBACK_TOOLS:
            order_id = _result_value(result, ("order_id", "order_ids")) or kwargs.get("order_id")
            if isinstance(order_id, Sequence) and not isinstance(order_id, (str, bytes, bytearray)):
                order_id = next((value for value in order_id if value not in (None, "")), None)
            if order_id not in (None, ""):
                return [
                    {"tool": f"{victim}:get_order", "fields": {"order_id": order_id}}
                    for victim in ("customer_service", "customer-service")
                ]

    if server == "legal-injection":
        if tool in {"inject_into_matter", "modify_matter"} and kwargs.get("matter_id"):
            return [{"tool": "legal:get_matter", "fields": {"matter_id": kwargs["matter_id"]}}]
        if tool == "inject_into_document" and kwargs.get("doc_id"):
            return [
                {"tool": f"legal:{name}", "fields": {"doc_id": kwargs["doc_id"]}}
                for name in ("get_document", "analyze_document")
            ]
        if tool == "inject_fake_document":
            doc_id = _result_value(result, ("document_id", "doc_id"))
            if doc_id:
                return [
                    {"tool": f"legal:{name}", "fields": {"doc_id": doc_id}}
                    for name in ("get_document", "analyze_document")
                ]
        if tool == "inject_extra_entry" and kwargs.get("target_tool"):
            return [{
                "tool": f"legal:{kwargs['target_tool']}", "fields": {},
                "locator_exact": True,
            }]

    if server == "travel-injection":
        if tool in {"inject_accommodation", "inject_restaurant"} and kwargs.get("city"):
            return [
                {
                    "tool": f"{victim}:query_{tool.removeprefix('inject_')}",
                    "fields": {"city": kwargs["city"]},
                }
                for victim in ("travel-suite", "travel")
            ]
        if tool == "inject_flight" and all(
            kwargs.get(key) for key in ("origin_city_name", "dest_city_name", "flight_date")
        ):
            return [
                {
                    "tool": f"{victim}:query_flight",
                    "fields": {
                        "departure": kwargs["origin_city_name"],
                        "destination": kwargs["dest_city_name"],
                        "departure_date": kwargs["flight_date"],
                    },
                }
                for victim in ("travel-suite", "travel")
            ]
        if tool == "inject_review" and kwargs.get("entity_type") in {
            "accommodation", "restaurant",
        } and kwargs.get("city"):
            return [
                {
                    "tool": f"{victim}:query_{kwargs['entity_type']}",
                    "fields": {"city": kwargs["city"]},
                }
                for victim in ("travel-suite", "travel")
            ]

    telecom = {
        "customer_partial_inject": ("query_customer", "id", "customer_id"),
        "account_partial_inject": ("query_account", "customer_id", "customer_id"),
        "bill_partial_inject": ("query_bill", "customer_id", "customer_id"),
        "calendar_event_partial_inject": ("query_calendar_event", "id", "event_id"),
        "email_partial_inject": ("query_email", "id", "email_id"),
        "order_partial_inject": ("query_order", "order_id", "order_id"),
        "outage_partial_inject": ("query_outage", "area", "area"),
        "sms_partial_inject": ("query_sms", "id", "sms_id"),
        "ticket_partial_inject": ("query_ticket", "id", "ticket_id"),
        "forum_thread_partial_inject": ("query_forum_thread", "id", "thread_id"),
        "forum_post_partial_inject": ("query_forum_post", "id", "post_id"),
    }
    if server == "telecom-injection" and tool in telecom:
        victim_tool, source_field, victim_field = telecom[tool]
        if kwargs.get(source_field) not in (None, ""):
            return [{
                "tool": f"telecom:{victim_tool}",
                "fields": {victim_field: kwargs[source_field]},
            }]
    if server == "telecom-injection" and tool in {"forum_thread_inject", "forum_post_inject"}:
        kind = "thread" if tool == "forum_thread_inject" else "post"
        identifier = _result_value(result, (f"{kind}_id", "id"))
        if identifier in (None, "") and isinstance(kwargs.get("data"), Mapping):
            identifier = kwargs["data"].get("id")
        if identifier not in (None, ""):
            return [{
                "tool": f"telecom:query_forum_{kind}",
                "fields": {f"{kind}_id": identifier},
            }]

    if server == "salesforce-injection":
        modules = {
            "inject_lead": "Leads", "inject_contact": "Contacts",
            "inject_account": "Accounts", "inject_opportunity": "Opportunities",
            "inject_call": "Calls", "inject_meeting": "Meetings",
            "inject_task": "Tasks", "inject_note": "Notes", "inject_case": "Cases",
        }
        module = modules.get(tool)
        record_id = _result_value(result, ("id", "record_id"))
        if module and record_id:
            return [
                {"tool": "salesforce:get_record", "fields": {
                    "module_name": module, "record_id": record_id,
                }},
                {"tool": "salesforce:get_entry", "fields": {
                    "module_name": module, "id": record_id,
                }},
            ]

    if server == "googledocs-injection":
        document_id = kwargs.get("document_id") or _result_value(
            result, ("document_id", "documentId", "id")
        )
        if document_id:
            names = ("get_comments",) if tool == "inject_comment" else ("get_document",)
            return [
                {"tool": f"googledocs:{name}", "fields": {"document_id": document_id}}
                for name in names
            ]

    if server == "slack-injection":
        channel = kwargs.get("channel_name")
        if channel:
            return [{
                "tool": "slack:channels_history", "fields": {"channel_name": channel},
            }]
        conversation = _result_value(result, ("conversation_id", "channel_id"))
        if conversation:
            return [{
                "tool": "slack:conversations_history",
                "fields": {"conversation_id": conversation},
            }]

    if server == "os-filesystem-injection":
        source_field = {
            "inject_file": "file_path", "inject_append": "file_path",
            "inject_executable": "file_path", "inject_symlink": "link_path",
        }.get(tool)
        if source_field and kwargs.get(source_field):
            return [
                {"tool": f"{victim}:read_file", "fields": {"filepath": kwargs[source_field]}}
                for victim in _VICTIM_SERVERS[server]
            ]

    if server == "research-injection" and tool in {
        "inject_fake_paper", "inject_html_paper_metadata",
    } and kwargs.get("paper_id"):
        return [
            {
                "tool": f"{victim}:fetch_arxiv_paper_html",
                "fields": {"paper_id": kwargs["paper_id"]},
            }
            for victim in ("Research", "research")
        ]

    if server == "calendar-injection":
        event_id = kwargs.get("event_id") or _result_value(result, ("event_id", "id"))
        if event_id:
            return [{"tool": "calendar:get_event", "fields": {"event_id": event_id}}]

    if server == "zoom-injection":
        meeting_id = kwargs.get("meeting_id") or _result_value(result, ("meeting_id", "id"))
        if meeting_id:
            return [{"tool": "zoom:meetings_get", "fields": {"meeting_id": meeting_id}}]

    if server == "whatsapp-injection":
        if tool == "send_whatsapp_message" and kwargs.get("phone_number"):
            return [{
                "tool": "whatsapp:get_whatsapp_chat",
                "fields": {"phone_number": kwargs["phone_number"]},
            }]
        if tool == "create_contact":
            contact_id = _result_value(result, ("contact_id", "id"))
            if contact_id:
                return [{
                    "tool": "whatsapp:get_contact", "fields": {"contact_id": contact_id},
                }]

    if server == "github-injection" and kwargs.get("owner") and kwargs.get("repo"):
        common = {"owner": kwargs["owner"], "repo": kwargs["repo"]}
        if tool == "inject_comment" and kwargs.get("number") not in (None, ""):
            target = "pull" if str(kwargs.get("target_type", "")).lower() in {"pull", "pr"} else "issue"
            return [{
                "tool": f"github:list_{target}_comments",
                "fields": {**common, "number": kwargs["number"]},
            }]
        if tool in {"inject_issue", "inject_prompt_chain"}:
            number = _result_value(result, ("number", "issue_number"))
            if number not in (None, ""):
                return [{"tool": "github:get_issue", "fields": {**common, "number": number}}]
        if tool == "inject_pull_request":
            number = _result_value(result, ("number", "pull_number"))
            if number not in (None, ""):
                return [{"tool": "github:get_pull", "fields": {**common, "number": number}}]
        if tool == "inject_commit":
            sha = _result_value(result, ("sha", "commit_sha"))
            if sha:
                return [{"tool": "github:get_commit", "fields": {**common, "sha": sha}}]

    return []


def _locator_match_basis(server: str, pattern: Mapping[str, Any]) -> str:
    """Describe locator precision independently from payload correlation."""
    tool = str(pattern.get("tool", ""))
    if pattern.get("locator_exact", bool(pattern.get("fields"))) is not True:
        return "payload_probe"
    if server == "travel-injection" or tool in {
        "slack:channels_history", "whatsapp:get_whatsapp_chat",
        "finance:browse_news", "finance:browse_markets", "finance:browse_stock_list",
    }:
        return "collection_locator"
    return "exact_locator"


def _finance_locator_patterns(
    tool: str, kwargs: Mapping[str, Any], result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    symbol = kwargs.get("symbol")
    article_id = kwargs.get("article_id")
    if tool == "create_news" and article_id is None:
        article_id = _article_id(result)
    patterns: list[dict[str, Any]] = []
    for victim_tool in _FINANCE_TOOLS[tool]:
        fields: dict[str, Any] = {}
        if victim_tool in {
            "finance:browse_stock", "finance:browse_news", "finance:browse_article",
            "finance:browse_portfolio", "finance:browse_options",
        } and symbol not in (None, ""):
            fields["symbol"] = str(symbol).upper()
        if victim_tool == "finance:browse_article":
            if article_id is None:
                continue
            fields["article_id"] = article_id
        if tool in _FINANCE_HTML_FEEDBACK_TOOLS and victim_tool == "finance:browse_stock":
            page = kwargs.get("page")
            if page in {"quote", "analysis"}:
                fields["section"] = page
        patterns.append({"tool": victim_tool, "fields": fields})
    return patterns


def build_environment_feedback_probe(
    injection: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Return one explicit probe or an unsupported classification.

    This deliberately has no name-based fallback.  Each supported mutator must
    be listed with its payload and victim-read semantics.
    """
    server = str(injection.get("server_name", "")).lower()
    tool = str(injection.get("tool_name", ""))
    step_index = injection.get("feedback_step_index")
    base = {
        "step_index": step_index,
        "injection_type": "environment",
        "supported": False,
        "unsupported_reason": "adapter_unsupported",
    }
    if not isinstance(step_index, int) or isinstance(step_index, bool) or step_index < 0:
        return {**base, "unsupported_reason": "identity_unavailable"}
    server_tools = _LINUX_PAYLOAD_FIELDS.get(server)
    if server_tools is None or tool not in server_tools:
        return base
    kwargs = injection.get("kwargs")
    if not isinstance(kwargs, Mapping):
        return {**base, "unsupported_reason": "identity_unavailable"}
    payload_values = (
        [
            value for field in server_tools[tool]
            if isinstance((value := kwargs.get(field)), str) and value
        ]
        if server == "finance-injection"
        else _first_string_values(kwargs, server_tools[tool])
    )
    if not payload_values:
        return {**base, "unsupported_reason": "identity_unavailable"}
    patterns = (
        _finance_locator_patterns(tool, kwargs, result)
        if server == "finance-injection"
        else _exact_locator_patterns(server, tool, kwargs, result) or [
            {"tool": victim_tool, "fields": {}, "locator_exact": False}
            for victim_tool in _PAYLOAD_READ_PATTERNS[server]
        ]
    )
    if not patterns:
        return {**base, "unsupported_reason": "identity_unavailable"}
    patterns = [
        {**pattern, "match_basis": _locator_match_basis(server, pattern)}
        for pattern in patterns
    ]
    return {
        "step_index": step_index,
        "injection_type": "environment",
        "supported": True,
        "victim_patterns": patterns,
        "payload_values": payload_values,
        "locator_observable": any(
            pattern.get("locator_exact", bool(pattern.get("fields")))
            for pattern in patterns
        ),
        "match_basis": "payload_probe",
    }


def build_static_feedback_probes(attack_config: Any) -> list[dict[str, Any]]:
    """Build prompt/tool/skill probes using the flattened attack-step identity.

    These probes are consumed only inside the victim process.  Their raw
    payload witnesses are never written to the retained MCP event stream.
    """
    probes: list[dict[str, Any]] = []
    step_index = 0
    turns = getattr(attack_config, "attack_turns", ()) if attack_config else ()
    for turn in turns:
        for step in getattr(turn, "attack_steps", ()):
            injection_type = str(getattr(step, "type", ""))
            if injection_type == "environment":
                step_index += 1
                continue
            content = getattr(step, "content", None)
            base = {
                "step_index": step_index,
                "injection_type": injection_type,
                "supported": False,
                "unsupported_reason": "adapter_unsupported",
            }
            if injection_type not in {"prompt", "tool", "skill"}:
                probes.append(base)
            elif not isinstance(content, str) or not content:
                probes.append({**base, "unsupported_reason": "identity_unavailable"})
            elif injection_type == "tool":
                target = getattr(step, "injected_tool", None)
                if not isinstance(target, str) or not target:
                    probes.append({**base, "unsupported_reason": "identity_unavailable"})
                else:
                    probes.append({
                        **base,
                        "supported": True,
                        "payload_values": [content],
                        "target_tool": target,
                        "presentation_source": "available_tools",
                        "match_basis": "payload_probe",
                    })
            else:
                probes.append({
                    **base,
                    "supported": True,
                    "payload_values": [content],
                    "presentation_source": (
                        "submitted_prompt" if injection_type == "prompt"
                        else "system_prompt"
                    ),
                    "match_basis": "payload_probe",
                })
            step_index += 1
    return probes


def _first_string_values(kwargs: Mapping[str, Any], fields: Sequence[str]) -> list[str]:
    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, Mapping):
            result: list[str] = []
            for child in value.values():
                result.extend(strings(child))
            return result
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            result = []
            for child in value:
                result.extend(strings(child))
            return result
        return []

    for field in fields:
        found = strings(kwargs.get(field))
        if found:
            return found
    return []


def value_contains_all(value: Any, probes: Sequence[str]) -> bool:
    """Compare raw string leaves in memory; callers retain only the boolean result.

    Serializing a tool result before matching changes newlines, quotes, and other
    escapes.  Real benchmark payloads are commonly multiline YAML scalars, so a
    byte-for-byte witness must be searched in the original string leaves.
    """
    if not probes:
        return False

    def contains(candidate: Any, probe: str) -> bool:
        if isinstance(candidate, str):
            return probe in candidate
        if isinstance(candidate, Mapping):
            return any(contains(child, probe) for child in candidate.values())
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            return any(contains(child, probe) for child in candidate)
        return False

    return all(contains(value, probe) for probe in probes)
