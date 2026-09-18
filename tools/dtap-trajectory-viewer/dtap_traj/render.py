"""Render the combined trajectory as one self-contained HTML file."""

from __future__ import annotations

import html
import json
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent / "assets"


def _asset(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


def _embed_json(value: object) -> str:
    # Escaping every literal '<' is valid JSON (unlike ad-hoc \! escapes) and
    # prevents script close tags and HTML comments from being recognized.
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def render_html(data: dict, title: str | None = None) -> str:
    victim = data.get("timeline") or []
    policy = data.get("policy_timeline") or []
    calls = sum(event.get("kind") == "tool_call" for event in victim + policy)
    trace_name = Path(data.get("trace") or data.get("policy_trace") or "trace").name
    bits = [
        f"<code>{html.escape(trace_name)}</code>",
        f"{len(policy)} policy events",
        f"{len(victim)} victim events",
        f"{calls} tool calls",
        f"{len(data.get('payloads') or [])} submitted payloads",
    ]
    if data.get("episode_id"):
        bits.append(f"episode {html.escape(str(data['episode_id']))}")
    evaluation = data.get("evaluation") or {}
    if evaluation.get("status"):
        bits.append(f"evaluation {html.escape(str(evaluation['status']))}")
    if evaluation.get("attack_success") is not None:
        bits.append(f"attack success {str(bool(evaluation['attack_success'])).lower()}")
    comparison = data.get("config_comparison")
    if comparison:
        bits.append("configs identical" if comparison["identical"] else "configs changed")
    page_title = title or f"DTAP Trajectories — {trace_name}"
    return (
        _asset("template.html")
        .replace("__CSS__", _asset("style.css"))
        .replace("__JS__", _asset("viewer.js"))
        .replace("__TITLE__", html.escape(page_title))
        .replace("__META__", " · ".join(bits))
        .replace("__DATA__", _embed_json(data))
    )


def write_html(data: dict, out_path: str | Path, title: str | None = None) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(data, title=title), encoding="utf-8")
    return path
