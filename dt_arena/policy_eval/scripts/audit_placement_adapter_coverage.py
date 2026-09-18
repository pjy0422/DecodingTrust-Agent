"""Audit every enabled DTAP injection tool against the M6 adapter registry."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import yaml

EXCLUDED = frozenset()
GUEST_PLATFORM_SERVERS = frozenset({"windows-injection", "macos-injection"})


def _decorated_tools(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tools: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not isinstance(func, ast.Attribute) or func.attr != "tool":
                continue
            name = node.name
            for keyword in decorator.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    name = str(keyword.value.value)
            tools.add(name)
    return tools


def audit(dtap_root: Path) -> dict:
    from dt_arena.src.env_verification import NON_PLACEMENT_TOOLS, SUPPORTED_PLACEMENT_TOOLS

    config = yaml.safe_load((dtap_root / "dt_arena/config/injection_mcp.yaml").read_text(encoding="utf-8"))
    rows = []
    missing = []
    base = dtap_root / "dt_arena/injection_mcp_server"
    for server in config["servers"]:
        name = str(server["name"])
        if not server.get("enabled") or name in EXCLUDED:
            continue
        source = base / str(server["path"])
        if not source.is_file():
            missing.append(name)
            continue
        tool_source = source
        if name == "finance-injection":
            tool_source = dtap_root / "dt_arena/mcp_server/finance/server/injection_mcp.py"
        discovered = _decorated_tools(tool_source)
        supported = set(SUPPORTED_PLACEMENT_TOOLS.get(name, frozenset()))
        non_placement = set(NON_PLACEMENT_TOOLS.get(name, frozenset()))
        overlap = sorted(supported & non_placement)
        unclassified = sorted(discovered - supported - non_placement)
        stale_supported = sorted(supported - discovered)
        stale_non_placement = sorted(non_placement - discovered)
        rows.append(
            {
                "server": name,
                "discovered_tools": len(discovered),
                "verified": sorted(discovered & supported),
                "not_applicable": sorted(discovered & non_placement),
                "unsupported": unclassified,
                "registry_overlap": overlap,
                "stale_supported_registry_entries": stale_supported,
                "stale_non_placement_registry_entries": stale_non_placement,
                "stale_registry_entries": sorted(stale_supported + stale_non_placement),
            }
        )
    stale = [row for row in rows if row["stale_registry_entries"]]
    unclassified = [row for row in rows if row["unsupported"]]
    overlaps = [row for row in rows if row["registry_overlap"]]
    return {
        "status": ("passed" if not missing and not stale and not unclassified and not overlaps else "failed"),
        "excluded": sorted(EXCLUDED),
        "enabled_servers": len(rows),
        "missing_implementations": missing,
        "verified_mutators": sum(len(row["verified"]) for row in rows),
        "classified_non_placement_tools": sum(len(row["not_applicable"]) for row in rows),
        "unclassified_tools": sum(len(row["unsupported"]) for row in rows),
        # Kept for compatibility with existing release-report consumers.
        "unsupported_mutators": sum(len(row["unsupported"]) for row in rows),
        "guest_platforms": {
            row["server"]: {
                "verified": len(row["verified"]),
                "not_applicable": len(row["not_applicable"]),
                "unsupported": len(row["unsupported"]),
            }
            for row in rows
            if row["server"] in GUEST_PLATFORM_SERVERS
        },
        "servers": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.dtap_root.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
