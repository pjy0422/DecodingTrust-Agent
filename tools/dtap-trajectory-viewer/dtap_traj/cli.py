"""Command-line interface for DTAP trajectory export and explorer."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path

from .bundle import load_judge_results
from .parser import build_timeline, find_policy_trace, find_victim_mcp_events, find_victim_trace
from .tuning import PHASES, STATUSES, TuningArtifactError, create_tuning_trial, index_tuning_root, update_tuning_trial


def _first(root: Path, names: tuple[str, ...]) -> Path | None:
    directory = root if root.is_dir() else root.parent
    for name in names:
        direct = directory / name
        if direct.is_file():
            return direct
    for name in names:
        hits = sorted(directory.rglob(name))
        if hits:
            return hits[0]
    return None


def _guess_original(root: Path) -> Path | None:
    return _first(root, ("original-config.yaml", "original_config.yaml", "original/config.yaml"))


def _guess_submitted(root: Path) -> Path | None:
    return _first(root, ("submitted-config.yaml", "submitted_config.yaml", "candidate/config.yaml", "attack.yaml"))


def _guess_policy_prompt(root: Path) -> Path | None:
    return _first(root, ("policy-prompt.txt", "policy_prompt.txt"))


def _guess_episode_meta(root: Path) -> dict:
    manifest = _first(root, ("episode-manifest.json",))
    if manifest is None:
        return {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"trajectory_warnings": ["episode manifest is unreadable"]}
    episode_id = payload.get("episode_id") if isinstance(payload, dict) else None
    return {"episode_id": str(episode_id)} if episode_id else {}


def _guess_evaluation(root: Path) -> dict:
    directory = root if root.is_dir() else root.parent
    evaluation: dict = {}
    result = _first(directory, ("result.json",))
    if result is not None:
        try:
            payload = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if isinstance(payload, dict):
            for key in (
                "status",
                "evaluation_completed",
                "failure_class",
                "attack_success",
                "episode_status",
                "placement_applicable",
                "placement_covered",
                "placement_actions",
                "placements_verified",
            ):
                if key in payload:
                    evaluation[key] = payload[key]
    judges = load_judge_results(directory)
    if judges["available"]:
        evaluation["judges"] = judges
        evaluation["judge"] = judges["raw"] or judges["reward_firewall"]
    return evaluation


def _resolve_optional(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def _build(src: Path, args: argparse.Namespace) -> dict:
    explicit_victim = _resolve_optional(args.victim_trace)
    explicit_policy = _resolve_optional(args.policy_trace)
    if src.is_file() and "policy" in src.name.lower() and not explicit_policy:
        policy_trace = src
        victim_trace = explicit_victim
    else:
        victim_trace = explicit_victim or find_victim_trace(src)
        policy_trace = explicit_policy or (find_policy_trace(src) if src.is_dir() else None)
    submitted = _resolve_optional(args.submitted_yaml or args.yaml)
    original = _resolve_optional(args.original_yaml)
    policy_prompt = _resolve_optional(args.policy_prompt)
    if not args.no_yaml:
        submitted = submitted or _guess_submitted(src)
        original = original or _guess_original(src)
    policy_prompt = policy_prompt or _guess_policy_prompt(src)
    victim_mcp_events = _resolve_optional(args.victim_mcp_events) or (
        find_victim_mcp_events(src) if src.is_dir() else None
    )
    if victim_trace is None and policy_trace is None:
        raise FileNotFoundError(f"no victim or policy trace found under {src}")
    data = build_timeline(
        victim_trace,
        meta=_guess_episode_meta(src),
        policy_trace_path=policy_trace,
        policy_prompt_path=policy_prompt,
        original_yaml_path=original,
        submitted_yaml_path=submitted,
        victim_mcp_events_path=victim_mcp_events,
    )
    evaluation = _guess_evaluation(src)
    if evaluation:
        data["evaluation"] = evaluation
        if "judges" in evaluation:
            data["judges"] = evaluation["judges"]
            data["timeline"].append(
                {
                    "kind": "judge",
                    "text": json.dumps(evaluation["judges"], ensure_ascii=False, indent=2),
                }
            )
    return data


def _explorer_main(argv: list[str]) -> int:
    command = argv[0]
    parser = argparse.ArgumentParser(prog=f"dtap-traj {command}")
    parser.add_argument("path", help="artifact root containing episode bundles")
    parser.add_argument("--db", help="SQLite index path (default: <path>/.dtap-traj.sqlite3)")
    if command == "serve":
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8765)
        parser.add_argument(
            "--watch", action="store_true", help="continuously index completed/updated episode bundles"
        )
        parser.add_argument("--refresh-seconds", type=float, default=2.0)
        parser.add_argument("--open", action="store_true", help="open the explorer in a browser")
    args = parser.parse_args(argv[1:])
    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        parser.error(f"no such path: {root}")
    db_path = Path(args.db).expanduser().resolve() if args.db else root / ".dtap-traj.sqlite3"
    from .db import TrajectoryDB
    from .indexer import index_root

    db = TrajectoryDB(db_path)
    result = index_root(root, db)
    tuning = index_tuning_root(root, db)
    if command == "index":
        print(
            f"✓ indexed {result['scanned']} episodes ({result['updated']} updated), {tuning['scanned']} tuning trials ({tuning['updated']} updated) → {db_path}"
        )
        if tuning["errors"]:
            for error in tuning["errors"]:
                print(f"! skipped {error['path']}: {error['error']}", file=sys.stderr)
            return 2
        return 0
    import uvicorn

    from .server import create_app

    url = f"http://{args.host}:{args.port}"
    print(f"✓ DTAP Explorer: {url} ({result['scanned']} episodes, {tuning['scanned']} tuning trials indexed)")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    uvicorn.run(
        create_app(root, db_path=db_path, watch=args.watch, refresh_seconds=args.refresh_seconds),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def _tuning_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="dtap-traj tune")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    create = subparsers.add_parser("create", help="create a planned manual tuning trial")
    create.add_argument("root", help="experiment artifact root")
    create.add_argument("--phase", choices=PHASES, required=True)
    create.add_argument("--hardware", required=True, help="hardware profile JSON")
    create.add_argument("--model", required=True, help="model profile JSON")
    create.add_argument("--workload", required=True, help="workload profile JSON")
    create.add_argument("--software", required=True, help="software profile JSON")
    create.add_argument("--config", required=True, help="candidate runtime config JSON")
    create.add_argument("--trial-id")
    create.add_argument("--hypothesis")
    create.add_argument("--parent", action="append", default=[], help="parent trial ID; repeatable")
    create.add_argument("--tag", action="append", default=[], help="search tag; repeatable")

    update = subparsers.add_parser("update", help="record progress or finish a manual trial")
    update.add_argument("trial_dir")
    update.add_argument("--status", choices=STATUSES, required=True)
    update.add_argument("--metrics", help="measured metrics JSON")
    update.add_argument("--failure-class")
    update.add_argument("--observation")
    update.add_argument("--next-step")

    args = parser.parse_args(argv)
    try:
        if args.operation == "create":
            trial_dir = create_tuning_trial(
                args.root,
                phase=args.phase,
                hardware=args.hardware,
                model=args.model,
                workload=args.workload,
                software=args.software,
                config=args.config,
                trial_id=args.trial_id,
                hypothesis=args.hypothesis,
                parent_trial_ids=args.parent,
                tags=args.tag,
            )
            print(f"✓ created planned tuning trial → {trial_dir}")
        else:
            update_tuning_trial(
                args.trial_dir,
                status=args.status,
                metrics=args.metrics,
                failure_class=args.failure_class,
                observation=args.observation,
                next_step=args.next_step,
            )
            print(f"✓ updated tuning trial → {Path(args.trial_dir).expanduser().resolve()}")
    except (FileExistsError, OSError, TuningArtifactError) as exc:
        parser.error(str(exc))
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in {"serve", "index"}:
        return _explorer_main(raw)
    if raw and raw[0] == "tune":
        return _tuning_main(raw[1:])

    parser = argparse.ArgumentParser(
        prog="dtap-traj",
        description="View a DTAP victim trace, policy trace, and config diff together.",
    )
    parser.add_argument("path", help="run directory, victim trace, or policy stream-json")
    parser.add_argument("--victim-trace", help="explicit DTAP OpenClaw JSONL")
    parser.add_argument("--victim-mcp-events", help="explicit redacted OpenClaw MCP proxy event JSONL")
    parser.add_argument("--policy-trace", help="explicit Claude policy stream-json JSONL")
    parser.add_argument("--policy-prompt", help="policy instruction text shown as the first event")
    parser.add_argument("--original-yaml", help="original benchmark config.yaml")
    parser.add_argument("--submitted-yaml", help="policy-submitted config.yaml")
    parser.add_argument("-y", "--yaml", help="backward-compatible alias for --submitted-yaml")
    parser.add_argument("-o", "--out", default="trajectory.html", help="output HTML")
    parser.add_argument("-t", "--title", help="page title")
    parser.add_argument("--json", action="store_true", help="print viewer data as JSON")
    parser.add_argument("--open", action="store_true", help="open generated HTML")
    parser.add_argument("--no-yaml", action="store_true", help="disable YAML auto-discovery")
    args = parser.parse_args(raw)
    src = Path(args.path).expanduser().resolve()
    if not src.exists():
        parser.error(f"no such path: {src}")
    try:
        data = _build(src, args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    from .render import write_html

    output = Path(args.out).expanduser().resolve()
    write_html(data, output, title=args.title)
    print(
        f"✓ {len(data['policy_timeline'])} policy events, {len(data['timeline'])} victim events, {len(data['payloads'])} submitted payloads → {output}"
    )
    if args.open:
        webbrowser.open(output.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
