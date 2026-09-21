"""Thin, read-only integration around the pinned upstream ``dt_arms`` runtime.

The vendored subtree remains an upstream snapshot.  This module owns only the
two-stage boundary: discover a successful DT Arms candidate, merge its Attack
section into an isolated task copy, replay it with DTAP, and adapt the raw
red-team trajectory for the existing viewer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

import yaml

from .attempt_runner import AttemptResult, DtapAttemptRunner
from .candidate_config import AttemptWorkspace, CandidateConfigError, parse_candidate_with_dtap
from .integrity import BenchmarkIntegrityGuard, safe_copy_tree


SNAPSHOT_PATH = Path(__file__).with_name("dt_arms_snapshot.json")


class DtArmsIntegrationError(RuntimeError):
    """A DT Arms candidate or replay boundary is ambiguous or invalid."""


def snapshot_provenance() -> dict[str, Any]:
    value = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DtArmsIntegrationError("invalid DT Arms snapshot provenance")
    return value


def snapshot_content_sha256(root: Path | str) -> str:
    """Hash snapshot file names and contents while ignoring Python caches."""

    root = Path(root).resolve()
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    )
    for path in paths:
        relative = path.relative_to(root.parent).as_posix()
        content = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{content}  {relative}\n".encode())
    return digest.hexdigest()


def task_identity(task_dir: Path | str) -> dict[str, str]:
    task = Path(task_dir).resolve()
    parts = task.parts
    indices = [index for index, value in enumerate(parts) if value == "dataset"]
    if not indices:
        raise DtArmsIntegrationError("task is not below dataset/")
    suffix = parts[indices[-1] + 1 :]
    if len(suffix) != 5 or suffix[1] != "malicious":
        raise DtArmsIntegrationError("task path does not use the malicious dataset shape")
    domain, _, threat_model, risk_category, task_id = suffix
    return {
        "domain": domain,
        "type": "malicious",
        "threat_model": threat_model,
        "risk_category": risk_category,
        "task_id": task_id,
    }


def write_task_list(task_dir: Path | str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(task_identity(task_dir), ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def discover_generation_artifacts(root: Path | str) -> tuple[Path | None, Path | None]:
    root = Path(root).resolve()
    attacks = [path for path in root.rglob("attack_result_*.yaml") if path.is_file() and not path.is_symlink()]
    trajectories = [
        path
        for path in root.rglob("red_teaming_agent_*.json")
        if path.is_file() and not path.is_symlink()
    ]
    if len(attacks) > 1 or len(trajectories) > 1:
        raise DtArmsIntegrationError("ambiguous DT Arms generation artifacts")
    return (attacks[0] if attacks else None, trajectories[0] if trajectories else None)


def merge_generated_attack(source_config: Path | str, generated_attack: Path | str) -> dict[str, Any]:
    try:
        source = yaml.safe_load(Path(source_config).read_text(encoding="utf-8"))
        overlay = yaml.safe_load(Path(generated_attack).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DtArmsIntegrationError("cannot parse DT Arms candidate") from exc
    if not isinstance(source, dict) or not isinstance(source.get("Attack"), dict):
        raise DtArmsIntegrationError("source task has no Attack section")
    if not isinstance(overlay, dict) or set(overlay) != {"Attack"} or not isinstance(overlay["Attack"], dict):
        raise DtArmsIntegrationError("DT Arms candidate must contain only an Attack section")
    base_attack = source["Attack"]
    proposed = overlay["Attack"]
    for field in ("risk_category", "threat_model", "malicious_goal"):
        if proposed.get(field) != base_attack.get(field):
            raise DtArmsIntegrationError(f"DT Arms candidate changed {field}")
    turns = proposed.get("attack_turns")
    if not isinstance(turns, list) or not turns:
        raise DtArmsIntegrationError("DT Arms candidate has no attack turns")
    merged = copy.deepcopy(source)
    merged["Attack"]["attack_turns"] = copy.deepcopy(turns)
    try:
        parse_candidate_with_dtap(merged)
    except CandidateConfigError as exc:
        raise DtArmsIntegrationError("DT Arms candidate is not a valid DTAP config") from exc
    return merged


def materialize_replay_workspace(
    *,
    source_task_dir: Path | str,
    merged_config: Mapping[str, Any],
    runtime_root: Path | str,
    output_root: Path | str,
) -> AttemptWorkspace:
    source = Path(source_task_dir).resolve()
    runtime = Path(runtime_root).resolve()
    output = Path(output_root).resolve()
    manifest = BenchmarkIntegrityGuard.capture(source)
    identity = task_identity(source)
    destination = runtime.joinpath(
        "dataset",
        identity["domain"],
        "malicious",
        identity["threat_model"],
        identity["risk_category"],
        identity["task_id"],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_copy_tree(source, destination, expected_manifest=manifest)
    _materialize_guest_setup_helper(source, runtime)
    (destination / "config.yaml").write_text(
        yaml.safe_dump(dict(merged_config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    BenchmarkIntegrityGuard.verify(source, manifest)
    return AttemptWorkspace(
        attempt_index=1,
        attempt_dir=runtime,
        task_dir=destination,
        config_path=destination / "config.yaml",
        output_root=output,
    )


def _materialize_guest_setup_helper(source: Path, runtime_root: Path) -> None:
    """Copy the trusted helper required by isolated Windows/macOS task trees."""

    indices = [index for index, part in enumerate(source.parts) if part == "dataset"]
    if not indices:
        raise DtArmsIntegrationError("source task must be below dataset/")
    dataset_index = indices[-1]
    suffix = source.parts[dataset_index:]
    if len(suffix) < 2 or suffix[1] not in {"windows", "macos"}:
        return
    platform = suffix[1]
    repository_root = Path(*source.parts[:dataset_index])
    helper = repository_root / "dt_arena" / "utils" / platform / "env_setup.py"
    if helper.is_symlink() or not helper.is_file() or not stat.S_ISREG(helper.stat().st_mode):
        raise DtArmsIntegrationError(f"trusted {platform} setup helper is unavailable")
    target = runtime_root / "dt_arena" / "utils" / platform / "env_setup.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(helper, target)


def _message(event_type: str, message_id: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": event_type,
        "source": "dt-arms-upstream",
        "message": {"id": message_id, "role": event_type, "content": blocks},
    }


def normalize_dt_arms_trajectory(source: Path | str, destination: Path | str) -> Path:
    """Create viewer-compatible JSONL while retaining the raw trajectory separately."""

    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    steps = payload.get("attack_trajectory") if isinstance(payload, dict) else None
    if not isinstance(steps, list):
        raise DtArmsIntegrationError("DT Arms trajectory has no attack_trajectory")
    events: list[dict[str, Any]] = []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            continue
        role = str(step.get("role") or "unknown")
        reasoning = step.get("reasoning")
        action = step.get("action")
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        if role == "attacker":
            blocks: list[dict[str, Any]] = []
            if isinstance(reasoning, str) and reasoning.strip():
                blocks.append({"type": "thinking", "thinking": reasoning})
            if isinstance(action, str) and action.strip():
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": f"dt-arms-{index}",
                        "name": f"dt_arms__{action}",
                        "input": metadata,
                    }
                )
            if blocks:
                events.append(_message("assistant", f"dt-arms-attacker-{index}", blocks))
            continue
        state = step.get("state")
        text = json.dumps(state, ensure_ascii=False, sort_keys=True) if not isinstance(state, str) else state
        if metadata:
            text = f"{text}\n\n{json.dumps(metadata, ensure_ascii=False, sort_keys=True)}"
        events.append(
            _message(
                "user",
                f"dt-arms-{role}-{index}",
                [{"type": "text", "text": f"[{role}] {text}"}],
            )
        )
    result = payload.get("attack_result") if isinstance(payload, dict) else None
    events.append(
        {
            "type": "result",
            "source": "dt-arms-upstream",
            "policy_engine": "dt-arms-upstream",
            "model": (payload.get("attack_config") or {}).get("agent_model"),
            "success": bool((result or {}).get("success")),
        }
    )
    target = Path(destination)
    target.write_text(
        "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return target


def export_replay_artifacts(
    result: AttemptResult,
    *,
    artifact_root: Path | str,
    attempt_root: Path | str,
) -> int:
    root = Path(artifact_root)
    attempt = Path(attempt_root)
    count = 0
    for source, name in (
        (result.trajectory_path, "victim-trajectory.json"),
        (result.mcp_events_path, "victim-mcp-events.jsonl"),
    ):
        if source is None or not source.is_file():
            continue
        for destination in (attempt / name, root / name):
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        count += 1
    if result.judge_result is not None:
        rendered = json.dumps(dict(result.judge_result), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        for destination in (attempt / "judge-result.json", root / "judge-result.json"):
            destination.write_text(rendered, encoding="utf-8")
        count += 1
    verdict = json.dumps({"attack_success": result.attack_success}, sort_keys=True) + "\n"
    for destination in (attempt / "judge-verdict.json", root / "judge-verdict.json"):
        destination.write_text(verdict, encoding="utf-8")
    return count + 1


async def replay_candidate(
    *,
    source_task_dir: Path,
    merged_config: Mapping[str, Any],
    artifacts_dir: Path,
    dtap_root: Path,
    python_executable: str,
    victim_model: str,
    victim_agent_type: str,
    victim_max_turns: int,
    timeout_seconds: float,
    port_range_start: int,
) -> tuple[AttemptResult, int]:
    attempts = artifacts_dir / "attempts"
    attempt = attempts / "attempt-0001"
    attempt.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="dt-arms-replay-") as temporary:
        workspace = materialize_replay_workspace(
            source_task_dir=source_task_dir,
            merged_config=merged_config,
            runtime_root=Path(temporary),
            output_root=attempt,
        )
        runner = DtapAttemptRunner(
            max_parallel=1,
            agent_type=victim_agent_type,
            model=victim_model,
            max_turns=victim_max_turns,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
            dtap_root=dtap_root,
            port_range_start=port_range_start,
        )
        result = await runner.run(workspace)
    return result, export_replay_artifacts(result, artifact_root=artifacts_dir, attempt_root=attempt)


def episode_id() -> str:
    return f"dt-arms-{uuid.uuid4().hex}"


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
