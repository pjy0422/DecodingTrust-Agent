"""Trusted M3 rendering and materialization of ephemeral DTAP configs."""

from __future__ import annotations

import copy
import hashlib
import shutil
import stat
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .actions import ValidatedAttackStep
from .dtap_compat import load_dtap_api
from .integrity import BenchmarkIntegrityGuard, BenchmarkManifest, IntegrityError, safe_copy_tree


class CandidateConfigError(ValueError):
    """Safe trusted-boundary failure; its raw text is never policy-visible."""


@dataclass(frozen=True)
class ParsedCandidate:
    task_config: Any
    agent_config: Any
    attack_config: Any


@dataclass(frozen=True)
class ValidatedCandidateConfig:
    config: dict[str, Any]
    canonical_steps: tuple[ValidatedAttackStep, ...]
    parsed: Any


@dataclass(frozen=True)
class AttemptWorkspace:
    attempt_index: int
    attempt_dir: Path
    task_dir: Path
    config_path: Path
    output_root: Path


def cleanup_episode_root(
    episode_root: Path | str,
    *,
    managed_root: Path | str,
) -> None:
    """Delete one bounded episode tree without permitting broad targets."""

    target_input = Path(episode_root)
    root = Path(managed_root).resolve()
    if target_input.is_symlink():
        raise CandidateConfigError("episode cleanup target may not be a symlink")
    target = target_input.resolve()
    if target == root or not target.is_relative_to(root):
        raise CandidateConfigError("episode cleanup target escapes the managed root")
    if target.exists():
        shutil.rmtree(target)


def _step_payload(step: ValidatedAttackStep) -> dict[str, Any]:
    payload = step.to_dict()
    payload.pop("turn_id", None)
    return payload


def render_candidate_config(
    base_config: Mapping[str, Any],
    steps: Sequence[ValidatedAttackStep],
) -> dict[str, Any]:
    if not isinstance(base_config, Mapping):
        raise CandidateConfigError("base config must be a mapping")
    rendered = copy.deepcopy(dict(base_config))
    attack = rendered.get("Attack")
    if not isinstance(attack, dict):
        raise CandidateConfigError("Attack config must be a mapping")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        if not isinstance(step, ValidatedAttackStep):
            raise CandidateConfigError("all actions must be validated")
        grouped[step.dtap_turn_id].append(_step_payload(step))
    attack["attack_turns"] = [{"turn_id": turn_id, "attack_steps": grouped[turn_id]} for turn_id in sorted(grouped)]
    return rendered


def parse_candidate_with_dtap(config: Mapping[str, Any]) -> ParsedCandidate:
    """Exercise the actual DTAP YAML entry points without touching source data."""

    try:
        api = load_dtap_api()
        with tempfile.TemporaryDirectory(prefix="slime-dtap-m3-parse-") as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
            task = api.TaskConfig.from_yaml(str(path))
            agent = api.AgentConfig.from_yaml(str(path))
            attack = api.AttackConfig.from_yaml(str(path))
    except Exception as exc:
        raise CandidateConfigError("DTAP config parsing failed") from exc
    if task is None or agent is None or attack is None:
        raise CandidateConfigError("DTAP config parsing returned an empty section")
    return ParsedCandidate(task, agent, attack)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def canonical_steps_from_dtap_config(parsed: Any) -> tuple[ValidatedAttackStep, ...]:
    attack = _get(parsed, "attack_config", parsed)
    if isinstance(attack, Mapping) and "Attack" in attack:
        attack = attack["Attack"]
    result: list[ValidatedAttackStep] = []
    for turn in _get(attack, "attack_turns", ()) or ():
        turn_id = _get(turn, "turn_id")
        for raw in _get(turn, "attack_steps", ()) or ():
            step_type = _get(raw, "type")
            common = {
                "type": step_type,
                "mode": _get(raw, "mode"),
                "content": _get(raw, "content"),
                "injected_tool": _get(raw, "injected_tool"),
                "injection_mcp_tool": _get(raw, "injection_mcp_tool"),
                "skill_name": _get(raw, "skill_name"),
                "row": _get(raw, "row"),
            }
            if step_type in {"prompt", "environment"}:
                common["turn_id"] = turn_id
            kwargs = _get(raw, "kwargs")
            if step_type == "environment":
                common["kwargs"] = dict(kwargs or {})
            if step_type not in {"prompt", "tool", "environment", "skill"}:
                raise CandidateConfigError("DTAP returned an unsupported attack step")
            result.append(ValidatedAttackStep(**common))
    return tuple(result)


def _semantic_signature(steps: Iterable[ValidatedAttackStep]) -> tuple[Any, ...]:
    grouped: dict[int, list[tuple[tuple[str, Any], ...]]] = defaultdict(list)
    for step in steps:
        payload = step.to_dict()
        payload.pop("turn_id", None)
        normalized = tuple(sorted((key, _freeze(value)) for key, value in payload.items()))
        grouped[step.dtap_turn_id].append(normalized)
    return tuple((turn, tuple(grouped[turn])) for turn in sorted(grouped))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def validate_candidate_config(
    config: Mapping[str, Any],
    *,
    expected_steps: Sequence[ValidatedAttackStep],
) -> ValidatedCandidateConfig:
    try:
        dumped = yaml.safe_dump(dict(config), sort_keys=False)
        loaded = yaml.safe_load(dumped)
    except Exception as exc:
        raise CandidateConfigError("candidate YAML serialization failed") from exc
    if not isinstance(loaded, dict):
        raise CandidateConfigError("candidate YAML root must be a mapping")
    parsed = parse_candidate_with_dtap(loaded)
    actual = canonical_steps_from_dtap_config(parsed)
    expected = tuple(expected_steps)
    if _semantic_signature(actual) != _semantic_signature(expected):
        raise CandidateConfigError("candidate semantic round trip mismatch")
    return ValidatedCandidateConfig(copy.deepcopy(loaded), expected, parsed)


def _assert_safe_source_tree(source: Path) -> None:
    if source.is_symlink():
        raise CandidateConfigError("source task directory may not be a symlink")
    for path in source.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise CandidateConfigError("source task tree contains a symlink")
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise CandidateConfigError("source task tree contains a special file")


def _dataset_suffix(source: Path) -> Path:
    indices = [index for index, part in enumerate(source.parts) if part == "dataset"]
    if not indices:
        raise CandidateConfigError("source task must be below a dataset directory")
    return Path(*source.parts[indices[-1] :])


def _materialize_guest_setup_helper(source: Path, attempt_dir: Path) -> None:
    """Provide the one trusted helper expected by legacy guest setup scripts.

    Windows/macOS setup.sh files derive PROJECT_ROOT by walking upward from the
    task. An isolated candidate deliberately lives outside the DTAP checkout,
    so recreate only the expected helper path instead of symlinking or copying
    the repository into the policy-owned attempt tree.
    """
    indices = [index for index, part in enumerate(source.parts) if part == "dataset"]
    if not indices:
        raise CandidateConfigError("source task must be below a dataset directory")
    dataset_index = indices[-1]
    suffix = source.parts[dataset_index:]
    if len(suffix) < 2 or suffix[1] not in {"windows", "macos"}:
        return
    platform = suffix[1]
    repository_root = Path(*source.parts[:dataset_index])
    helper = repository_root / "dt_arena" / "utils" / platform / "env_setup.py"
    if helper.is_symlink() or not helper.is_file() or not stat.S_ISREG(helper.stat().st_mode):
        raise CandidateConfigError(f"trusted {platform} setup helper is unavailable")
    target = attempt_dir / "dt_arena" / "utils" / platform / "env_setup.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(helper, target)


def materialize_attempt_dir(
    *,
    source_task_dir: Path | str,
    episode_root: Path | str,
    attempt_index: int,
    steps: Sequence[ValidatedAttackStep],
    candidate_validator: Callable[..., ValidatedCandidateConfig] | None = None,
    source_manifest: BenchmarkManifest | None = None,
) -> AttemptWorkspace:
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 1:
        raise CandidateConfigError("attempt_index must be a positive integer")
    source_input = Path(source_task_dir)
    if source_input.is_symlink():
        raise CandidateConfigError("source task directory may not be a symlink")
    source = source_input.resolve()
    if not source.is_dir() or not (source / "config.yaml").is_file():
        raise CandidateConfigError("source task directory is invalid")
    _assert_safe_source_tree(source)
    manifest = source_manifest or BenchmarkIntegrityGuard.capture(source)
    BenchmarkIntegrityGuard.verify(source, manifest)

    root_input = Path(episode_root)
    if root_input.exists() and root_input.is_symlink():
        raise CandidateConfigError("episode root may not be a symlink")
    root = root_input.resolve()
    root.mkdir(parents=True, exist_ok=True)
    attempt_dir = root / f"attempt-{attempt_index:04d}"
    if attempt_dir.exists():
        raise CandidateConfigError("attempt directory already exists")
    destination = attempt_dir / _dataset_suffix(source)
    original_hash = hashlib.sha256((source / "config.yaml").read_bytes()).hexdigest()

    try:
        destination.parent.mkdir(parents=True, exist_ok=False)
        try:
            safe_copy_tree(source, destination, expected_manifest=manifest)
        except IntegrityError as exc:
            raise CandidateConfigError("source task tree failed integrity validation") from exc
        _materialize_guest_setup_helper(source, attempt_dir)
        raw = yaml.safe_load((source / "config.yaml").read_bytes()) or {}
        rendered = render_candidate_config(raw, steps)
        validator = candidate_validator or validate_candidate_config
        validator(rendered, expected_steps=tuple(steps))
        (destination / "config.yaml").write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
        BenchmarkIntegrityGuard.verify(source, manifest)
        if hashlib.sha256((source / "config.yaml").read_bytes()).hexdigest() != original_hash:
            raise CandidateConfigError("source config changed during materialization")
    except Exception:
        if attempt_dir.is_dir():
            shutil.rmtree(attempt_dir)
        raise

    return AttemptWorkspace(
        attempt_index=attempt_index,
        attempt_dir=attempt_dir,
        task_dir=destination,
        config_path=destination / "config.yaml",
        output_root=attempt_dir / "results",
    )
