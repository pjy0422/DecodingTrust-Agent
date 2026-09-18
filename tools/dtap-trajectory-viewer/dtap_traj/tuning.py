"""Artifact contract and indexing helpers for manual performance tuning trials."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

TUNING_SCHEMA_VERSION = 1
PHASES = ("sglang", "megatron", "integrated")
STATUSES = (
    "planned",
    "running",
    "success",
    "oom",
    "timeout",
    "nccl_error",
    "engine_crash",
    "invalid_config",
    "infra_error",
)
PROFILE_FILES = ("hardware.json", "model.json", "workload.json", "software.json")
TRIAL_FILES = (
    "tuning-manifest.json",
    *PROFILE_FILES,
    "config.json",
    "metrics.json",
    "result.json",
)
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "password",
    "private_key",
    "secret",
    "token",
}


class TuningArtifactError(ValueError):
    """Raised when a tuning artifact violates the versioned contract."""


def _validate_trial_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TuningArtifactError("trial_id must be a non-empty string")
    value = value.strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    if any(character not in allowed for character in value):
        raise TuningArtifactError("trial_id may contain only letters, digits, '-', '_', and '.'")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise TuningArtifactError(f"{field} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _read_object(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise TuningArtifactError(f"missing {path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TuningArtifactError(f"unreadable {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise TuningArtifactError(f"{path.name} must contain a JSON object")
    _reject_secrets(value, path.name)
    return value


def _reject_secrets(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SECRET_KEYS or normalized.endswith(("_api_key", "_auth_token", "_password")):
                raise TuningArtifactError(f"{location} contains forbidden credential field {key!r}")
            _reject_secrets(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{location}[{index}]")
    elif isinstance(value, str) and "://" in value:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise TuningArtifactError(f"{location} contains a URL with embedded credentials")


def _canonical_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _label(profile: dict[str, Any], fallback: str) -> str:
    for key in ("label", "name", "model_name", "accelerator_model"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _validate_metrics(metrics: dict[str, Any]) -> None:
    for key in (
        "wall_time_s",
        "gpu_seconds",
        "peak_vram_gb",
        "rollout_tok_s",
        "train_tok_s",
        "step_time_s",
        "trainable_rollouts",
        "trainable_tokens",
        "repeats",
    ):
        value = _number(metrics.get(key))
        if value is not None and value < 0:
            raise TuningArtifactError(f"metrics.json field {key!r} must be nonnegative")
    for key in ("wait_ratio", "infra_failure_rate"):
        value = _number(metrics.get(key))
        if value is not None and not 0 <= value <= 1:
            raise TuningArtifactError(f"metrics.json field {key!r} must be between 0 and 1")
    durations = metrics.get("durations_s", {})
    if not isinstance(durations, dict):
        raise TuningArtifactError("metrics.json field 'durations_s' must be an object")
    for key, raw in durations.items():
        value = _number(raw)
        if value is None or value < 0:
            raise TuningArtifactError(f"duration {key!r} must be a nonnegative number")


def _artifact_inventory(path: Path) -> list[dict[str, Any]]:
    artifacts = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        try:
            relative = candidate.relative_to(path)
            size = candidate.stat().st_size
        except OSError:
            continue
        artifacts.append({"path": relative.as_posix(), "size_bytes": size})
        if len(artifacts) >= 1000:
            break
    return artifacts


def _objective(phase: str, metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    name = metrics.get("objective_name")
    value = _number(metrics.get("objective_value"))
    if isinstance(name, str) and name.strip() and value is not None:
        return name.strip(), value

    if phase == "sglang" and (value := _number(metrics.get("rollout_tok_s"))) is not None:
        return "rollout_tok_s", value
    if phase == "megatron" and (value := _number(metrics.get("train_tok_s"))) is not None:
        return "train_tok_s", value

    gpu_seconds = _number(metrics.get("gpu_seconds"))
    trainable_tokens = _number(metrics.get("trainable_tokens"))
    if gpu_seconds and trainable_tokens is not None:
        return "trainable_tokens_per_gpu_second", trainable_tokens / gpu_seconds
    trainable_rollouts = _number(metrics.get("trainable_rollouts"))
    if gpu_seconds and trainable_rollouts is not None:
        return "trainable_rollouts_per_gpu_hour", trainable_rollouts * 3600 / gpu_seconds
    return None, None


def discover_tuning_dirs(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return []
    candidates = {path.parent.resolve() for path in root.rglob("tuning-manifest.json")}
    if (root / "tuning-manifest.json").is_file():
        candidates.add(root)
    return sorted(candidates)


def load_tuning_bundle(path: str | Path, *, include_artifacts: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    manifest = _read_object(path / "tuning-manifest.json")
    version = manifest.get("schema_version")
    if version != TUNING_SCHEMA_VERSION:
        raise TuningArtifactError(f"unsupported tuning schema {version!r}; expected {TUNING_SCHEMA_VERSION}")
    phase = manifest.get("phase")
    if phase not in PHASES:
        raise TuningArtifactError(f"phase must be one of {', '.join(PHASES)}")
    manifest["trial_id"] = _validate_trial_id(manifest.get("trial_id"))
    manifest["parent_trial_ids"] = _string_list(manifest.get("parent_trial_ids", []), "parent_trial_ids")
    manifest["tags"] = _string_list(manifest.get("tags", []), "tags")
    if manifest.get("hypothesis") is not None and not isinstance(manifest["hypothesis"], str):
        raise TuningArtifactError("hypothesis must be a string or null")

    profiles = {name.removesuffix(".json"): _read_object(path / name) for name in PROFILE_FILES}
    config = _read_object(path / "config.json")
    metrics = _read_object(path / "metrics.json", required=False)
    _validate_metrics(metrics)
    result = _read_object(path / "result.json", required=False)
    status = result.get("status", "planned")
    if status not in STATUSES:
        raise TuningArtifactError(f"status must be one of {', '.join(STATUSES)}")
    for field in ("failure_class", "observation", "next_step"):
        if result.get(field) is not None and not isinstance(result[field], str):
            raise TuningArtifactError(f"{field} must be a string or null")
    bundle = {
        "manifest": manifest,
        "profiles": profiles,
        "config": config,
        "metrics": metrics,
        "result": {**result, "status": status},
    }
    if include_artifacts:
        bundle["artifacts"] = _artifact_inventory(path)
    return bundle


def _source_mtime_ns(path: Path) -> int:
    values: list[int] = []
    for name in TRIAL_FILES:
        try:
            values.append((path / name).stat().st_mtime_ns)
        except OSError:
            pass
    return max(values, default=0)


def extract_tuning_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    bundle = load_tuning_bundle(path, include_artifacts=False)
    manifest = bundle["manifest"]
    profiles = bundle["profiles"]
    config = bundle["config"]
    metrics = bundle["metrics"]
    result = bundle["result"]
    phase = manifest["phase"]
    objective_name, objective_value = _objective(phase, metrics)

    return {
        "trial_id": manifest["trial_id"].strip(),
        "phase": phase,
        "status": result["status"],
        "failure_class": result.get("failure_class"),
        "hardware_fingerprint": _canonical_digest(profiles["hardware"]),
        "model_fingerprint": _canonical_digest(profiles["model"]),
        "workload_fingerprint": _canonical_digest(profiles["workload"]),
        "software_fingerprint": _canonical_digest(profiles["software"]),
        "config_digest": _canonical_digest(config),
        "hardware_label": _label(profiles["hardware"], "unlabelled hardware"),
        "model_label": _label(profiles["model"], "unlabelled model"),
        "workload_label": _label(profiles["workload"], "unlabelled workload"),
        "objective_name": objective_name,
        "objective_value": objective_value,
        "wall_time_s": _number(metrics.get("wall_time_s")),
        "gpu_seconds": _number(metrics.get("gpu_seconds")),
        "peak_vram_gb": _number(metrics.get("peak_vram_gb")),
        "rollout_tok_s": _number(metrics.get("rollout_tok_s")),
        "train_tok_s": _number(metrics.get("train_tok_s")),
        "step_time_s": _number(metrics.get("step_time_s")),
        "wait_ratio": _number(metrics.get("wait_ratio")),
        "trainable_rollouts": _integer(metrics.get("trainable_rollouts")),
        "trainable_tokens": _integer(metrics.get("trainable_tokens")),
        "infra_failure_rate": _number(metrics.get("infra_failure_rate")),
        "repeats": _integer(metrics.get("repeats")),
        "sglang_tp": _integer(_nested(config, "sglang", "tp")),
        "sglang_replicas": _integer(_nested(config, "sglang", "replicas")),
        "concurrency": _integer(_nested(config, "sglang", "concurrency")),
        "megatron_tp": _integer(_nested(config, "megatron", "tp")),
        "megatron_pp": _integer(_nested(config, "megatron", "pp")),
        "megatron_cp": _integer(_nested(config, "megatron", "cp")),
        "megatron_ep": _integer(_nested(config, "megatron", "ep")),
        "megatron_etp": _integer(_nested(config, "megatron", "etp")),
        "train_gpus": _integer(_nested(config, "allocation", "train_gpus")),
        "rollout_gpus": _integer(_nested(config, "allocation", "rollout_gpus")),
        "hypothesis": manifest.get("hypothesis"),
        "observation": result.get("observation"),
        "next_step": result.get("next_step"),
        "parent_trial_ids": manifest.get("parent_trial_ids", []),
        "tags": manifest.get("tags", []),
        "created_at": manifest.get("created_at"),
        "artifact_path": str(path),
        "source_mtime_ns": _source_mtime_ns(path),
    }


def index_tuning_root(root: str | Path, db: Any) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    scanned = 0
    updated = 0
    errors: list[dict[str, str]] = []
    for trial_dir in discover_tuning_dirs(root):
        scanned += 1
        try:
            item = extract_tuning_metadata(trial_dir)
        except TuningArtifactError as exc:
            errors.append({"path": str(trial_dir), "error": str(exc)})
            continue
        existing = db.get_tuning_trial(item["trial_id"])
        if existing and existing.get("artifact_path") != item["artifact_path"]:
            errors.append(
                {
                    "path": str(trial_dir),
                    "error": f"duplicate trial_id {item['trial_id']!r} already indexes another path",
                }
            )
            continue
        if existing and existing.get("source_mtime_ns") == item["source_mtime_ns"]:
            continue
        db.upsert_tuning_trial(item)
        updated += 1
    return {
        "root": str(root),
        "scanned": scanned,
        "updated": updated,
        "errors": errors,
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in sorted(value.items()):
        path = f"{prefix}.{key}" if prefix else key
        flattened.update(_flatten(child, path))
    return flattened


def compare_tuning_bundles(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    if not 2 <= len(bundles) <= 8:
        raise TuningArtifactError("compare requires between 2 and 8 trials")
    trials = []
    config_keys: set[str] = set()
    for bundle in bundles:
        flat = _flatten(bundle["config"])
        config_keys.update(flat)
        trials.append(
            {
                "trial_id": bundle["manifest"]["trial_id"],
                "phase": bundle["manifest"]["phase"],
                "status": bundle["result"]["status"],
                "config": flat,
                "metrics": bundle["metrics"],
            }
        )
    varying = [
        key
        for key in sorted(config_keys)
        if len({json.dumps(item["config"].get(key), sort_keys=True) for item in trials}) > 1
    ]
    return {"trials": trials, "varying_config_keys": varying}


def rank_successful_trials(items: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        item for item in items if item.get("status") == "success" and item.get("objective_value") is not None
    ]
    if not successful:
        return {"recommended": None, "pareto": [], "reason": "no successful measured trials"}
    fingerprints = {
        (
            item.get("phase"),
            item.get("hardware_fingerprint"),
            item.get("model_fingerprint"),
            item.get("workload_fingerprint"),
            item.get("software_fingerprint"),
            item.get("objective_name"),
        )
        for item in successful
    }
    if len(fingerprints) != 1:
        return {
            "recommended": None,
            "pareto": [],
            "reason": "select one phase, objective, and matching hardware/model/workload/software cohort",
        }

    pareto = []
    for candidate in successful:
        dominated = any(
            other["trial_id"] != candidate["trial_id"]
            and other["objective_value"] >= candidate["objective_value"]
            and other.get("peak_vram_gb") is not None
            and candidate.get("peak_vram_gb") is not None
            and other["peak_vram_gb"] <= candidate["peak_vram_gb"]
            and (
                other["objective_value"] > candidate["objective_value"]
                or other["peak_vram_gb"] < candidate["peak_vram_gb"]
            )
            for other in successful
        )
        if not dominated:
            pareto.append(candidate)
    pareto.sort(key=lambda item: item["objective_value"], reverse=True)
    return {"recommended": max(successful, key=lambda item: item["objective_value"]), "pareto": pareto, "reason": None}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def create_tuning_trial(
    root: str | Path,
    *,
    phase: str,
    hardware: str | Path,
    model: str | Path,
    workload: str | Path,
    software: str | Path,
    config: str | Path,
    trial_id: str | None = None,
    hypothesis: str | None = None,
    parent_trial_ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> Path:
    if phase not in PHASES:
        raise TuningArtifactError(f"phase must be one of {', '.join(PHASES)}")
    trial_id = _validate_trial_id(
        trial_id or f"tune-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    parent_trial_ids = _string_list(parent_trial_ids or [], "parent_trial_ids")
    tags = _string_list(tags or [], "tags")
    sources = {
        "hardware.json": _read_object(Path(hardware).expanduser().resolve()),
        "model.json": _read_object(Path(model).expanduser().resolve()),
        "workload.json": _read_object(Path(workload).expanduser().resolve()),
        "software.json": _read_object(Path(software).expanduser().resolve()),
        "config.json": _read_object(Path(config).expanduser().resolve()),
    }
    trial_dir = Path(root).expanduser().resolve() / "tuning" / trial_id
    if trial_dir.exists():
        raise FileExistsError(f"trial already exists: {trial_dir}")
    trial_dir.mkdir(parents=True)
    for name, value in sources.items():
        _atomic_json(trial_dir / name, value)
    _atomic_json(
        trial_dir / "tuning-manifest.json",
        {
            "schema_version": TUNING_SCHEMA_VERSION,
            "trial_id": trial_id,
            "phase": phase,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hypothesis": hypothesis,
            "parent_trial_ids": parent_trial_ids,
            "tags": tags,
        },
    )
    _atomic_json(trial_dir / "result.json", {"status": "planned"})
    return trial_dir


def update_tuning_trial(
    trial_dir: str | Path,
    *,
    status: str,
    metrics: str | Path | None = None,
    failure_class: str | None = None,
    observation: str | None = None,
    next_step: str | None = None,
) -> None:
    if status not in STATUSES:
        raise TuningArtifactError(f"status must be one of {', '.join(STATUSES)}")
    trial_dir = Path(trial_dir).expanduser().resolve()
    bundle = load_tuning_bundle(trial_dir, include_artifacts=False)
    if metrics is not None:
        measured = _read_object(Path(metrics).expanduser().resolve())
        _validate_metrics(measured)
        _atomic_json(trial_dir / "metrics.json", measured)
    if status == "success" and not (trial_dir / "metrics.json").is_file():
        raise TuningArtifactError("a successful trial requires metrics.json")
    result = {**bundle["result"], "status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    if status in {"planned", "running", "success"} and failure_class is None:
        result.pop("failure_class", None)
    elif failure_class is not None:
        result["failure_class"] = failure_class
    if observation is not None:
        result["observation"] = observation
    if next_step is not None:
        result["next_step"] = next_step
    _atomic_json(trial_dir / "result.json", result)
