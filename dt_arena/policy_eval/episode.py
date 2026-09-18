from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .dtap_compat import DtapApi, load_dtap_api
from .integrity import BenchmarkIntegrityGuard, BenchmarkManifest


@dataclass(frozen=True)
class TaskSnapshot:
    """Trusted-side snapshot. Never serialize this object to the policy."""

    task_dir: Path
    config_sha256: str
    raw_config: dict[str, Any]
    task_config: Any
    attack_config: Any
    agent_config: Any
    injection_config: dict[str, Any]
    benchmark_manifest: BenchmarkManifest | None = None

    def assert_config_unchanged(self) -> None:
        if self.benchmark_manifest is not None:
            BenchmarkIntegrityGuard.verify(self.task_dir, self.benchmark_manifest)
            return
        config_path = self.task_dir / "config.yaml"
        current_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if current_hash != self.config_sha256:
            raise RuntimeError(f"DTAP config.yaml changed during the episode: {config_path}")


def load_task_snapshot(task_dir: Path | str, *, dtap_api: DtapApi | None = None) -> TaskSnapshot:
    task_dir = Path(task_dir).expanduser().resolve()
    config_path = task_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"DTAP config not found: {config_path}")

    manifest = BenchmarkIntegrityGuard.capture(task_dir)
    raw_bytes = config_path.read_bytes()
    parsed = yaml.safe_load(raw_bytes) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"DTAP config root must be a mapping: {config_path}")

    api = dtap_api or load_dtap_api()
    config_str = str(config_path)

    snapshot = TaskSnapshot(
        task_dir=task_dir,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw_config=copy.deepcopy(parsed),
        task_config=api.TaskConfig.from_yaml(config_str),
        attack_config=api.AttackConfig.from_yaml(config_str),
        agent_config=api.AgentConfig.from_yaml(config_str),
        injection_config=copy.deepcopy(api.parse_injection_config(copy.deepcopy(parsed))),
        benchmark_manifest=manifest,
    )
    snapshot.assert_config_unchanged()
    return snapshot
