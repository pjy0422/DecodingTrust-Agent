"""Authenticated, constrained launcher for policy-eval experiment YAML files."""

from __future__ import annotations

import json
import importlib.util
import os
import re
import secrets
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_MAX_YAML_BYTES = 128 * 1024


class ExperimentLaunchError(ValueError):
    """The submitted experiment is unsafe or invalid."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ExperimentManager:
    """Own templates, validate YAML, and launch durable worker processes.

    Paths and the Python executable are server authority. A browser can tune the
    closed experiment schema, but cannot turn the viewer into a command runner or
    write artifacts outside the indexed root.
    """

    def __init__(
        self,
        *,
        artifact_root: Path,
        config_dir: Path,
        runner_root: Path,
        python: Path,
        state_dir: Path,
        launch_token: str,
    ) -> None:
        self.artifact_root = artifact_root.resolve()
        self.config_dir = config_dir.resolve()
        self.runner_root = runner_root.resolve()
        self.python = python.resolve()
        self.state_dir = state_dir.resolve()
        self.launch_token = launch_token
        self.jobs_dir = self.state_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return bool(self.launch_token)

    def authorize(self, supplied: str | None) -> bool:
        return bool(supplied) and secrets.compare_digest(supplied, self.launch_token)

    def templates(self) -> list[dict[str, str]]:
        if not self.config_dir.is_dir():
            return []
        return [
            {"name": path.name, "yaml": path.read_text(encoding="utf-8")}
            for path in sorted(self.config_dir.glob("*.yaml"))
            if path.is_file()
        ]

    def template(self, name: str) -> dict[str, str]:
        path = (self.config_dir / name).resolve()
        try:
            path.relative_to(self.config_dir)
        except ValueError as exc:
            raise ExperimentLaunchError("invalid template name") from exc
        if path.suffix != ".yaml" or not path.is_file():
            raise ExperimentLaunchError("experiment template not found")
        return {"name": path.name, "yaml": path.read_text(encoding="utf-8")}

    def _normalized(self, yaml_text: str, run_name: str) -> tuple[str, dict[str, Any], Path]:
        if not _RUN_NAME.fullmatch(run_name):
            raise ExperimentLaunchError(
                "run_name must be 1-80 letters, digits, dots, underscores, or hyphens"
            )
        if len(yaml_text.encode("utf-8")) > _MAX_YAML_BYTES:
            raise ExperimentLaunchError("experiment YAML exceeds 128 KiB")
        try:
            document = yaml.safe_load(yaml_text)
        except yaml.YAMLError as exc:
            raise ExperimentLaunchError(f"invalid YAML: {exc}") from exc
        if not isinstance(document, dict):
            raise ExperimentLaunchError("experiment YAML must contain a mapping")
        paths = document.setdefault("paths", {})
        if not isinstance(paths, dict):
            raise ExperimentLaunchError("paths must be a mapping")
        output = (self.artifact_root / run_name).resolve()
        output.relative_to(self.artifact_root)
        paths.update(
            {
                "dtap_root": str(self.runner_root),
                "artifacts_root": str(output),
                "python": str(self.python),
            }
        )
        normalized = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
        with tempfile.TemporaryDirectory(dir=self.state_dir) as temporary:
            config_path = Path(temporary) / "config.yaml"
            config_path.write_text(normalized, encoding="utf-8")
            try:
                module_path = self.runner_root / "dt_arena/policy_eval/experiment_config.py"
                spec = importlib.util.spec_from_file_location("dtap_experiment_config", module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot load experiment config validator: {module_path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                resolved = module.load_experiment_config(config_path)
            except (ImportError, ValueError) as exc:
                raise ExperimentLaunchError(str(exc)) from exc
        if resolved["max_parallel"] > 24:
            raise ExperimentLaunchError("execution.max_parallel must be <= 24 from the viewer")
        if resolved["max_submissions"] > 20:
            raise ExperimentLaunchError("budgets.h_victim_executions must be <= 20 from the viewer")
        if resolved["max_submit_calls"] > 200 or resolved["max_placement_actions"] > 200:
            raise ExperimentLaunchError("Q and placement budgets must be <= 200 from the viewer")
        if resolved["timeout"] > 86_400:
            raise ExperimentLaunchError("execution.timeout_seconds must be <= 86400")
        public = {
            key: value
            for key, value in resolved.items()
            if key not in {"config", "dtap_root", "artifacts_root", "python"}
        }
        public["artifacts_root"] = str(output)
        return normalized, public, output

    def validate(self, yaml_text: str, run_name: str) -> dict[str, Any]:
        normalized, resolved, output = self._normalized(yaml_text, run_name)
        return {
            "valid": True,
            "normalized_yaml": normalized,
            "resolved": resolved,
            "output_exists": output.exists(),
        }

    def launch(self, yaml_text: str, run_name: str) -> dict[str, Any]:
        normalized, resolved, output = self._normalized(yaml_text, run_name)
        if output.exists() and not resolved["resume"]:
            raise ExperimentLaunchError(
                f"artifact run already exists: {run_name}; choose another name or enable resume"
            )
        job_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(4)
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir()
        config_path = job_dir / "config.yaml"
        config_path.write_text(normalized, encoding="utf-8")
        record = {
            "job_id": job_id,
            "run_name": run_name,
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "artifact_path": str(output),
            "config_path": str(config_path),
            "log_path": str(job_dir / "runner.log"),
            "resolved": resolved,
        }
        _atomic_json(job_dir / "job.json", record)
        command = [
            str(self.python),
            "-m",
            "dtap_traj.experiment_worker",
            "--job-dir",
            str(job_dir),
            "--runner-root",
            str(self.runner_root),
            "--python",
            str(self.python),
        ]
        process = subprocess.Popen(
            command,
            cwd=self.runner_root,
            env=os.environ.copy(),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        current = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        current["worker_pid"] = process.pid
        current["updated_at"] = _now()
        _atomic_json(job_dir / "job.json", current)
        return current

    def jobs(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.jobs_dir.glob("*/job.json"), reverse=True):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return records

    def job(self, job_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9TZ-]+[a-f0-9]{8}", job_id):
            raise ExperimentLaunchError("invalid job id")
        path = self.jobs_dir / job_id / "job.json"
        if not path.is_file():
            raise ExperimentLaunchError("experiment job not found")
        record = json.loads(path.read_text(encoding="utf-8"))
        log_path = self.jobs_dir / job_id / "runner.log"
        if log_path.is_file():
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 64 * 1024))
                record["log_tail"] = stream.read().decode("utf-8", errors="replace")
        else:
            record["log_tail"] = ""
        return record


__all__ = ["ExperimentLaunchError", "ExperimentManager", "_atomic_json", "_now"]
