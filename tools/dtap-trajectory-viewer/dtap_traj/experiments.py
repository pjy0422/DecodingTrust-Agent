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
_VIEWER_MAX_PARALLEL = 16


class ExperimentLaunchError(ValueError):
    """The submitted experiment is unsafe or invalid."""


def discover_dataset_tasks(runner_root: Path) -> list[dict[str, str]]:
    """Return runnable malicious tasks using DTAP's canonical directory shape."""

    dataset_root = runner_root.resolve() / "dataset"
    items: list[dict[str, str]] = []
    if not dataset_root.is_dir():
        return items
    for config in dataset_root.glob("*/malicious/*/*/*/config.yaml"):
        try:
            config.resolve().relative_to(dataset_root)
        except ValueError:
            continue
        relative = config.parent.relative_to(dataset_root)
        if len(relative.parts) != 5:
            continue
        domain, malicious, threat_model, risk_category, task_id = relative.parts
        if malicious != "malicious":
            continue
        items.append(
            {
                "path": relative.as_posix(),
                "domain": domain,
                "threat_model": threat_model,
                "risk_category": risk_category,
                "task_id": task_id,
            }
        )

    def sort_key(item: dict[str, str]) -> tuple[Any, ...]:
        task_id = item["task_id"]
        task_key: tuple[int, Any] = (0, int(task_id)) if task_id.isdigit() else (1, task_id.casefold())
        return (
            item["domain"].casefold(),
            item["threat_model"].casefold(),
            item["risk_category"].casefold(),
            task_key,
        )

    return sorted(items, key=sort_key)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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
        # Preserve a virtualenv symlink: resolving it selects the base interpreter
        # and silently drops the environment's installed dependencies.
        self.python = Path(os.path.abspath(python.expanduser()))
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

    def datasets(self) -> dict[str, Any]:
        items = discover_dataset_tasks(self.runner_root)
        return {
            "items": items,
            "total": len(items),
            "max_parallel": _VIEWER_MAX_PARALLEL,
        }

    def _normalized(
        self,
        yaml_text: str,
        run_name: str,
        selected_tasks: list[str] | None = None,
    ) -> tuple[str, dict[str, Any], Path]:
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
        if selected_tasks is not None:
            if not isinstance(selected_tasks, list):
                raise ExperimentLaunchError("selected dataset tasks must be a list")
            if not all(isinstance(item, str) for item in selected_tasks):
                raise ExperimentLaunchError("selected dataset tasks must be strings")
            if len(set(selected_tasks)) != len(selected_tasks):
                raise ExperimentLaunchError("selected dataset tasks must not contain duplicates")
            selection = document.setdefault("selection", {})
            if not isinstance(selection, dict):
                raise ExperimentLaunchError("selection must be a mapping")
            if not selected_tasks:
                # An explicit empty browser selection must clear tasks written
                # by an earlier normalization, rather than silently rerunning
                # a stale exact-task list.
                selection.pop("tasks", None)
            else:
                catalog = {item["path"]: item for item in discover_dataset_tasks(self.runner_root)}
                unknown = sorted(set(selected_tasks) - set(catalog))
                if unknown:
                    raise ExperimentLaunchError(
                        f"unknown or non-runnable dataset task: {unknown[0]}"
                    )
                selection["tasks"] = selected_tasks
                selection["domains"] = sorted({catalog[path]["domain"] for path in selected_tasks})
                selection["threat_models"] = sorted(
                    {catalog[path]["threat_model"] for path in selected_tasks}
                )
                execution = document.setdefault("execution", {})
                if not isinstance(execution, dict):
                    raise ExperimentLaunchError("execution must be a mapping")
                execution["max_parallel"] = _VIEWER_MAX_PARALLEL
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
        if resolved["max_parallel"] > _VIEWER_MAX_PARALLEL:
            raise ExperimentLaunchError(
                f"execution.max_parallel must be <= {_VIEWER_MAX_PARALLEL} from the viewer"
            )
        if resolved["selected_tasks"]:
            catalog_paths = {
                item["path"] for item in discover_dataset_tasks(self.runner_root)
            }
            unknown = sorted(set(resolved["selected_tasks"]) - catalog_paths)
            if unknown:
                raise ExperimentLaunchError(
                    f"unknown or non-runnable dataset task: {unknown[0]}"
                )
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

    def validate(
        self,
        yaml_text: str,
        run_name: str,
        selected_tasks: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized, resolved, output = self._normalized(yaml_text, run_name, selected_tasks)
        return {
            "valid": True,
            "normalized_yaml": normalized,
            "resolved": resolved,
            "output_exists": output.exists(),
        }

    def launch(
        self,
        yaml_text: str,
        run_name: str,
        selected_tasks: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized, resolved, output = self._normalized(yaml_text, run_name, selected_tasks)
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
        bootstrap_log = job_dir / "worker-bootstrap.log"
        with bootstrap_log.open("ab", buffering=0) as bootstrap:
            process = subprocess.Popen(
                command,
                cwd=self.runner_root,
                env=os.environ.copy(),
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=bootstrap,
                stderr=subprocess.STDOUT,
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
                record = json.loads(path.read_text(encoding="utf-8"))
                record["progress"] = self._job_progress(record)
                records.append(record)
            except (OSError, json.JSONDecodeError):
                continue
        return records

    def _job_progress(self, record: dict[str, Any]) -> dict[str, Any]:
        resolved = record.get("resolved") if isinstance(record.get("resolved"), dict) else {}
        selected = resolved.get("selected_tasks")
        root = Path(str(record.get("artifact_path", ""))).resolve()
        try:
            root.relative_to(self.artifact_root)
        except ValueError:
            return {"total": 0, "completed": 0, "running": 0, "queued": 0, "failed": 0, "tasks": []}

        planned: list[tuple[str, Path]] = []
        if isinstance(selected, list) and selected:
            for task in selected:
                if not isinstance(task, str):
                    continue
                parts = Path(task).parts
                if len(parts) != 5 or parts[1] != "malicious":
                    continue
                domain, _, threat, risk, task_id = parts
                planned.append((task, root / domain / threat / risk / task_id))
        else:
            domains = resolved.get("domains", [])
            threats = resolved.get("threat_models", [])
            if isinstance(domains, list) and isinstance(threats, list):
                for domain in domains:
                    for threat in threats:
                        if isinstance(domain, str) and isinstance(threat, str):
                            planned.append(
                                (f"{domain}/malicious/{threat}/<profile-selected-task>", root / domain / threat)
                            )

        tasks: list[dict[str, Any]] = []
        h_limit = resolved.get("max_submissions")
        for label, task_root in planned:
            result = _read_json(task_root / "result.json")
            state = _read_json(task_root / "episode-state.json")
            attempts_root = task_root / "attempts"
            attempts = (
                len([path for path in attempts_root.glob("attempt-*") if path.is_dir()])
                if attempts_root.is_dir()
                else 0
            )
            if result:
                status = "completed" if result.get("status") == "passed" else "failed"
                stage = "attack succeeded" if result.get("attack_success") is True else "evaluation complete"
            elif state.get("evaluation_delegate_completed"):
                status, stage = "running", "attempt complete; policy continuing"
            elif attempts or (task_root / "submitted-config.yaml").is_file():
                status, stage = "running", "victim evaluation"
            elif (task_root / "policy.jsonl").is_file() or (task_root / "policy-prompt.txt").is_file():
                status, stage = "running", "policy planning"
            elif task_root.is_dir():
                status, stage = "running", "starting"
            else:
                status, stage = "queued", "waiting for worker"
            tasks.append(
                {
                    "task": label,
                    "status": status,
                    "stage": stage,
                    "attempts": attempts,
                    "h_limit": h_limit,
                    "attack_success": result.get("attack_success"),
                    "episode_status": result.get("episode_status") or state.get("status"),
                    "error": result.get("error_tail") or result.get("error"),
                }
            )
        counts = {
            status: sum(task["status"] == status for task in tasks)
            for status in ("completed", "running", "queued", "failed")
        }
        return {"total": len(tasks), **counts, "tasks": tasks}

    def job(self, job_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9TZ-]+[a-f0-9]{8}", job_id):
            raise ExperimentLaunchError("invalid job id")
        path = self.jobs_dir / job_id / "job.json"
        if not path.is_file():
            raise ExperimentLaunchError("experiment job not found")
        record = json.loads(path.read_text(encoding="utf-8"))
        record["progress"] = self._job_progress(record)
        chunks = []
        for name in ("worker-bootstrap.log", "runner.log"):
            log_path = self.jobs_dir / job_id / name
            if log_path.is_file():
                with log_path.open("rb") as stream:
                    stream.seek(max(0, log_path.stat().st_size - 64 * 1024))
                    content = stream.read().decode("utf-8", errors="replace")
                if content:
                    chunks.append(f"== {name} ==\n{content}")
        record["log_tail"] = "\n".join(chunks)
        return record


__all__ = [
    "ExperimentLaunchError",
    "ExperimentManager",
    "discover_dataset_tasks",
    "_atomic_json",
    "_now",
]
