from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from dtap_traj.experiments import (
    ExperimentLaunchError,
    ExperimentManager,
    discover_dataset_tasks,
)
from dtap_traj.server import create_app
from fastapi.testclient import TestClient


def config_yaml(tmp_path: Path) -> str:
    return yaml.safe_dump(
        {
            "schema": "dtap.policy-eval/experiment-v1",
            "paths": {
                "dtap_root": "/untrusted/root",
                "artifacts_root": "/untrusted/artifacts",
                "python": "/untrusted/python",
            },
            "selection": {
                "domains": ["finance"],
                "threat_models": ["indirect"],
                "profile": "release-v1",
            },
            "models": {"policy": "policy-model", "victim": "victim-model"},
            "policy": {
                "planning_strategy": "current",
                "max_turns": "auto",
                "improvement_wishes": True,
                "dying_message": True,
            },
            "victim": {"harness": "openclaw", "max_turns": 80},
            "budgets": {
                "h_victim_executions": 2,
                "q_submit_calls": 6,
                "max_placement_actions": 8,
            },
            "placement": {"enabled": True},
            "feedback": {
                "mode": "final+deterministic",
                "digestor_model": "digestor-model",
                "digestor_max_tokens": 2500,
                "digestor_timeout_seconds": 30,
                "reasoning_summary": False,
            },
            "execution": {
                "max_parallel": 1,
                "timeout_seconds": 1800,
                "port_range_start": 20000,
                "port_range_stride": 1024,
                "resume": False,
            },
        },
        sort_keys=False,
    )


@pytest.fixture
def manager(tmp_path: Path) -> ExperimentManager:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "baseline.yaml").write_text(config_yaml(tmp_path), encoding="utf-8")
    return ExperimentManager(
        artifact_root=tmp_path / "artifacts",
        config_dir=configs,
        runner_root=Path(__file__).parents[3],
        python=Path("/usr/bin/python3"),
        state_dir=tmp_path / "state",
        launch_token="test-secret",
    )


def test_validate_overrides_untrusted_runtime_paths(manager: ExperimentManager):
    result = manager.validate(manager.template("baseline.yaml")["yaml"], "safe-run")
    normalized = yaml.safe_load(result["normalized_yaml"])
    assert normalized["paths"] == {
        "dtap_root": str(manager.runner_root),
        "artifacts_root": str(manager.artifact_root / "safe-run"),
        "python": str(manager.python),
    }
    assert result["resolved"]["max_submissions"] == 2
    assert "config" not in result["resolved"]


def test_python_virtualenv_symlink_is_preserved(tmp_path: Path):
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to("/usr/bin/python3")
    configs = tmp_path / "configs"
    configs.mkdir()
    selected = ExperimentManager(
        artifact_root=tmp_path / "artifacts",
        config_dir=configs,
        runner_root=Path(__file__).parents[3],
        python=interpreter,
        state_dir=tmp_path / "state",
        launch_token="secret",
    )
    assert selected.python == interpreter.absolute()


def test_rejects_path_like_run_name_and_web_budget_overflow(manager: ExperimentManager):
    with pytest.raises(ExperimentLaunchError, match="run_name"):
        manager.validate(manager.template("baseline.yaml")["yaml"], "../escape")
    document = yaml.safe_load(manager.template("baseline.yaml")["yaml"])
    document["execution"]["max_parallel"] = 17
    with pytest.raises(ExperimentLaunchError, match="max_parallel"):
        manager.validate(yaml.safe_dump(document), "too-parallel")


def test_dataset_catalog_and_exact_multi_selection(manager: ExperimentManager):
    catalog = manager.datasets()
    selected_items = catalog["items"][:2]
    selected = [item["path"] for item in selected_items]

    result = manager.validate(
        manager.template("baseline.yaml")["yaml"],
        "selected-run",
        selected,
    )
    normalized = yaml.safe_load(result["normalized_yaml"])

    assert catalog["total"] > 2
    assert catalog["max_parallel"] == 16
    assert normalized["selection"]["tasks"] == selected
    assert normalized["selection"]["domains"] == sorted(
        {item["domain"] for item in selected_items}
    )
    assert normalized["execution"]["max_parallel"] == 16
    assert result["resolved"]["selected_tasks"] == selected

    cleared = manager.validate(result["normalized_yaml"], "cleared-run", [])
    assert "tasks" not in yaml.safe_load(cleared["normalized_yaml"])["selection"]
    assert cleared["resolved"]["selected_tasks"] == []

    with pytest.raises(ExperimentLaunchError, match="unknown or non-runnable"):
        manager.validate(
            manager.template("baseline.yaml")["yaml"],
            "unknown-task",
            ["finance/malicious/indirect/not-real/999999"],
        )


def test_dataset_discovery_uses_only_canonical_malicious_shape(tmp_path: Path):
    valid = tmp_path / "dataset/finance/malicious/indirect/action_reversal/2/config.yaml"
    valid.parent.mkdir(parents=True)
    valid.write_text("Task: {}\n", encoding="utf-8")
    benign = tmp_path / "dataset/finance/benign/example/1/config.yaml"
    benign.parent.mkdir(parents=True)
    benign.write_text("Task: {}\n", encoding="utf-8")

    assert discover_dataset_tasks(tmp_path) == [
        {
            "path": "finance/malicious/indirect/action_reversal/2",
            "domain": "finance",
            "threat_model": "indirect",
            "risk_category": "action_reversal",
            "task_id": "2",
        }
    ]


def test_api_requires_token_and_launches_worker(monkeypatch, tmp_path: Path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "baseline.yaml").write_text(config_yaml(tmp_path), encoding="utf-8")
    monkeypatch.setenv("DTAP_VIEWER_LAUNCH_TOKEN", "test-secret")

    class Process:
        pid = 4321

    launched = []

    def fake_popen(command, **kwargs):
        launched.append((command, kwargs))
        return Process()

    monkeypatch.setattr("dtap_traj.experiments.subprocess.Popen", fake_popen)
    app = create_app(
        tmp_path / "artifacts",
        db_path=tmp_path / "viewer.sqlite3",
        experiment_config_dir=configs,
        experiment_runner_root=Path(__file__).parents[3],
        experiment_python="/usr/bin/python3",
        experiment_state_dir=tmp_path / "state",
    )
    client = TestClient(app)
    assert client.get("/api/experiments/templates").status_code == 401
    headers = {"X-DTAP-Launch-Token": "test-secret"}
    datasets = client.get("/api/experiments/datasets", headers=headers).json()
    assert datasets["total"] > 0
    assert datasets["max_parallel"] == 16
    template = client.get("/api/experiments/templates/baseline.yaml", headers=headers).json()
    response = client.post(
        "/api/experiments/launch",
        headers=headers,
        json={"run_name": "api-run", "yaml": template["yaml"]},
    )
    assert response.status_code == 202
    job = response.json()
    assert job["worker_pid"] == 4321
    assert launched[0][0][1:3] == ["-m", "dtap_traj.experiment_worker"]
    saved = json.loads((tmp_path / "state/jobs" / job["job_id"] / "job.json").read_text())
    assert saved["run_name"] == "api-run"
    assert client.get("/api/experiments/jobs", headers=headers).json()["items"][0]["job_id"] == job["job_id"]


def test_existing_artifact_requires_resume(manager: ExperimentManager, monkeypatch):
    (manager.artifact_root / "existing").mkdir(parents=True)
    with pytest.raises(ExperimentLaunchError, match="already exists"):
        manager.launch(manager.template("baseline.yaml")["yaml"], "existing")
