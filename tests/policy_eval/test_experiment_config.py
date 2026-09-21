from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from dt_arena.policy_eval.experiment_config import (
    SCHEMA,
    ExperimentConfigError,
    load_experiment_config,
    resolved_experiment_document,
)
from dt_arena.policy_eval.scripts.run_domain_matrix import _case_dir, _matrix_tasks


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "dt_arena/policy_eval/configs/finance-travel-indirect-openclaw.yaml"


def test_checked_in_experiment_config_controls_independent_budgets() -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    loaded = load_experiment_config(EXAMPLE)

    assert loaded["dtap_root"] == ROOT
    assert loaded["domains"] == ["finance", "travel"]
    assert loaded["threat_models"] == ["indirect"]
    assert loaded["max_submissions"] == raw["budgets"]["h_victim_executions"]
    assert loaded["max_submit_calls"] == raw["budgets"]["q_submit_calls"]
    assert loaded["max_placement_actions"] == raw["budgets"]["max_placement_actions"]
    assert loaded["policy_engine"] == "claude-code"
    assert loaded["dt_arms_max_iterations"] == 10
    assert loaded["policy_max_turns"] is None
    assert loaded["harness_protocol"] == "v1"
    assert loaded["improvement_wishes"] is True
    assert loaded["dying_message"] is True
    assert loaded["digestor_max_tokens"] == 50_000
    assert loaded["victim_agent_type"] == "openclaw"


def test_config_is_closed_and_rejects_credentials(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["provider"] = {"api_key": "must-not-be-stored"}
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="unknown top-level field.*provider"):
        load_experiment_config(target)


def test_q_cannot_be_lower_than_h(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["budgets"]["h_victim_executions"] = 5
    raw["budgets"]["q_submit_calls"] = 4
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="q_submit_calls must be >="):
        load_experiment_config(target)


def test_relative_paths_are_resolved_from_config_location(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["paths"]["dtap_root"] = "repo"
    raw["paths"]["artifacts_root"] = "runs/example"
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_experiment_config(target)
    assert loaded["dtap_root"] == (tmp_path / "repo").resolve()
    assert loaded["artifacts_root"] == (tmp_path / "runs/example").resolve()


def test_resolved_document_contains_no_environment_or_credentials() -> None:
    values = load_experiment_config(EXAMPLE)
    values["policy_max_turns"] = 160
    args = Namespace(**values)

    document = resolved_experiment_document(args)

    assert document["schema"] == SCHEMA
    assert document["budgets"] == {
        "h_victim_executions": values["max_submissions"],
        "q_submit_calls": values["max_submit_calls"],
        "max_placement_actions": values["max_placement_actions"],
    }
    assert document["policy"]["improvement_wishes"] is True
    assert document["policy"]["dying_message"] is True
    assert document["policy"]["harness_protocol"] == "v1"
    assert document["policy"]["engine"] == "claude-code"
    assert document["dt_arms"]["max_iterations"] == 10
    rendered = yaml.safe_dump(document).lower()
    assert "api_key" not in rendered
    assert "auth_token" not in rendered
    assert "environment" not in rendered


def test_lazy_schema_protocol_is_explicit_and_closed(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["policy"]["engine"] = "claude-code"
    raw["policy"]["harness_protocol"] = "lazy-schema-v2"
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert load_experiment_config(target)["harness_protocol"] == "lazy-schema-v2"

    raw["policy"]["harness_protocol"] = "future-v3"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="unsupported policy harness protocol"):
        load_experiment_config(target)


def test_dt_arms_rejects_claude_only_lazy_schema_protocol(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["policy"]["engine"] = "dt-arms-upstream"
    raw["policy"]["harness_protocol"] = "lazy-schema-v2"
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="applies only.*claude-code"):
        load_experiment_config(target)


def test_policy_engine_is_explicit_and_closed(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["policy"]["engine"] = "claude-code"
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert load_experiment_config(target)["policy_engine"] == "claude-code"

    raw["policy"]["engine"] = "other"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="unsupported policy engine"):
        load_experiment_config(target)


def test_dt_arms_rejects_victim_harness_not_supported_upstream(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["policy"]["engine"] = "dt-arms-upstream"
    raw["victim"]["harness"] = "hermes"
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="does not support victim harness"):
        load_experiment_config(target)

    raw["policy"]["engine"] = "claude-code"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_experiment_config(target)["victim_agent_type"] == "hermes"


def test_explicit_dataset_tasks_are_closed_and_retained(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["selection"]["tasks"] = [
        "finance/malicious/indirect/action_reversal/1",
        "travel/malicious/direct/privacy_violation/2",
    ]
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_experiment_config(target)
    document = resolved_experiment_document(Namespace(**loaded))

    assert loaded["selected_tasks"] == raw["selection"]["tasks"]
    assert document["selection"]["tasks"] == raw["selection"]["tasks"]

    raw["selection"]["tasks"] = ["../../outside"]
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="entries must match"):
        load_experiment_config(target)


def test_explicit_matrix_tasks_get_distinct_artifact_directories(tmp_path: Path) -> None:
    paths = [
        "finance/malicious/indirect/action_reversal/1",
        "finance/malicious/indirect/action_reversal/2",
    ]
    for relative in paths:
        task = tmp_path / "dataset" / relative
        task.mkdir(parents=True)
        (task / "config.yaml").write_text("Task: {}\n", encoding="utf-8")
    args = Namespace(
        dtap_root=tmp_path,
        selected_tasks=paths,
        domains=["finance"],
        threat_models=["indirect"],
        selection_profile="release-v1",
    )

    tasks = _matrix_tasks(args)

    assert [task.task_id for task in tasks] == ["1", "2"]
    assert [_case_dir(tmp_path / "artifacts", task) for task in tasks] == [
        tmp_path / "artifacts/finance/indirect/action_reversal/1",
        tmp_path / "artifacts/finance/indirect/action_reversal/2",
    ]
