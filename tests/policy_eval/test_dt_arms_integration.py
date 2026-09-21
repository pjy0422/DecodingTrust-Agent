from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from dt_arena.policy_eval.dt_arms_integration import (
    DtArmsIntegrationError,
    discover_generation_artifacts,
    merge_generated_attack,
    materialize_replay_workspace,
    normalize_dt_arms_trajectory,
    snapshot_content_sha256,
    snapshot_provenance,
    task_identity,
    write_task_list,
)
from dt_arena.policy_eval.scripts import run_dt_arms_e2e


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "dataset/finance/malicious/indirect/action_reversal/1"


def test_vendored_snapshot_matches_pinned_upstream_content() -> None:
    provenance = snapshot_provenance()

    assert provenance["commit"] == "62595a72a8d798a0704cbb56a09521e15d0f30c2"
    assert provenance["tree"] == "ef538379d19f8462ebe503f5edda38a4742ace77"
    assert provenance["files"] == 279
    assert snapshot_content_sha256(ROOT / "dt_arms") == provenance["content_sha256"]


def test_task_list_uses_dt_arms_native_dataset_identity(tmp_path: Path) -> None:
    expected = {
        "domain": "finance",
        "type": "malicious",
        "threat_model": "indirect",
        "risk_category": "action_reversal",
        "task_id": "1",
    }

    assert task_identity(TASK) == expected
    output = write_task_list(TASK, tmp_path / "task.jsonl")
    assert json.loads(output.read_text()) == expected


def test_generated_attack_merge_changes_only_attack_turns(tmp_path: Path) -> None:
    source = yaml.safe_load((TASK / "config.yaml").read_text(encoding="utf-8"))
    generated = {
        "Attack": {
            "risk_category": source["Attack"]["risk_category"],
            "threat_model": source["Attack"]["threat_model"],
            "malicious_goal": source["Attack"]["malicious_goal"],
            "attack_turns": source["Attack"]["attack_turns"],
        }
    }
    candidate = tmp_path / "attack_result_environment.yaml"
    candidate.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")

    merged = merge_generated_attack(TASK / "config.yaml", candidate)

    assert merged["Task"] == source["Task"]
    assert merged["Agent"] == source["Agent"]
    assert merged["RedTeamingAgent"] == source["RedTeamingAgent"]
    assert merged["Attack"]["attack_turns"] == generated["Attack"]["attack_turns"]

    generated["Attack"]["malicious_goal"] = "changed"
    candidate.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
    with pytest.raises(DtArmsIntegrationError, match="changed malicious_goal"):
        merge_generated_attack(TASK / "config.yaml", candidate)


def test_authoritative_replay_workspace_is_isolated(tmp_path: Path) -> None:
    source_bytes = (TASK / "config.yaml").read_bytes()
    source = yaml.safe_load(source_bytes)
    workspace = materialize_replay_workspace(
        source_task_dir=TASK,
        merged_config=source,
        runtime_root=tmp_path / "runtime",
        output_root=tmp_path / "artifacts",
    )

    assert workspace.task_dir.is_relative_to(tmp_path / "runtime")
    assert yaml.safe_load(workspace.config_path.read_text(encoding="utf-8")) == source
    assert (TASK / "config.yaml").read_bytes() == source_bytes


def test_generation_artifacts_are_unambiguous(tmp_path: Path) -> None:
    assert discover_generation_artifacts(tmp_path) == (None, None)
    attack = tmp_path / "nested/attack_result_environment.yaml"
    attack.parent.mkdir()
    attack.write_text("Attack: {}\n", encoding="utf-8")
    trajectory = tmp_path / "nested/red_teaming_agent_1.json"
    trajectory.write_text("{}\n", encoding="utf-8")
    assert discover_generation_artifacts(tmp_path) == (attack, trajectory)

    (tmp_path / "attack_result_prompt.yaml").write_text("Attack: {}\n", encoding="utf-8")
    with pytest.raises(DtArmsIntegrationError, match="ambiguous"):
        discover_generation_artifacts(tmp_path)


def test_raw_dt_arms_trajectory_is_normalized_only_for_viewer(tmp_path: Path) -> None:
    raw = {
        "attack_config": {"agent_model": "attacker-model"},
        "attack_trajectory": [
            {
                "role": "attacker",
                "action": "inject_env",
                "reasoning": "Choose the listing surface.",
                "metadata": {"tool": "travel-injection:inject_review"},
            },
            {
                "role": "victim",
                "state": "I ignored the injected request.",
                "metadata": {"query": "Book the cheapest listing."},
            },
        ],
        "attack_result": {"success": False},
    }
    source = tmp_path / "trajectory.json"
    destination = tmp_path / "policy.jsonl"
    source.write_text(json.dumps(raw), encoding="utf-8")

    normalize_dt_arms_trajectory(source, destination)
    events = [json.loads(line) for line in destination.read_text().splitlines()]

    assert source.read_text(encoding="utf-8") == json.dumps(raw)
    assert events[0]["source"] == "dt-arms-upstream"
    assert events[0]["message"]["content"][1]["name"] == "dt_arms__inject_env"
    assert events[-1] == {
        "model": "attacker-model",
        "policy_engine": "dt-arms-upstream",
        "source": "dt-arms-upstream",
        "success": False,
        "type": "result",
    }


@pytest.mark.asyncio
async def test_generation_exhaustion_is_not_reported_as_authoritative_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Process:
        returncode = 0
        pid = 12345

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"native search complete", b""

    async def fake_exec(*command: str, **_kwargs: object) -> Process:
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / "red_teaming_agent_1.json").write_text(
            json.dumps(
                {
                    "attack_config": {"agent_model": "attacker-model"},
                    "attack_trajectory": [],
                    "attack_result": {"success": False},
                }
            ),
            encoding="utf-8",
        )
        return Process()

    monkeypatch.setattr(run_dt_arms_e2e.asyncio, "create_subprocess_exec", fake_exec)
    args = Namespace(
        task_dir=TASK,
        dtap_root=ROOT,
        python="python",
        artifacts_dir=tmp_path / "artifacts",
        attacker_model="attacker-model",
        victim_model="victim-model",
        victim_agent_type="openclaw",
        judge_model="judge-model",
        max_iterations=4,
        victim_max_turns=80,
        timeout=30,
        port_range_start=20_000,
        use_memory=False,
        update_memory=False,
        memory_save_mode="success",
        auto_aggregate_memory=False,
        allow_quit=True,
        multi_turn=False,
        max_turns_per_session=5,
        injection_override=None,
        allowed_skill_names=None,
        allowed_skill_types=None,
    )

    assert await run_dt_arms_e2e._run(args) == 0
    result = json.loads((args.artifacts_dir / "result.json").read_text(encoding="utf-8"))
    assert result["episode_status"] == "generation_exhausted"
    assert result["dt_arms_success"] is False
    assert result["candidate_generated"] is False
    assert result["evaluation_completed"] is False
    assert result["attack_success"] is None
    assert (args.artifacts_dir / "dt-arms/trajectory.json").is_file()
    assert (args.artifacts_dir / "policy.jsonl").is_file()
