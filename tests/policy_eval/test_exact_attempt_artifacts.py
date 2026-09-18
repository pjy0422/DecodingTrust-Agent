from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from dt_arena.policy_eval.attempt_runner import DtapAttemptRunner
from dt_arena.policy_eval.feedback.deterministic import _matches
from dt_arena.policy_eval.feedback.targets import AccessPattern, TargetField
from dt_arena.policy_eval.scripts.run_policy_e2e import RecordingRunner


def _trajectory(path: Path, marker: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trajectory": [{"marker": marker}]}), encoding="utf-8")
    return path


def test_attempt_artifact_resolution_requires_one_unambiguous_file(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path / "nested" / "victim.json", "exact")
    events = tmp_path / "nested" / "victim.mcp-events.jsonl"
    events.write_text("{}\n", encoding="utf-8")

    assert DtapAttemptRunner._single_trajectory_artifact(tmp_path) == trajectory.resolve()
    assert DtapAttemptRunner._single_regular_artifact(
        tmp_path, "*.mcp-events.jsonl"
    ) == events.resolve()

    _trajectory(tmp_path / "other.json", "ambiguous")
    (tmp_path / "other.mcp-events.jsonl").write_text("{}\n", encoding="utf-8")
    assert DtapAttemptRunner._single_trajectory_artifact(tmp_path) is None
    assert DtapAttemptRunner._single_regular_artifact(
        tmp_path, "*.mcp-events.jsonl"
    ) is None


def test_attempt_artifact_resolution_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    outside = _trajectory(tmp_path / "outside.json", "outside")
    (root / "escaped.json").symlink_to(outside)

    assert DtapAttemptRunner._single_trajectory_artifact(root) is None


def test_exporter_copies_only_runner_selected_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "attempt-output"
    exact_trajectory = _trajectory(output_root / "selected" / "exact.json", "selected")
    exact_events = output_root / "selected" / "exact.mcp-events.jsonl"
    exact_events.write_text('{"selected": true}\n', encoding="utf-8")
    _trajectory(output_root / "distractor.json", "wrong")
    (output_root / "distractor.mcp-events.jsonl").write_text(
        '{"selected": false}\n', encoding="utf-8"
    )

    artifacts = tmp_path / "artifacts"
    runner = RecordingRunner(SimpleNamespace(), artifacts_dir=artifacts)
    runner.plans.append([])
    workspace = SimpleNamespace(output_root=output_root, attempt_index=2)
    result = SimpleNamespace(
        trajectory_path=exact_trajectory,
        mcp_events_path=exact_events,
    )

    runner._export_artifacts(workspace, result)

    attempt_dir = artifacts / "attempts" / "attempt-0002"
    assert json.loads((attempt_dir / "victim-trajectory.json").read_text())["trajectory"] == [
        {"marker": "selected"}
    ]
    assert (attempt_dir / "victim-mcp-events.jsonl").read_text() == '{"selected": true}\n'
    assert (artifacts / "victim-mcp-events.jsonl").read_text() == '{"selected": true}\n'


def test_deterministic_matching_prefers_redacted_arguments_in_schema_v2() -> None:
    expected = "matter-17"
    pattern = AccessPattern(
        ("legal:get_matter",),
        (TargetField("matter_id", expected),),
    )
    event = {
        "server": "legal",
        "tool": "get_matter",
        # Raw viewer data is deliberately inconsistent: deterministic matching
        # must continue to use the stable redacted contract.
        "arguments": {"matter_id": "different-raw-value"},
        "arguments_redacted": {
            "matter_id": {
                "type": "string",
                "length": len(expected),
                "sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
            }
        },
    }

    assert _matches(pattern, event) is True
