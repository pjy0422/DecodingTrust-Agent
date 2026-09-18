"""Integration coverage for the checked-in 24-case live matrix.

This test is skipped when the slime repository fixture is not present (e.g. when
this package is tested standalone), and runs automatically from the repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dtap_traj.bundle import load_episode_bundle
from dtap_traj.db import TrajectoryDB
from dtap_traj.indexer import discover_episode_dirs, index_root


def _matrix_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "examples" / "dtap_agent_rl" / "artifacts" / "p0-p2-live-matrix-20260906"
        if candidate.is_dir():
            return candidate
    return None


@pytest.mark.integration
def test_checked_in_live_matrix_is_explorer_ready(tmp_path):
    root = _matrix_root()
    if root is None:
        pytest.skip("checked-in slime live matrix not available")

    dirs = discover_episode_dirs(root)
    assert len(dirs) == 24

    db = TrajectoryDB(tmp_path / "live-matrix.sqlite3")
    indexed = index_root(root, db)
    assert indexed["scanned"] == 24
    assert db.facets()["total"] == 24
    assert len(db.facets()["domains"]) == 12
    assert db.list_episodes(threat_model="direct")["total"] == 12
    assert db.list_episodes(threat_model="indirect")["total"] == 12

    # Exercise the production parser against representative direct/indirect bundles.
    for domain, threat in (("browser", "direct"), ("workflow", "indirect")):
        row = db.list_episodes(domain=domain, threat_model=threat)["items"][0]
        bundle = load_episode_bundle(row["artifact_path"])
        assert bundle["policy_timeline"], (domain, threat)
        assert bundle["timeline"], (domain, threat)
        assert bundle["config_comparison"] is not None
        assert "evaluation" in bundle
        assert bundle["judges"]["available"] is True
        assert len(bundle["judges"]["components"]) == 2
        assert bundle["judges"]["reward_firewall"]

    # The live matrix contains both judge modes in a single episode: the task
    # check is deterministic while the attack check is LLM-as-judge.
    row = db.list_episodes(domain="customer-service", threat_model="direct")["items"][0]
    judges = load_episode_bundle(row["artifact_path"])["judges"]
    assert [item["source"] for item in judges["components"]] == [
        "deterministic",
        "llm_as_judge",
    ]
    assert judges["components"][1]["metadata"]["llm_model"] == "deepseek-v4-flash"
