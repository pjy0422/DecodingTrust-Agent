from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from dtap_traj.bundle import load_episode_bundle
from dtap_traj.db import TrajectoryDB
from dtap_traj.indexer import discover_episode_dirs, extract_episode_metadata, index_root
from dtap_traj.server import create_app
from fastapi.testclient import TestClient


def test_viewer_refresh_is_manual_and_does_not_reset_reading_position():
    app_js = (Path(__file__).parents[1] / "dtap_traj" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'id="refreshEpisodes"' in app_js
    assert "window.setInterval" not in app_js


def test_experiment_ui_has_hierarchical_multi_task_picker():
    web = Path(__file__).parents[1] / "dtap_traj" / "web"
    app_js = (web / "app.js").read_text(encoding="utf-8")
    css = (web / "app.css").read_text(encoding="utf-8")

    assert "/api/experiments/datasets" in app_js
    assert "syncDatasetSelection" in app_js
    assert "data-dataset-level" in app_js
    assert "data-dataset-task" in app_js
    assert "Select all IDs" in app_js
    assert "Clear all IDs" in app_js
    assert "Select matches" in app_js
    assert "max_parallel" in app_js
    assert "job-progress-summary" in app_js
    assert "Task progress updates when Refresh is clicked" in app_js
    assert "Full feedback evidence" in app_js
    assert 'id="experimentPolicyEngine"' in app_js
    assert 'id="experimentHarnessProtocol"' in app_js
    assert 'id="experimentPlanningStrategy"' in app_js
    assert 'id="experimentVictimHarness"' in app_js
    assert 'id="experimentFeedbackMode"' in app_js
    assert "updateExperimentControl" in app_js
    assert "writeYamlControl" in app_js
    assert "lazy-schema-v2" in (web / "yaml_controls.js").read_text(encoding="utf-8")
    assert "dt-arms-upstream" in (web / "yaml_controls.js").read_text(encoding="utf-8")
    assert "experimentControlValue('policy','engine','claude-code')" in app_js
    controls = (web / "yaml_controls.js").read_text(encoding="utf-8")
    assert controls.index("['claude-code'") < controls.index("['dt-arms-upstream'")
    assert ".dataset-browser" in css
    assert ".runtime-fields" in css
    assert ".job-task-list" in css


def test_managed_viewer_bridges_ollama_for_native_dt_arms():
    script = (
        Path(__file__).parents[1] / "scripts" / "serve-managed.sh"
    ).read_text(encoding="utf-8")

    assert "DTAP_ARMS_OPENAI_API_KEY" in script
    assert "DTAP_ARMS_OPENAI_BASE_URL" in script
    assert "OLLAMA_OPENAI_BASE_URL" in script


def write_episode(root: Path, domain: str, threat: str, i: int, attack: bool = False) -> Path:
    d = root / domain / threat
    d.mkdir(parents=True, exist_ok=True)
    episode_id = f"ep-{i:02d}-{domain}-{threat}"
    (d / "episode-manifest.json").write_text(json.dumps({"episode_id": episode_id}))
    (d / "result.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "domain": domain,
                "threat_model": threat,
                "status": "passed",
                "episode_status": "succeeded" if attack else "exhausted",
                "risk_category": f"{domain}-risk",
                "task_id": str(i),
                "policy_engine": "claude-code",
                "policy_model": "policy-test-model",
                "victim_model": "victim-test-model",
                "victim_agent_type": "openai_sdk",
                "attack_success": attack,
                "evaluation_completed": True,
                "submissions": 1,
                "placement_applicable": threat == "indirect",
                "placement_covered": threat == "indirect",
                "placement_actions": 1 if threat == "indirect" else 0,
                "placements_verified": 1 if threat == "indirect" else 0,
                "environment_steps": i % 3,
            }
        )
    )
    policy = [
        {"type": "system", "subtype": "thinking_tokens", "estimated_tokens_delta": 4},
        {
            "type": "assistant",
            "message": {
                "id": "a",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Inspect the configuration and choose an injection target."}
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "id": "b",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "mcp__filesystem__read_file",
                        "input": {"path": "config.yaml"},
                    }
                ],
            },
        },
        # Replayed stream snapshots must not turn one provider tool call into two.
        {
            "type": "assistant",
            "message": {
                "id": "b-replay",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "mcp__filesystem__read_file",
                        "input": {"path": "config.yaml"},
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "id": "c",
                "role": "assistant",
                "content": [{"type": "text", "text": "Prepared candidate configuration."}],
            },
        },
        {"type": "result", "usage": {"input_tokens": 10, "output_tokens": 8}},
    ]
    (d / "policy.jsonl").write_text("\n".join(json.dumps(x) for x in policy) + "\n")
    victim = {
        "traj_info": {
            "metadata": {
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "reasoning_tokens": 2,
                }
            }
        },
        "trajectory": [
            {"role": "user", "state": "Process the pending task."},
            {
                "role": "agent",
                "metadata": {"tool_name": "list_files", "server": "filesystem", "tool_params": {"path": "/workspace"}},
            },
            {"role": "tool", "state": "config.yaml\nREADME.md"},
            {"role": "agent", "metadata": {"message": "Task completed safely."}},
        ],
    }
    (d / "victim-trajectory.json").write_text(json.dumps(victim))
    (d / "policy-prompt.txt").write_text("You are the attack policy. Produce a valid candidate config.")
    (d / "original-config.yaml").write_text(
        f"Task:\n  task_id: {domain}-malicious-{threat}-{domain}-risk-{i:03d}\nmode: safe\nlimit: 1\n"
    )
    (d / "submitted-config.yaml").write_text("mode: injected\nlimit: 1\n")
    (d / "judge-result.json").write_text(
        json.dumps(
            {
                "task_success": True,
                "task_message": "Task completed",
                "task_metadata": {"check": "deterministic"},
                "attack_success": attack,
                "attack_message": "Attack evaluated",
                "attack_metadata": (
                    {
                        "judge_tool": "llm_judge",
                        "llm_model": "test-model",
                        "llm_judge": {"reasoning": "test evidence"},
                        "verdict_source": "llm_judge",
                    }
                    if attack
                    else {"check": "deterministic"}
                ),
                "error": None,
            }
        )
    )
    (d / "judge-verdict.json").write_text(json.dumps({"attack_success": attack}))
    return d


def matrix(tmp_path: Path) -> Path:
    root = tmp_path / "p0-p2-live-matrix-20260906"
    domains = [
        "browser",
        "code",
        "crm",
        "customer-service",
        "finance",
        "legal",
        "medical",
        "os-filesystem",
        "research",
        "telecom",
        "travel",
        "workflow",
    ]
    i = 0
    for domain in domains:
        for threat in ("direct", "indirect"):
            i += 1
            write_episode(root, domain, threat, i, attack=i in {1, 2, 24})
    return root


def test_direct_run_without_result_uses_trusted_judge_verdict(tmp_path):
    root = tmp_path / "artifacts" / "direct-run"
    episode = write_episode(root, "browser", "direct", 7, attack=False)
    (episode / "result.json").unlink()

    metadata = extract_episode_metadata(episode, root)
    assert metadata["attack_success"] is False
    assert metadata["evaluation_completed"] is True

    bundle = load_episode_bundle(episode)
    assert bundle["evaluation"]["attack_success"] is False
    assert bundle["evaluation"]["evaluation_completed"] is True

    db = TrajectoryDB(tmp_path / "stale.sqlite3")
    db.upsert_episode({**metadata, "attack_success": None, "evaluation_completed": None})
    indexed = index_root(root, db)
    assert indexed["updated"] == 1
    refreshed = db.get_episode(metadata["episode_id"])
    assert refreshed["attack_success"] is False
    assert refreshed["evaluation_completed"] is True


def test_index_24_case_matrix_and_filters(tmp_path):
    root = matrix(tmp_path)
    db = TrajectoryDB(tmp_path / "index.sqlite3")
    assert len(discover_episode_dirs(root)) == 24
    result = index_root(root, db)
    assert result == {"root": str(root.resolve()), "scanned": 24, "updated": 24}
    assert db.facets()["total"] == 24
    assert db.facets()["attack_successes"] == 3
    asr = db.list_episodes()["asr"]
    assert asr["h1"] == {"successes": 3, "evaluated": 24, "rate": 0.125}
    assert asr["h2"] == {"successes": 0, "evaluated": 0, "rate": None}
    assert asr["cumulative"] == {
        "successes": 3,
        "evaluated": 24,
        "rate": 0.125,
    }
    assert db.list_episodes(domain="workflow")["total"] == 2
    assert db.list_episodes(threat_model="indirect")["total"] == 12
    assert db.list_episodes(attack_success=True)["total"] == 3
    second = index_root(root, db)
    assert second["updated"] == 0


def test_api_policy_victim_combined_and_config(tmp_path):
    root = matrix(tmp_path)
    app = create_app(root, db_path=tmp_path / "api.sqlite3")
    client = TestClient(app)
    assert client.get("/api/health").json()["ok"] is True
    assert client.get("/api/tuning/facets").json()["total"] == 0
    episodes = client.get("/api/episodes", params={"domain": "browser"}).json()
    assert episodes["total"] == 2
    ep = episodes["items"][0]
    assert ep["task_id"].startswith("browser-malicious-")
    assert ep["dataset_path"].startswith("browser/malicious/")
    assert ep["policy_engine"] == "claude-code"
    assert ep["policy_model"] == "policy-test-model"
    assert ep["victim_model"] == "victim-test-model"
    assert ep["victim_agent_type"] == "openai_sdk"
    assert ep["policy_events"] == 1
    assert ep["policy_usage"] == {
        "input_tokens": 10,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 8,
        "reasoning_tokens": 4,
        "reasoning_source": "stream_estimate",
        "tokens_with_reasoning": 18,
        "tokens_without_reasoning": 14,
    }
    assert ep["victim_usage"]["tokens_with_reasoning"] == 30
    assert ep["victim_usage"]["tokens_without_reasoning"] == 28
    eid = ep["episode_id"]
    policy = client.get(f"/api/episodes/{eid}/trajectory", params={"view": "policy"}).json()
    assert "policy" in policy and "victim" not in policy
    assert any(e["kind"] == "tool_call" for e in policy["policy"])
    victim = client.get(f"/api/episodes/{eid}/trajectory", params={"view": "victim"}).json()
    assert "victim" in victim and "policy" not in victim
    assert victim["victim_usage"]["reasoning_tokens"] == 2
    combined = client.get(f"/api/episodes/{eid}/trajectory", params={"view": "combined"}).json()
    assert combined["policy"] and combined["victim"]
    config = client.get(f"/api/episodes/{eid}/config").json()["comparison"]
    assert config["identical"] is False
    assert "-mode: safe" in config["diff"] and "+mode: injected" in config["diff"]
    judges = client.get(f"/api/episodes/{eid}/judges").json()["judges"]
    assert judges["available"] is True
    assert [item["name"] for item in judges["components"]] == ["task", "attack"]
    assert judges["components"][0]["source"] == "deterministic"
    assert judges["reward_firewall"] == {"attack_success": ep["attack_success"]}


def test_prompt_tab_uses_exact_artifacts_and_marks_unretained_components(tmp_path):
    root = tmp_path / "prompts"
    episode = write_episode(root, "browser", "direct", 1, attack=True)
    (episode / "original-config.yaml").write_text(
        "Agent:\n  system_prompt: original victim prompt\n",
        encoding="utf-8",
    )
    (episode / "submitted-config.yaml").write_text(
        "Agent:\n  system_prompt: effective submitted victim prompt\n",
        encoding="utf-8",
    )
    (episode / "prompt-snapshots.jsonl").write_text(
        json.dumps(
            {
                "schema": "dtap-policy-eval-prompt-snapshot",
                "schema_version": 1,
                "sequence": 1,
                "component": "digestor",
                "label": "Digestor request prompt",
                "role": "user",
                "prompt": "exact dynamic digestor request",
                "source": "feedback.digestor",
                "exact": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(root, db_path=tmp_path / "prompts.sqlite3"))
    episode_id = json.loads((episode / "result.json").read_text())["episode_id"]

    payload = client.get(f"/api/episodes/{episode_id}/prompts").json()["prompts"]
    by_component = {item["component"]: item for item in payload["components"]}
    assert by_component["policy"]["label"] == "Policy launch prompt"
    assert by_component["policy"]["role"] == "user"
    assert by_component["victim"]["prompt"] == "effective submitted victim prompt"
    assert by_component["victim"]["role"] == "system"
    assert by_component["digestor"]["prompt"] == "exact dynamic digestor request"
    assert all(item["exact"] is True for item in payload["components"])
    unavailable = {item["component"]: item["reason"] for item in payload["unavailable"]}
    assert "Claude Code owns this prompt" in unavailable["policy_runtime"]
    assert "runtime did not retain" in unavailable["attack_judge"]


def test_prompt_tab_frontend_has_copyable_prompt_components():
    web = Path(__file__).parents[1] / "dtap_traj" / "web"
    app_js = (web / "app.js").read_text(encoding="utf-8")
    css = (web / "app.css").read_text(encoding="utf-8")
    assert "/prompts${suffix}" in app_js
    assert "function promptsHtml" in app_js
    assert "Only exact prompts retained by this episode" in app_js
    assert 'data-tab="${t}"' in app_js
    assert ".prompt-card" in css


def test_llm_as_judge_is_distinct_from_deterministic_and_firewall(tmp_path):
    root = tmp_path / "judges"
    episode_dir = write_episode(root, "browser", "direct", 1, attack=True)
    app = create_app(root, db_path=tmp_path / "judges.sqlite3")
    client = TestClient(app)
    episode_id = json.loads((episode_dir / "result.json").read_text())["episode_id"]
    judges = client.get(f"/api/episodes/{episode_id}/judges").json()["judges"]
    assert judges["components"][0]["source"] == "deterministic"
    assert judges["components"][1]["source"] == "llm_as_judge"
    assert judges["components"][1]["metadata"]["llm_model"] == "test-model"
    assert judges["reward_firewall"] == {"attack_success": True}
    assert judges["raw"]["attack_message"] == "Attack evaluated"


def test_h2_attempt_selector_returns_each_config_victim_and_judge(tmp_path):
    root = tmp_path / "h2"
    episode = write_episode(root, "finance", "indirect", 1)
    attempts = episode / "attempts"
    for index, attack in ((1, False), (2, True)):
        attempt = attempts / f"attempt-{index:04d}"
        attempt.mkdir(parents=True)
        (attempt / "submitted-config.yaml").write_text(f"attempt: {index}\n")
        (attempt / "victim-trajectory.json").write_text(
            json.dumps(
                {
                    "traj_info": {
                        "metadata": {
                            "token_usage": {
                                "input_tokens": index * 100,
                                "output_tokens": index * 10,
                                "reasoning_tokens": index * 3,
                                "tokens_with_reasoning": index * 110,
                                "tokens_without_reasoning": index * 107,
                                "reasoning_source": "provider",
                            }
                        }
                    },
                    "trajectory": [{"role": "agent", "metadata": {"message": f"victim-{index}"}}],
                }
            )
        )
        (attempt / "judge-result.json").write_text(
            json.dumps({"task_success": True, "attack_success": attack, "error": None})
        )
        (attempt / "judge-verdict.json").write_text(json.dumps({"attack_success": attack}))
        (attempt / "feedback-evidence.json").write_text(
            json.dumps({"schema_version": 3, "final_response": f"feedback-{index}"})
        )
    (episode / "prompt-snapshots.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "schema": "dtap-policy-eval-prompt-snapshot",
                    "schema_version": 1,
                    "sequence": index,
                    "attempt_index": index,
                    "component": "digestor",
                    "label": "Digestor request prompt",
                    "role": "user",
                    "prompt": f"digestor-H{index}",
                    "source": "feedback.digestor",
                }
            )
            for index in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )

    client = TestClient(create_app(root, db_path=tmp_path / "h2.sqlite3"))
    episode_id = json.loads((episode / "result.json").read_text())["episode_id"]
    latest = client.get(f"/api/episodes/{episode_id}/trajectory").json()
    assert latest["attempt_index"] == 2
    assert latest["attempts"] == [
        {"index": 1, "attack_success": False},
        {"index": 2, "attack_success": True},
    ]
    first = client.get(f"/api/episodes/{episode_id}/trajectory", params={"attempt": 1}).json()
    assert any("victim-1" in event.get("text", "") for event in first["victim"])
    assert first["victim_usage"]["input_tokens"] == 100
    assert first["feedback_evidence"] == {
        "schema_version": 3,
        "final_response": "feedback-1",
    }
    config = client.get(f"/api/episodes/{episode_id}/config", params={"attempt": 1}).json()["comparison"]
    assert "+attempt: 1" in config["diff"]
    judges = client.get(f"/api/episodes/{episode_id}/judges", params={"attempt": 2}).json()["judges"]
    assert judges["reward_firewall"] == {"attack_success": True}
    prompts = client.get(f"/api/episodes/{episode_id}/prompts", params={"attempt": 1}).json()["prompts"]
    digestor_prompts = [item for item in prompts["components"] if item["component"] == "digestor"]
    assert [item["prompt"] for item in digestor_prompts] == ["digestor-H1"]
    cohort = client.get("/api/episodes").json()["asr"]
    assert cohort["h1"] == {"successes": 0, "evaluated": 1, "rate": 0.0}
    assert cohort["h2"] == {"successes": 1, "evaluated": 1, "rate": 1.0}
    assert cohort["cumulative"] == {
        "successes": 1,
        "evaluated": 1,
        "rate": 1.0,
    }
    assert client.get(f"/api/episodes/{episode_id}/trajectory", params={"attempt": 3}).status_code == 404


def test_server_side_search_and_pagination(tmp_path):
    root = matrix(tmp_path)
    client = TestClient(create_app(root, db_path=tmp_path / "q.sqlite3"))
    assert client.get("/api/episodes", params={"q": "workflow"}).json()["total"] == 2
    page = client.get("/api/episodes", params={"limit": 5, "offset": 5}).json()
    assert page["total"] == 24 and len(page["items"]) == 5 and page["offset"] == 5


def test_run_summary_models_and_trace_metadata_fill_legacy_episode(tmp_path):
    root = tmp_path / "artifacts"
    run = root / "legacy-run"
    episode = write_episode(run, "browser", "direct", 7)
    result_path = episode / "result.json"
    result = json.loads(result_path.read_text())
    result.pop("policy_model")
    result.pop("victim_model")
    result.pop("victim_agent_type")
    result_path.write_text(json.dumps(result))
    (run / "summary.json").write_text(
        json.dumps(
            {
                "policy_model": "deepseek-v4-flash",
                "victim_model": "deepseek-v4-flash",
                "victim_agent_type": "openclaw",
            }
        )
    )

    db = TrajectoryDB(tmp_path / "legacy.sqlite3")
    assert index_root(root, db)["updated"] == 1
    item = db.list_episodes()["items"][0]
    assert item["task_id"] == "browser-malicious-direct-browser-risk-007"
    assert item["dataset_path"] == "browser/malicious/direct/browser-risk/7"
    assert item["policy_model"] == "deepseek-v4-flash"
    assert item["victim_model"] == "deepseek-v4-flash"
    assert item["victim_agent_type"] == "openclaw"
    assert db.list_episodes(q="openclaw")["total"] == 1
    assert db.list_episodes(q="deepseek-v4-flash")["total"] == 1
    assert db.list_episodes(q=item["task_id"])["total"] == 1


def test_existing_index_schema_is_migrated_without_dropping_rows(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE episodes ("
            "episode_id TEXT PRIMARY KEY, run_name TEXT NOT NULL, domain TEXT, "
            "threat_model TEXT, status TEXT, episode_status TEXT, risk_category TEXT, "
            "attack_success INTEGER, evaluation_completed INTEGER, "
            "placement_applicable INTEGER, placement_covered INTEGER, "
            "placement_actions INTEGER, placements_verified INTEGER, "
            "environment_steps INTEGER, policy_events INTEGER, victim_events INTEGER, "
            "artifact_path TEXT NOT NULL, source_mtime_ns INTEGER NOT NULL, "
            "indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO episodes (episode_id, run_name, artifact_path, source_mtime_ns) VALUES ('old-episode', 'old-run', '/tmp/old', 1)"
        )

    db = TrajectoryDB(path)
    item = db.get_episode("old-episode")
    assert item is not None
    assert item["task_id"] is None
    assert item["dataset_path"] is None
    assert item["policy_model"] is None
    assert item["victim_model"] is None
    assert item["victim_agent_type"] is None


def test_explorer_assets_include_persistent_light_theme():
    web = Path(__file__).resolve().parents[1] / "dtap_traj" / "web"
    html = (web / "index.html").read_text()
    javascript = (web / "app.js").read_text()
    css = (web / "app.css").read_text()
    assert "dtap-explorer-theme" in html
    assert 'id="themeToggle"' in javascript
    assert "localStorage.setItem('dtap-explorer-theme',value)" in javascript
    assert ':root[data-theme="light"]' in css


def test_explorer_displays_victim_agentic_harness():
    web = Path(__file__).resolve().parents[1] / "dtap_traj" / "web"
    javascript = (web / "app.js").read_text()
    assert "victim harness" in javascript
    assert "OpenClaw" in javascript
    assert "OpenAI SDK" in javascript


def test_explorer_has_per_component_copy_controls():
    web = Path(__file__).resolve().parents[1] / "dtap_traj" / "web"
    javascript = (web / "app.js").read_text()
    css = (web / "app.css").read_text()

    assert "data-copy-block" in javascript
    assert "data-copy-content" in javascript
    assert "navigator.clipboard.writeText" in javascript
    assert "document.execCommand('copy')" in javascript
    assert ".copy-button" in css


def test_explorer_exposes_shareable_top_level_trajectory_and_performance_tabs():
    web = Path(__file__).resolve().parents[1] / "dtap_traj" / "web"
    javascript = (web / "app.js").read_text()

    assert 'role="tablist"' in javascript
    assert 'role="tab"' in javascript
    assert "searchParams.set('view',value)" in javascript
    assert "modeFromLocation()" in javascript
    assert 'data-mode="trajectories"' in javascript
    assert 'data-mode="performance"' in javascript


def test_attack_filter_distinguishes_failed_from_not_evaluated(tmp_path):
    root = matrix(tmp_path)
    missing = root / "browser" / "direct" / "result.json"
    payload = json.loads(missing.read_text())
    payload["attack_success"] = None
    missing.write_text(json.dumps(payload))
    # A genuinely unevaluated artifact has no trusted judge output. Keeping a
    # false verdict here would mean evaluation completed and the attack failed.
    (missing.parent / "judge-result.json").unlink()
    (missing.parent / "judge-verdict.json").unlink()
    app = create_app(root, db_path=tmp_path / "outcomes.sqlite3")
    client = TestClient(app)
    assert client.get("/api/episodes", params={"attack_success": False}).json()["total"] == 21
    assert client.get("/api/episodes", params={"attack_evaluated": False}).json()["total"] == 1
    assert client.get("/api/episodes", params={"attack_evaluated": True}).json()["total"] == 23


def test_collection_root_preserves_run_names(tmp_path):
    root = tmp_path / "runs"
    write_episode(root / "run-a", "browser", "direct", 1)
    write_episode(root / "run-b", "workflow", "indirect", 2)
    db = TrajectoryDB(tmp_path / "runs.sqlite3")
    assert index_root(root, db)["scanned"] == 2
    assert {item["value"] for item in db.facets()["runs"]} == {"run-a", "run-b"}


def test_api_refuses_database_path_outside_artifact_root(tmp_path):
    root = tmp_path / "inside"
    write_episode(root, "browser", "direct", 1)
    outside = write_episode(tmp_path / "outside", "workflow", "indirect", 2)
    app = create_app(root, db_path=tmp_path / "paths.sqlite3")
    item = extract_episode_metadata(outside, tmp_path / "outside")
    app.state.db.upsert_episode(item)
    client = TestClient(app)
    episode_id = item["episode_id"]
    assert client.get(f"/api/episodes/{episode_id}/trajectory").status_code == 404
    assert client.get(f"/api/episodes/{episode_id}/config").status_code == 404


def test_unchanged_reindex_does_not_reread_large_trajectories(tmp_path, monkeypatch):
    root = matrix(tmp_path)
    db = TrajectoryDB(tmp_path / "watch.sqlite3")
    assert index_root(root, db)["updated"] == 24

    def unexpected_read(_path):
        raise AssertionError("unchanged trajectory was read")

    monkeypatch.setattr("dtap_traj.indexer._policy_metrics", unexpected_read)
    monkeypatch.setattr("dtap_traj.indexer._victim_count", unexpected_read)
    assert index_root(root, db)["updated"] == 0
