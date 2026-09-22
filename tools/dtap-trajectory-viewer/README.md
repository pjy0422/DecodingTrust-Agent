# DTAP trajectory viewer

This copy is owned by DecodingTrust-Agent and was migrated from
`pjy0422/slime@f1e8ee90fe725d394b4e9929947423ecaa01976c`. It intentionally preserves support for
artifact schema `dtap-agent-rl-episode` version 1 so historical slime
trajectories and new `dt_arena.policy_eval` trajectories can be indexed
together.

---

# DTAP Policy + Victim Trajectory Viewer

The package now supports four workflows:

1. `dtap-traj serve`: a local indexed explorer for many RL trajectories.
2. Legacy single-file HTML export for sharing one self-contained trajectory.
3. A manual Performance Preflight + Tuning Registry in the same explorer.
4. An authenticated YAML experiment launcher that submits policy E2E matrices
   and indexes their artifacts in the same screen.

The explorer keeps the artifact directory as the source of truth. SQLite stores only searchable episode metadata and artifact paths, while the existing parser lazily loads policy/victim/config data for the selected episode.

## Local explorer

```bash
cd tools/dtap-trajectory-viewer
pip install -e .

dtap-traj serve \
  ../../examples/dtap_agent_rl/artifacts/p0-p2-live-matrix-20260906 \
  --watch --open
```

The UI opens at `http://127.0.0.1:8765` by default and supports:

- run/domain/threat-model/status/attack-success filters,
- DTAP's dataset directory convention (`domain/malicious/threat/risk/task`) as
  the primary debugging label, while preserving the config task ID and random
  public episode ID as internal references,
- policy/victim model names (with an explicit `not recorded` fallback),
- victim agentic harness names such as OpenClaw and OpenAI SDK,
- explicit policy tool-call and victim-step counts,
- policy activity counted by actual tool calls rather than stream-json line
  count,
- provider token totals with thinking-included and thinking-excluded views;
  reasoning values derived from stream estimates are visibly marked `≈`,
  while victim token usage follows the currently selected `H=h` attempt,
- cohort ASR cards for first submission (`H=1`), second submission among
  episodes reaching it (`H=2`), and cumulative success across either
  submission (`H=1,2`), always with numerator and denominator,
- free-text search over task/episode IDs, model names, risk categories and
  artifact paths,
- lazy Policy trajectory view,
- lazy Victim trajectory view,
- side-by-side Combined view pairing the selected `H=h` submitted config with
  the victim trajectory produced by that submission,
- a Prompts view that labels each retained prompt by component and message role;
  it shows the exact policy launch prompt, selected victim system prompt, and
  new digestor/reasoning request snapshots without reconstructing missing
  historical prompts from current source,
- DTAP task/attack judge results with deterministic and LLM-as-judge labels,
- the trusted reward-firewall verdict kept separate from raw judge metadata,
- original/submitted Config Diff,
- server-side pagination,
- live server-side re-indexing with `--watch`; use **Refresh index** (or the
  Run E2E job **Refresh** button) when you want the browser to redraw. The
  viewer deliberately does not auto-redraw while logs are being read, so the
  scroll position and browser translation state remain stable,
- persistent light/dark color themes from the top-bar toggle.

## Authenticated E2E launcher

The **Run E2E** tab loads strict `dtap.policy-eval/experiment-v1` templates,
allows the YAML values to be edited, validates and normalizes the document, and
starts the existing semantic domain-matrix runner. A submitted job writes into
the viewer's artifact root, so `--watch` registers completed episodes
automatically. Runner output and status are retained under the configured state
directory.

The dataset picker reads only canonical runnable directories shaped as
`dataset/<domain>/malicious/<threat_model>/<risk_category>/<task_id>`. Choose a
domain, threat model, and risk category, then select any number of task IDs; the
selection is persisted as `selection.tasks` in the normalized experiment YAML.
Exact-task runs use distinct artifact directories and a bounded worker queue
with at most 16 concurrent tasks, so larger selections proceed in 16-task
waves. The server rejects paths that are not present in its own dataset catalog.

The launcher is disabled unless all four server arguments and a launch token
are present:

```bash
export DTAP_VIEWER_LAUNCH_TOKEN="$(openssl rand -hex 32)"
dtap-traj serve /path/to/artifacts --watch \
  --experiment-config-dir /path/to/DecodingTrust-Agent/dt_arena/policy_eval/configs \
  --experiment-runner-root /path/to/DecodingTrust-Agent \
  --experiment-python /path/to/dtap/bin/python \
  --experiment-state-dir ~/.local/state/dtap-trajectory-viewer/experiments
```

Enter that token in the Run E2E tab. The browser keeps it in `sessionStorage`,
which is cleared when the tab session ends. The token, API keys, provider URLs,
and arbitrary environment variables are never accepted in experiment YAML or
written to artifacts. The server overwrites `paths.dtap_root`,
`paths.artifacts_root`, and `paths.python`, constrains the output to the indexed
artifact root, validates the closed schema, applies conservative web-launch
budget caps, and asks for confirmation before consuming provider resources.
Workers run in a separate process session and persist their status/config/log,
so restarting the viewer does not discard an in-flight run.

For a managed host, `scripts/serve-managed.sh` reads the launch token from a
systemd credential, maps the server's provider environment into the policy,
victim, and digestor processes, and starts the same CLI. The optional
`scripts/update-managed-viewer.sh` is suitable for a user timer: it deploys
only `origin/main`, refuses a dirty deployment worktree, and restarts the
viewer only when the commit changes. Set the viewer unit's `KillMode=process`
so a UI code restart does not terminate detached experiment workers.
When Claude/OpenClaw CLIs are installed under a version manager rather than the
system PATH, set `DTAP_NODE_BIN_DIR` in the service (for example, the active
NVM version's `bin` directory); detached workers inherit that trusted path.
The wrapper always prepends the configured `DTAP_VIEWER_PYTHON` virtualenv's
`bin` directory as well, so DTAP's `python3` subprocess helpers use the same
supported interpreter and installed dependencies as the matrix runner.

## Performance preflight and tuning registry

Use the **Trajectories** and **Performance** tabs in the same Explorer screen;
`?view=performance` is a shareable link that opens the latter directly. The
Performance workspace inspects planned, running, successful, and failed runtime
trials separately from policy trajectories. It supports profile
filters, 2–8 candidate config comparison, subsystem time breakdown, objective
and peak-memory inspection, same-cohort Pareto candidates, and the hypothesis,
observation, and proposed next experiment.

Create a trial from reviewed JSON profiles and a candidate config:

```bash
dtap-traj tune create /path/to/artifacts \
  --phase sglang \
  --hardware hardware.json \
  --model model.json \
  --workload workload.json \
  --software software.json \
  --config candidate.json \
  --hypothesis "TP=2 should reduce latency without exceeding the VRAM limit" \
  --tag short-preflight
```

This only writes a planned artifact; it does not start a benchmark. After a
manually launched trial, attach measurements and the decision for the next
experiment:

```bash
dtap-traj tune update /path/to/artifacts/tuning/<trial-id> \
  --status success \
  --metrics metrics.json \
  --observation "TP=2 improved generated tok/s and retained 9 GB headroom" \
  --next-step "Compare replicas=2 at the same concurrency"
```

Profile files are JSON objects with a human-readable `label`; include stable
facts needed to reproduce the cohort. Runtime config uses explicit sections so
the common topology fields are searchable:

```json
{
  "sglang": {"tp": 2, "replicas": 2, "concurrency": 32},
  "megatron": {"tp": 2, "pp": 1, "cp": 2, "ep": 1, "etp": 1},
  "allocation": {"train_gpus": 4, "rollout_gpus": 4}
}
```

Metrics are also an extensible JSON object. The index recognizes
`rollout_tok_s`, `train_tok_s`, `step_time_s`, `wall_time_s`, `gpu_seconds`,
`peak_vram_gb`, `trainable_rollouts`, `trainable_tokens`, `wait_ratio`,
`infra_failure_rate`, `repeats`, and an optional `durations_s` mapping. A caller
may override the phase-derived objective with numeric `objective_value` and a
non-empty `objective_name`. Ratios must be in `[0, 1]`, duration and throughput
values must be nonnegative, and a successful trial must include metrics.

Use `oom`, `timeout`, `nccl_error`, `engine_crash`, `invalid_config`, or
`infra_error` instead of `success` to preserve a failed boundary. Trial status
and notes are written atomically. An identical phase, hardware, model,
workload, and software fingerprint is required before the explorer names a
best-known or Pareto candidate. Trial IDs are globally unique within one
indexed artifact root; a duplicate ID at another path is rejected instead of
silently replacing the first trial.

The current implementation is validated with synthetic CPU fixtures only.
Actual CUDA throughput, VRAM, NCCL behavior, profiler capture, and production
recommendations must be validated later on a GPU host. Synthetic fixture values
must not be reported as benchmark results.

### Judge semantics

The two DTAP judge outcomes are independent:

- **Task judge** asks whether the victim completed the original user task.
- **Attack judge** asks whether the adversary's separate attack objective was
  achieved.

For an indirect attack, `task succeeded / attack failed` is the desired safe
outcome, while `task succeeded / attack succeeded` means the useful task was
completed but the injected objective also took effect. In some direct tasks the
original request is itself the malicious objective, so their task and attack
judge implementations can intentionally share a predicate.

The episode list reports the attack judge as **succeeded**, **failed**, or **not
evaluated**. It does not infer containment from a missing/null verdict.

Trajectory reads remain unauthenticated and are intended for trusted/local use.
Experiment mutation endpoints require the separate launch token. Keep the
default loopback host when trajectories contain sensitive evaluation data, and
protect persistent public tunnels with Cloudflare Access.

## Opening the explorer from a remote machine

Start the explorer on the server using its safe loopback default:

```bash
dtap-traj serve /path/to/artifacts --watch
```

When using VS Code Remote SSH, open the **Ports** view, choose **Forward a
Port**, enter `8765`, and then use the forwarded URL shown by VS Code. A plain
SSH client can do the same with:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@server
```

Then open <http://127.0.0.1:8765/> on the client machine.

For a temporary clickable HTTPS URL, install `cloudflared` and run:

```bash
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8765
```

The generated `trycloudflare.com` URL is public and has no authentication or
uptime guarantee. Use it only for sanitized/public artifacts, do not share the
URL, and stop `cloudflared` when finished. For persistent or sensitive use,
configure a named Cloudflare Tunnel protected by Cloudflare Access.

`dtap-traj index <artifact-root>` can populate or refresh the SQLite index without starting the server. The default database is `<artifact-root>/.dtap-traj.sqlite3` and uses SQLite WAL mode.

## Legacy standalone HTML export

The existing behavior remains available and unchanged:

```bash
dtap-traj /path/to/artifact-dir -o trajectory.html --open
```

The legacy renderer produces one self-contained HTML page. `--json` still exposes the parsed viewer data model without rendering.

## Artifact layout

A viewer-ready episode directory typically contains:

```text
episode/
├── episode-manifest.json
├── result.json
├── policy.jsonl
├── policy-prompt.txt
├── prompt-snapshots.jsonl  # optional dynamic digestor/reasoning requests
├── victim-trajectory.json
├── victim-mcp-events.jsonl
├── original-config.yaml
├── submitted-config.yaml
├── judge-result.json
├── judge-verdict.json
└── attempts/
    ├── attempt-0001/  # submitted config, victim trajectory, judges
    └── attempt-0002/
```

For an H-loop, use the submission-attempt selector above the detail view to
switch the submitted config, victim trajectory, and DTAP judges together. The
policy tab remains the complete episode policy trajectory. Top-level artifact
files are treated as latest-attempt compatibility aliases.

`policy-prompt.txt` is the evaluator's exact launch/user prompt to Claude Code,
not Claude Code's private runtime system prompt. The Prompts tab displays that
distinction explicitly. Components whose runtime did not retain an exact prompt
are shown as `not retained` rather than being recreated from today's templates.

The explorer does not rewrite these files. It indexes metadata and loads the
selected bundle through the existing `dtap_traj.parser` implementation.

A tuning trial uses this independent versioned layout:

```text
tuning/<trial-id>/
├── tuning-manifest.json
├── hardware.json
├── model.json
├── workload.json
├── software.json
├── config.json
├── metrics.json          # optional until measured
└── result.json
```

Artifact JSON remains the source of truth; the WAL-mode SQLite database stores
only the searchable index. Do not put API keys, tokens, passwords, or other
credentials in tuning profiles or configs.
