# Artifact contract v1

Migration preserves the existing viewer contract byte-for-byte where it
matters. The historical schema identifier `dtap-agent-rl-episode` remains
version 1 even though the owning runtime is now `dt_arena.policy_eval`.

```text
episode/
├── episode-manifest.json
├── episode-state.json
├── result.json
├── policy.jsonl
├── policy.stderr.log
├── policy-prompt.txt
├── original-config.yaml
├── submitted-config.yaml
├── victim-trajectory.json
├── victim-mcp-events.jsonl
├── judge-result.json
├── judge-verdict.json
└── attempts/
    └── attempt-0001/
        ├── submitted-config.yaml
        ├── victim-trajectory.json
        ├── victim-mcp-events.jsonl
        ├── judge-result.json
        └── judge-verdict.json
```

Top-level attempt artifacts remain compatibility aliases for the latest
retained attempt. The viewer treats the artifact directory as source of
truth; its SQLite database is only an index.

For `policy_engine: claude-code`, do not normalize or recreate
`policy.jsonl`: it is raw Claude Code `stream-json` stdout and is part of
experiment reproducibility.

For `policy_engine: dt-arms-upstream`, the untouched native trajectory is
retained as `dt-arms/trajectory.json`; `policy.jsonl` is a deterministic
viewer adapter generated from that file. The generated attack is retained as
`dt-arms/attack-result.yaml`. If candidate generation succeeds, the ordinary
top-level and `attempts/attempt-0001/` victim/judge artifacts contain the one
fresh authoritative DTAP replay, not one of DT Arms' internal scouting runs.
