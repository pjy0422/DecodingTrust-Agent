# Current policy-evaluation baseline

Source provenance:

- slime source commit: `f1e8ee90fe725d394b4e9929947423ecaa01976c`
- DTAP migration base: `e0323a521ba4ef88f8e14c1eccf68d0a3d19a458`
- canonical source baseline expected by the migrator: `f1e8ee90fe725d394b4e9929947423ecaa01976c`
- canonical DTAP base expected by the migrator: `e0323a521ba4ef88f8e14c1eccf68d0a3d19a458`
- retained latest E2E artifact: `p3-holdout-live-20260909`
- retained prompt SHA-256: `94422599f87769747e90f184022f25bd123dd227177929fa701a05afd2580380`

Current execution behavior is deliberately frozen during migration:

- Claude Code CLI is invoked with `-p`, `--output-format stream-json`,
  `--verbose`, `--max-turns`, strict MCP config, explicit settings,
  explicit allow/deny tool sets, and `--model`.
- placement-enabled policy surface has six MCP tools:
  `get_task_spec`, `get_attack_surface`, `validate_attack_step`,
  `apply_attack_step`, `validate_placement`, `submit_attack`.
- H counts victim executions. Invalid submissions consume Q but not H.
- default live feedback is `final+deterministic`.
- the policy path preserves raw Claude Code stdout as `policy.jsonl`.
- the real DTAP victim/judge path remains subprocess-isolated.

This document is the control-group definition for later ReAct,
reflection, or plan/execute research. Strategy research must not silently
mutate the current baseline.
