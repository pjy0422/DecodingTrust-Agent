# Current policy-evaluation baseline

Source provenance:

- slime source commit: `f1e8ee90fe725d394b4e9929947423ecaa01976c`
- DTAP migration base: `e0323a521ba4ef88f8e14c1eccf68d0a3d19a458`
- canonical source baseline expected by the migrator: `f1e8ee90fe725d394b4e9929947423ecaa01976c`
- canonical DTAP base expected by the migrator: `e0323a521ba4ef88f8e14c1eccf68d0a3d19a458`
- retained latest E2E artifact: `p3-holdout-live-20260909`
- current prompt SHA-256: `6139b4807bcfb9cc27d3db33edc2e4ba39f646596ae217e0fb320b02dfcec2c2`

Current execution behavior is versioned and regression-tested:

- Claude Code CLI is invoked with `-p`, `--output-format stream-json`,
  `--verbose`, `--max-turns`, strict MCP config, explicit settings,
  explicit allow/deny tool sets, and `--model`.
- placement-enabled policy surface has six MCP tools:
  `get_task_spec`, `get_attack_surface`, `validate_attack_step`,
  `apply_attack_step`, `validate_placement`, `submit_attack`.
- `apply_attack_step` accepts an optional list of owned, positively validated
  prerequisite qualified tool names. Internal receipt ids remain private.
  Dependencies are exact resource-contract matches,
  are replayed in the same fresh sandbox, and never grant arbitrary read-back.
- argument constraints and generated-resource roles are declared by each
  injection MCP through JSON Schema and namespaced MCP metadata. Policy-eval
  contains no domain-specific tool/value registry; resource identity is
  confirmed by comparing placement-adapter canonical locators after replay.
- placement diagnostics keep locator fields separate from adapter-proven,
  retryable repair fields.
- H counts victim executions. Invalid submissions consume Q but not H.
- default live feedback is `final+deterministic`.
- policy-visible feedback uses compact schema v4; detailed sanitized v3
  evidence is retained per attempt as `feedback-evidence.json`.
- every accepted H execution produces an honest report; optional harness wishes
  and terminal handoff messages are prompt-level feature flags.
- the policy path preserves raw Claude Code stdout as `policy.jsonl`.
- the real DTAP victim/judge path remains subprocess-isolated.

This document is the control-group definition for later ReAct, reflection, or
plan/execute research. Changes to the current strategy require an explicit
prompt-hash update and parity tests.
