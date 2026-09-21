# Current policy-evaluation baseline

This file freezes the default `policy.engine: claude-code` control group. The
opt-in native DT Arms engine is documented separately in `DT_ARMS_ENGINE.md`;
selecting it does not alter the Claude prompt or its parity fixtures.

Source provenance:

- slime source commit: `f1e8ee90fe725d394b4e9929947423ecaa01976c`
- DTAP migration base: `e0323a521ba4ef88f8e14c1eccf68d0a3d19a458`
- canonical source baseline expected by the migrator: `f1e8ee90fe725d394b4e9929947423ecaa01976c`
- canonical DTAP base expected by the migrator: `e0323a521ba4ef88f8e14c1eccf68d0a3d19a458`
- retained latest E2E artifact: `p3-holdout-live-20260909`
- current prompt SHA-256: `aa928d060212afe3ae792049db55eca0a374885f72c47db40452ea48ef25504f`

Current execution behavior is versioned and regression-tested:

- Claude Code CLI is invoked with `-p`, `--output-format stream-json`,
  `--verbose`, `--max-turns`, strict MCP config, explicit settings,
  explicit allow/deny tool sets, and `--model`.
- placement-enabled policy surface has six MCP tools:
  `get_task_spec`, `get_attack_surface`, `validate_attack_step`,
  `apply_attack_step`, `validate_placement`, `submit_attack`.
- `policy.harness_protocol: v1` preserves that frozen six-tool control. The
  opt-in `lazy-schema-v2` protocol adds `get_tool_schema`, removes eager tool
  schemas from `get_attack_surface`, and requires target schema lookup before
  validating tool or environment steps. Existing public tool descriptions are
  preserved unchanged.
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
