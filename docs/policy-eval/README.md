# Policy evaluation

`dt_arena.policy_eval` is the DTAP-owned research runtime for evaluating an
attack-planning policy against the real DTAP victim + judge stack.

The current baseline launches **Claude Code CLI directly on the host**. It
does not use the historical slime `M6ClaudeCodeHarness`. Claude Code receives
only the DTAP policy MCP surface; native Bash/file/web tools are denied.

Execution flow:

```text
Claude Code policy
    -> policy-scoped DTAP MCP
       -> task / attack-surface projection
       -> step validation
       -> environment apply + placement receipt (when applicable)
          -> explicit replay of owned, positively validated prerequisites
       -> bounded submit transaction
          -> fresh DTAP victim subprocess
          -> judge
          -> bounded feedback on every failed H, including terminal H
    -> artifact v1 bundle
    -> tools/dtap-trajectory-viewer
```

Placement failures distinguish locator metadata from actionable repairs.
`diagnostic.locator_fields` only explains how the requested target was
addressed. A policy may retry only when `diagnostic.retryable` is true and must
then change only `repair.fields`. Generated-resource consumers can pass prior
owned receipt ids through `apply_attack_step(..., depends_on=[...])`; DTAP
replays only compatible verified providers in the same fresh sandbox.

Injection MCPs own their placement contract; policy-eval does not maintain a
domain/tool-name table. Constrained arguments must be expressed in the tool's
normal JSON Schema (for Python tools, prefer `Literal` or an enum). A tool that
creates or consumes a generated resource may additionally publish trusted MCP
metadata such as:

```json
{"dtap":{"placement":{"resource":{"kind":"example.record","role":"provider"}}}}
```

Use `role: "consumer"` on the matching consumer. The placement adapter for
both tools must return the same canonical locator for the same resource. DTAP
replays only an explicitly owned and validated provider receipt, then compares
the adapter-proven locators; malformed or unknown metadata fails closed. This
is the extension point for other domains and is not exposed to policy code.

Public source names describe behavior (`EvaluationSecurityPolicy`,
`create_policy_mcp_server`, `run_policy_e2e`, `run_domain_matrix`) rather
than historical development milestones.

## Commands

```bash
python -m dt_arena.policy_eval run --help
python -m dt_arena.policy_eval matrix --help
python -m dt_arena.policy_eval holdout --help
```

For repeatable live matrices, use a strict versioned experiment file instead
of a long command line:

```bash
export ANTHROPIC_API_KEY=...  # credentials remain environment-only
python -m dt_arena.policy_eval matrix \
  --config dt_arena/policy_eval/configs/finance-travel-indirect-openclaw.yaml
```

The `budgets` section controls three independent limits:

- `h_victim_executions`: H, consumed only when a victim execution starts.
- `q_submit_calls`: Q, consumed by every `submit_attack` call, including an
  invalid submission. Q must be at least H.
- `max_placement_actions`: consumed by `apply_attack_step`; placement
  validation itself does not consume this budget.

After every accepted H execution, the policy emits a concise honest report.
Two optional report fields are controlled independently under `policy`:

- `improvement_wishes`: add a non-binding wish about policy-facing harness
  tools, feedback, budgets, or observability. The policy is explicitly told it
  cannot modify victim-side prompts, tools, harness, environment, or judge.
- `dying_message`: on the final H or another terminal receipt, add a concise
  evidence/dead-end handoff for a future policy.

When either value is `false`, its instructions are absent from the policy
prompt rather than merely asking the model to leave the field blank.

Hosted reasoning models may need a larger digestor output budget than ordinary
JSON-only models. Configure it separately with
`feedback.digestor_max_tokens`; this does not change policy or victim token
budgets. For `deepseek-v4.1-flash`, the checked-in example uses 8,000 tokens
and a 120-second timeout because a retained E2E digest was truncated at the
legacy 2,500-token limit and completed successfully at 8,000.

Feedback schema v3 adds one bounded attribution for every submitted step:
semantic effect, a closed reason-class vocabulary, confidence, and references
to deterministic evidence. OpenClaw records payload inclusion at its
`llm_input` provider boundary; retained observations contain only step ids and
booleans, while the ephemeral exact-match probe file is removed during agent
cleanup. An apparent authority-channel/surface mismatch is written separately
to `research-feedback.json`. That metric is researcher-only and never crosses
the policy response contract.

`policy.max_turns: auto` resolves to `max(64, 32 * H)`. Relative DTAP and
artifact paths are resolved from the YAML file. Explicit CLI options override
YAML values. Every matrix writes the effective, absolute, credential-free configuration to
`experiment-config.resolved.yaml` in its artifact root, alongside
`summary.json`. The schema is closed: unknown fields (including attempted API
keys, provider URLs, or arbitrary environment variables) are rejected.

The standalone viewer remains its own installable tool:

```bash
pip install -e tools/dtap-trajectory-viewer
dtap-traj --help
```
