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
          -> bounded feedback on non-terminal miss
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

The standalone viewer remains its own installable tool:

```bash
pip install -e tools/dtap-trajectory-viewer
dtap-traj --help
```
