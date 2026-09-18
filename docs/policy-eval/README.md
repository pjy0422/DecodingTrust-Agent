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
       -> bounded submit transaction
          -> fresh DTAP victim subprocess
          -> judge
          -> bounded feedback on non-terminal miss
    -> artifact v1 bundle
    -> tools/dtap-trajectory-viewer
```

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
