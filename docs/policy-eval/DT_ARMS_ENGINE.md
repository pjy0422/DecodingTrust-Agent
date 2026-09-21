# DT Arms policy engine

`policy.engine: dt-arms-upstream` runs the vendored upstream DT Arms subtree as
the attack policy. The snapshot is copied without source edits from
`AI-secure/DecodingTrust-Agent` branch `yuzhou/macos-osworld-judges-v2`; exact
commit, Git tree, and a checkout-verifiable content digest are recorded in
`dt_arena/policy_eval/dt_arms_snapshot.json`.

The boundary is intentionally two-stage:

1. `python -m dt_arms.run` performs its original native loop. It reads the
   normal DTAP task/config, uses its own skills and attacker reasoning, runs its
   own victim and judge feedback, and may stop early or exhaust
   `dt_arms.max_iterations`.
2. Only a successful `attack_result_*.yaml` crosses the boundary. Integration
   code rejects ambiguous output, changed task identity or malicious goal, and
   invalid DTAP syntax. It copies only the generated `Attack.attack_turns` into
   an isolated copy of the original task.
3. The candidate receives one fresh victim/judge execution through
   `DtapAttemptRunner`. This replay is the authoritative ASR result shown by
   the matrix and trajectory viewer. DT Arms' scouting verdict is retained as
   provenance but is not counted as the final benchmark verdict.

The integration layer lives outside `dt_arms/` so the pinned source stays
auditable. It does not translate the native loop into Claude Code, the policy
MCP protocol, or a placement-receipt workflow.

Example:

```bash
python -m dt_arena.policy_eval matrix \
  --config dt_arena/policy_eval/configs/finance-travel-indirect-openclaw.yaml
```

The viewer's **Policy engine** dropdown selects `DT Arms upstream` by default.
Claude Code remains available as the legacy policy-MCP control. DT Arms uses
the upstream provider environment described in `docs/policy-eval/README.md`;
secrets must not be stored in experiment YAML.
