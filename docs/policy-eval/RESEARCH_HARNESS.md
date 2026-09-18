# Research harness boundary

The migrated runtime has three explicit layers:

```text
episode / evaluation orchestration
        -> PlanningStrategy
        -> PolicyHarness
           -> ClaudeCodeHarness
```

`ClaudeCodeHarness` owns only process transport: Claude CLI argv, cwd,
environment, MCP config, tool allow/deny lists, model, timeout, and raw
stream-json capture. It knows nothing about ReAct, reflection, placement,
H/Q budgets, victim execution, or judges.

`PlanningStrategy` owns reasoning behavior. `CurrentPlanningStrategy` is
the frozen control and emits the byte-identical latest E2E prompt. A ReAct
strategy can normally override only `build_prompt()`. A reflection or
plan/execute strategy that needs multiple explicit phases can override
`execute()` and call the same `PolicyHarness` repeatedly while the same
episode MCP authority remains alive.

Placement receipts, H/Q budgets, leakage rules, victim isolation, judge
execution, and artifact production stay in the episode/evaluation layer.
Therefore strategy experiments do not require a new harness implementation.
