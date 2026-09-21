"""Run the disjoint holdout policy-evaluation matrix.

This is the semantic successor to the historical P3 holdout gate.  Runtime
compatibility is already native in the DTAP checkout; this command never applies
an external overlay and has no slime-root argument.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dt_arena.policy_eval.protocol import HARNESS_PROTOCOLS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtap-root", required=True, type=Path)
    parser.add_argument("--artifacts-root", required=True, type=Path)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--retry-passes", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    parser.add_argument("--policy-model", default="deepseek-v4-flash")
    parser.add_argument("--planning-strategy", default="current")
    parser.add_argument(
        "--harness-protocol",
        choices=HARNESS_PROTOCOLS,
        default="v1",
    )
    parser.add_argument("--victim-model", default="deepseek-v4-flash")
    parser.add_argument(
        "--feedback-mode",
        choices=("disabled", "final", "final+deterministic", "final+deterministic+digestor"),
        default="final+deterministic",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for the holdout matrix")
    dtap = args.dtap_root.expanduser().resolve()
    artifacts = args.artifacts_root.expanduser().resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(part for part in (str(dtap), env.get("PYTHONPATH", "")) if part)
    command = [
        sys.executable,
        "-m",
        "dt_arena.policy_eval.scripts.run_domain_matrix",
        "--dtap-root",
        str(dtap),
        "--artifacts-root",
        str(artifacts),
        "--selection-profile",
        "holdout-v1",
        "--max-parallel",
        str(args.max_parallel),
        "--policy-model",
        args.policy_model,
        "--planning-strategy",
        args.planning_strategy,
        "--harness-protocol",
        args.harness_protocol,
        "--victim-model",
        args.victim_model,
        "--victim-agent-type",
        "openclaw",
        "--max-submissions",
        "2",
        "--feedback-mode",
        args.feedback_mode,
    ]
    if args.resume:
        command.append("--resume")
    parallel_index = command.index("--max-parallel") + 1
    completed = subprocess.run(command, cwd=dtap, env=env, check=False)
    retry_parallel = args.max_parallel
    for _ in range(args.retry_passes):
        if completed.returncode == 0:
            break
        time.sleep(max(0.0, args.retry_delay))
        retry_parallel = max(1, retry_parallel // 2)
        command[parallel_index] = str(retry_parallel)
        if "--resume" not in command:
            command.append("--resume")
        completed = subprocess.run(command, cwd=dtap, env=env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
