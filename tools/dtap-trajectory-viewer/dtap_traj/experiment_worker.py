"""Durable worker process for one viewer-submitted policy-eval matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .experiments import _atomic_json, _now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    record_path = args.job_dir / "job.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.update({"status": "running", "started_at": _now(), "updated_at": _now()})
    _atomic_json(record_path, record)
    command = [
        str(args.python),
        "-m",
        "dt_arena.policy_eval.scripts.run_domain_matrix",
        "--config",
        str(args.job_dir / "config.yaml"),
    ]
    log_path = args.job_dir / "runner.log"
    try:
        with log_path.open("ab", buffering=0) as log:
            completed = subprocess.run(
                command,
                cwd=args.runner_root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        status = "completed" if completed.returncode == 0 else "failed"
        record.update({"status": status, "exit_code": completed.returncode})
    except BaseException as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nworker failure: {type(exc).__name__}: {exc}\n")
        record.update({"status": "failed", "exit_code": None, "worker_error": str(exc)})
    record.update({"finished_at": _now(), "updated_at": _now()})
    _atomic_json(record_path, record)


if __name__ == "__main__":
    main()
