"""Trusted child entry point for one M6 environment-placement probe."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from dt_arena.src.env_verification import PlacementValidationError
from dt_arena.policy_eval.scripts.check_environment_verification import run


def _write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)


async def _main(task_dir: Path, result_path: Path) -> int:
    try:
        result = await run(task_dir, strict=False)
        placements = result.get("placements") if isinstance(result, dict) else None
        if not isinstance(placements, list) or len(placements) != 1:
            return 2
        proof = placements[0]
        status = proof.get("status")
        payload = {
            "schema": "m6-placement-v1",
            "applied": True,
            "valid": status == "verified",
            "status": status,
            "locator": proof.get("locator", ""),
            "code": "PLACEMENT_VERIFIED" if status == "verified" else "UNSUPPORTED_PLACEMENT",
        }
        _write_exclusive(result_path, payload)
        return 0
    except Exception as exc:
        # Only the structured DTAP placement exception is safe to cross the
        # child boundary. Everything else is an opaque infrastructure failure.
        if not isinstance(exc, PlacementValidationError):
            return 2
        payload = {
            "schema": "m6-placement-v1",
            "applied": getattr(exc, "code", "") != "INJECTION_FAILED",
            "valid": False,
            "status": "invalid",
            "locator": str(getattr(exc, "locator", "")),
            "code": str(getattr(exc, "code", "PLACEMENT_MISMATCH")),
            "repair_fields": list(getattr(exc, "repair_fields", ())),
        }
        _write_exclusive(result_path, payload)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    args = parser.parse_args()
    return asyncio.run(_main(args.task_dir.resolve(), args.result_path.resolve()))


if __name__ == "__main__":
    raise SystemExit(main())
