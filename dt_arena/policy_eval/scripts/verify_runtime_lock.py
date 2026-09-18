"""Verify the pinned M6 DTAP/OpenClaw release runtime without printing secrets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path


def _run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def verify(lock_path: Path, dtap_root: Path, *, check_images: bool) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["dtap_commit"] = _run("git", "rev-parse", "HEAD", cwd=dtap_root) == lock["dtap_commit"]
    checks["python_version"] = platform.python_version() == lock["python_version"]
    openclaw = shutil.which("openclaw")
    version = _run(openclaw, "--version") if openclaw else ""
    match = re.search(r"OpenClaw\s+([^\s]+)", version)
    checks["openclaw_version"] = bool(match and match.group(1) == lock["openclaw_version"])
    for tool, expected in lock.get("system_tools", {}).items():
        executable = shutil.which(tool)
        actual_version = _run(executable, "--version") if executable else ""
        checks[f"system_tool:{tool}:version"] = actual_version == expected["version"]
        actual_digest = ""
        if executable:
            actual_digest = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
        checks[f"system_tool:{tool}:sha256"] = actual_digest == expected["sha256"]
    for package, expected in lock["python_packages"].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = ""
        checks[f"python_package:{package}"] = actual == expected
    if check_images:
        for tag, digest in lock["container_images"].items():
            try:
                payload = json.loads(_run("docker", "image", "inspect", tag))[0]
                actual = payload.get("RepoDigests") or []
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                actual = []
            checks[f"container_image:{tag}"] = digest in actual
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtap-root", type=Path, required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).parents[1] / "runtime-lock.json",
    )
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    result = verify(args.lock.resolve(), args.dtap_root.resolve(), check_images=not args.skip_images)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
