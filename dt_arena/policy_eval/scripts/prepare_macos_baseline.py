"""Prepare DTAP's snapshot-only macOS image for parallel cold boots.

The published qcow2 stores the installed system in an internal ``booted``
snapshot.  External qcow2 overlays hide internal snapshots, while loading the
VM-state snapshot directly is host-CPU dependent.  This command copies the
image, applies only the snapshot's disk state, removes its saved VM state, and
leaves a cold-bootable baseline that DTAP can safely share read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


VERSION = "14"
REQUIRED_COMPANIONS = (
    "base.dmg",
    "macos.id",
    "macos.mac",
    "macos.mlb",
    "macos.rom",
    "macos.sn",
    "macos.vars",
    "macos_vars.qcow2",
)


def normalize_data_root(path: Path) -> Path:
    """Accept either the data root or its version subdirectory."""
    resolved = path.expanduser().resolve()
    if (resolved / "data.qcow2").is_file():
        return resolved.parent
    if (resolved / VERSION / "data.qcow2").is_file():
        return resolved
    raise FileNotFoundError(
        f"expected data.qcow2 at {resolved / 'data.qcow2'} or " f"{resolved / VERSION / 'data.qcow2'}"
    )


def _qemu_img(
    image: str,
    root: Path,
    *arguments: str,
    read_only: bool = False,
) -> subprocess.CompletedProcess[str]:
    mount = f"{root}:/data" + (":ro" if read_only else "")
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "qemu-img",
            "-v",
            mount,
            image,
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def snapshot_names(root: Path, image: str) -> set[str]:
    result = _qemu_img(
        image,
        root,
        "snapshot",
        "-l",
        f"/data/{VERSION}/data.qcow2",
        read_only=True,
    )
    names: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit():
            names.add(fields[1])
    return names


def prepare(source: Path, output: Path, image: str) -> Path:
    source_root = normalize_data_root(source)
    output_root = output.expanduser().resolve()
    if source_root == output_root:
        raise ValueError("output must differ from the source image root")
    if "booted" not in snapshot_names(source_root, image):
        raise RuntimeError("source qcow2 has no 'booted' internal snapshot")
    if output_root.exists():
        try:
            ready_root = normalize_data_root(output_root)
        except FileNotFoundError:
            ready_root = None
        if ready_root and "booted" not in snapshot_names(ready_root, image):
            return ready_root
        raise FileExistsError(f"output exists but is not a prepared baseline: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-",
            dir=output_root.parent,
        )
    )
    try:
        source_version = source_root / VERSION
        temp_version = temp_root / VERSION
        temp_version.mkdir()
        shutil.copy2(source_version / "data.qcow2", temp_version / "data.qcow2")
        for name in REQUIRED_COMPANIONS:
            path = source_version / name
            if not path.is_file():
                raise FileNotFoundError(f"missing macOS companion image: {path}")
            shutil.copy2(path, temp_version / name)

        _qemu_img(
            image,
            temp_root,
            "snapshot",
            "-a",
            "booted",
            f"/data/{VERSION}/data.qcow2",
        )
        _qemu_img(
            image,
            temp_root,
            "snapshot",
            "-d",
            "booted",
            f"/data/{VERSION}/data.qcow2",
        )
        if snapshot_names(temp_root, image):
            raise RuntimeError("prepared qcow2 unexpectedly retains VM snapshots")

        stat = (source_version / "data.qcow2").stat()
        (temp_root / "prepared-baseline.json").write_text(
            json.dumps(
                {
                    "schema": "dtap-macos-cold-boot-baseline",
                    "schema_version": 1,
                    "source_size": stat.st_size,
                    "source_mtime_ns": stat.st_mtime_ns,
                    "source_snapshot": "booted",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp_root, output_root)
        return output_root
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-image", default="decodingtrustagent/macos")
    args = parser.parse_args()
    prepared = prepare(args.source, args.output, args.docker_image)
    print(prepared)


if __name__ == "__main__":
    main()
