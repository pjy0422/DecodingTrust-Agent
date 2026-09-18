"""Whole-task manifests and no-follow copies for M4."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path


class IntegrityError(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class ManifestEntry:
    relative_path: str
    kind: str
    mode: int
    size: int
    sha256: str | None


@dataclass(frozen=True)
class BenchmarkManifest:
    entries: tuple[ManifestEntry, ...]
    root_digest: str


def _hash_regular_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    try:
        fd = os.open(path, flags)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise IntegrityError("task entry is not a regular file")
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise IntegrityError("task file could not be read safely") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise IntegrityError("task file changed while hashing")
    return digest.hexdigest()


class BenchmarkIntegrityGuard:
    @staticmethod
    def capture(task_dir: Path | str) -> BenchmarkManifest:
        root_input = Path(task_dir)
        if root_input.is_symlink():
            raise IntegrityError("task root may not be a symlink")
        root = root_input.resolve()
        if not root.is_dir():
            raise IntegrityError("task root is not a directory")
        entries: list[ManifestEntry] = []
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                raise IntegrityError("task tree contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                entries.append(ManifestEntry(relative, "directory", mode, 0, None))
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise IntegrityError("task tree contains a multiply-linked file")
                entries.append(ManifestEntry(relative, "file", mode, info.st_size, _hash_regular_file(path)))
            else:
                raise IntegrityError("task tree contains a special file")
        serialized = json.dumps(
            [entry.__dict__ for entry in entries],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return BenchmarkManifest(tuple(entries), hashlib.sha256(serialized).hexdigest())

    @classmethod
    def verify(cls, task_dir: Path | str, expected: BenchmarkManifest) -> None:
        actual = cls.capture(task_dir)
        if not isinstance(expected, BenchmarkManifest) or actual != expected:
            raise IntegrityError("benchmark task manifest changed")


def _copy_regular_from_fd(
    source_directory_fd: int,
    name: str,
    destination: Path,
    expected_info: os.stat_result,
    mode: int,
) -> None:
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(name, source_flags, dir_fd=source_directory_fd)
    try:
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino) != (expected_info.st_dev, expected_info.st_ino)
        ):
            raise IntegrityError("source changed to an unsafe file")
        destination_fd = os.open(destination, destination_flags, mode)
        try:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
            os.fchmod(destination_fd, mode)
        finally:
            os.close(destination_fd)
        after = os.fstat(source_fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise IntegrityError("source changed while copying")
    finally:
        os.close(source_fd)


def safe_copy_tree(
    source_task_dir: Path | str,
    destination_task_dir: Path | str,
    *,
    expected_manifest: BenchmarkManifest | None = None,
) -> BenchmarkManifest:
    source_input = Path(source_task_dir)
    destination = Path(destination_task_dir)
    expected = expected_manifest or BenchmarkIntegrityGuard.capture(source_input)
    BenchmarkIntegrityGuard.verify(source_input, expected)
    if destination.exists() or destination.is_symlink():
        raise IntegrityError("copy destination already exists")
    destination.mkdir(parents=False, mode=0o700)

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

    def recurse(source_fd: int, target: Path) -> None:
        for entry in sorted(os.scandir(source_fd), key=lambda item: item.name):
            target_path = target / entry.name
            info = entry.stat(follow_symlinks=False)
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                raise IntegrityError("source task tree contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                target_path.mkdir(mode=mode)
                os.chmod(target_path, mode)
                child_fd = os.open(entry.name, directory_flags, dir_fd=source_fd)
                try:
                    opened = os.fstat(child_fd)
                    if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                        raise IntegrityError("source directory changed while copying")
                    recurse(child_fd, target_path)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode):
                _copy_regular_from_fd(source_fd, entry.name, target_path, info, mode)
            else:
                raise IntegrityError("source task tree contains a special file")

    try:
        root_fd = os.open(source_input, directory_flags)
        try:
            recurse(root_fd, destination)
        finally:
            os.close(root_fd)
    except OSError as exc:
        raise IntegrityError("source tree changed during no-follow copy") from exc
    os.chmod(destination, stat.S_IMODE(source_input.stat().st_mode))
    BenchmarkIntegrityGuard.verify(source_input, expected)
    copied = BenchmarkIntegrityGuard.capture(destination)
    if copied != expected:
        raise IntegrityError("copied task does not match the source manifest")
    return expected
