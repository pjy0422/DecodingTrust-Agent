"""Retain exact non-secret model prompts for trajectory inspection."""

from __future__ import annotations

import inspect
import json
import os
import threading
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .artifact_contract import PROMPT_SNAPSHOTS


class PromptSnapshotWriter:
    """Append exact prompt requests without retaining provider credentials."""

    def __init__(self, artifact_root: Path | None) -> None:
        self.path = artifact_root / PROMPT_SNAPSHOTS if artifact_root is not None else None
        self._lock = threading.Lock()
        self._sequence = 0
        self._attempt_index: ContextVar[int | None] = ContextVar(
            "dtap_prompt_snapshot_attempt_index",
            default=None,
        )

    @contextmanager
    def attempt(self, index: int | None):
        token = self._attempt_index.set(index)
        try:
            yield
        finally:
            self._attempt_index.reset(token)

    def record(
        self,
        *,
        component: str,
        label: str,
        role: str,
        prompt: str,
        source: str,
    ) -> None:
        if self.path is None:
            return
        if role not in {"system", "user"}:
            raise ValueError("prompt role must be system or user")
        with self._lock:
            self._sequence += 1
            payload = {
                "schema": "dtap-policy-eval-prompt-snapshot",
                "schema_version": 1,
                "sequence": self._sequence,
                "component": component,
                "label": label,
                "role": role,
                "prompt": prompt,
                "source": source,
                "exact": True,
            }
            attempt_index = self._attempt_index.get()
            if attempt_index is not None:
                payload["attempt_index"] = attempt_index
            encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("could not append prompt snapshot")
                    remaining = remaining[written:]
            finally:
                os.close(descriptor)

    def wrap(
        self,
        complete: Callable[[str], Awaitable[Any] | Any],
        *,
        component: str,
        label: str,
        role: str = "user",
        source: str,
    ) -> Callable[[str], Awaitable[Any]]:
        async def recorded(prompt: str) -> Any:
            self.record(
                component=component,
                label=label,
                role=role,
                prompt=prompt,
                source=source,
            )
            result = complete(prompt)
            return await result if inspect.isawaitable(result) else result

        return recorded
