"""Trusted host-side episode state for the read-only M1 MCP server."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .attack_surface import AttackSurface, ToolCatalogProvider, build_attack_surface
from .episode import TaskSnapshot
from .task_projection import PolicyTaskSpec, ProjectionPolicy, project_task


class EpisodeAccessError(PermissionError):
    """Uniform auth/lookup failure; do not reveal whether a token ever existed."""


@dataclass(frozen=True)
class EpisodeView:
    """The complete M1 policy-visible state for one rollout."""

    task: PolicyTaskSpec
    attack_surface: AttackSurface


async def build_episode_view(
    snapshot: TaskSnapshot,
    provider: ToolCatalogProvider,
    *,
    projection_policy: ProjectionPolicy | None = None,
) -> EpisodeView:
    """Build an immutable read-only view before the policy starts."""

    if projection_policy is None:
        projection_policy = ProjectionPolicy()

    snapshot.assert_config_unchanged()
    task = project_task(snapshot, policy=projection_policy)
    surface = await build_attack_surface(snapshot, provider)
    snapshot.assert_config_unchanged()
    return EpisodeView(task=task, attack_surface=surface)


class EpisodeRegistry:
    """Thread-safe in-memory capability map: opaque token -> sanitized view.

    Registration/unregistration are trusted-side operations only. The MCP tools
    never enumerate tokens and never expose this object to policy code.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._episodes: dict[str, EpisodeView] = {}

    def register(self, token: str, view: EpisodeView) -> None:
        token = _validate_token(token)
        with self._lock:
            if token in self._episodes:
                raise ValueError("episode token already registered")
            self._episodes[token] = view

    def resolve(self, token: str) -> EpisodeView:
        token = _validate_token(token)
        with self._lock:
            view = self._episodes.get(token)
        if view is None:
            raise EpisodeAccessError("unauthorized episode")
        return view

    def unregister(self, token: str) -> None:
        token = _validate_token(token)
        with self._lock:
            self._episodes.pop(token, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._episodes)


def _validate_token(token: str) -> str:
    if not isinstance(token, str) or len(token.strip()) < 16:
        raise EpisodeAccessError("unauthorized episode")
    return token.strip()
