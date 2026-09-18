"""Small trusted-side helpers used by generate.py in M1 and later milestones."""

from __future__ import annotations

from contextlib import contextmanager

from .service import EpisodeRegistry, EpisodeView


@contextmanager
def registered_episode(registry: EpisodeRegistry, *, token: str, view: EpisodeView):
    """Register exactly for the policy lifetime and clean up on every exit path."""

    registry.register(token, view)
    try:
        yield
    finally:
        registry.unregister(token)
