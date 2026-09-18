"""Validated inventory for DTAP domain/threat-model evaluation matrices."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


MANIFEST_PATH = Path(__file__).with_name("benchmark_manifest.json")


def _load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported DTAP benchmark manifest schema")
    threat_models = value.get("threat_models")
    selection_profiles = value.get("selection_profiles")
    domains = value.get("domains")
    if (
        not isinstance(threat_models, list)
        or not threat_models
        or not all(isinstance(item, str) and item for item in threat_models)
        or len(set(threat_models)) != len(threat_models)
    ):
        raise ValueError("benchmark manifest has invalid threat models")
    if not isinstance(selection_profiles, Mapping) or not selection_profiles:
        raise ValueError("benchmark manifest has no selection profiles")
    profile_indices: set[int] = set()
    for name, profile in selection_profiles.items():
        if not isinstance(name, str) or not name or not isinstance(profile, Mapping):
            raise ValueError("benchmark manifest has an invalid selection profile")
        index = profile.get("benchmark_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index in profile_indices:
            raise ValueError(f"selection profile {name!r} has an invalid index")
        profile_indices.add(index)
    if not isinstance(domains, list) or not domains:
        raise ValueError("benchmark manifest has no domains")
    names: set[str] = set()
    for entry in domains:
        if not isinstance(entry, Mapping):
            raise ValueError("benchmark manifest domain must be an object")
        name = entry.get("name")
        platform = entry.get("platform")
        vm_backed = entry.get("vm_backed")
        default_enabled = entry.get("default_enabled")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("benchmark manifest has invalid or duplicate domain")
        if not isinstance(platform, str) or not platform:
            raise ValueError(f"benchmark manifest domain {name!r} has no platform")
        if not isinstance(vm_backed, bool) or not isinstance(default_enabled, bool):
            raise ValueError(f"benchmark manifest domain {name!r} has invalid gates")
        if vm_backed and default_enabled:
            raise ValueError(f"VM-backed domain {name!r} must remain explicit opt-in")
        names.add(name)
    return value


BENCHMARK_MANIFEST = _load_manifest()
MANIFEST_SHA256 = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
DOMAIN_ENTRIES = tuple(BENCHMARK_MANIFEST["domains"])
THREAT_MODELS = tuple(BENCHMARK_MANIFEST["threat_models"])
SELECTION_PROFILES = {
    str(name): int(profile["benchmark_index"]) for name, profile in BENCHMARK_MANIFEST["selection_profiles"].items()
}
ALL_DOMAINS = tuple(str(entry["name"]) for entry in DOMAIN_ENTRIES)
DOMAINS = tuple(str(entry["name"]) for entry in DOMAIN_ENTRIES if entry["default_enabled"])
EXCLUDED_PLATFORM_DOMAINS = frozenset(str(entry["name"]) for entry in DOMAIN_ENTRIES if entry["vm_backed"])


def matrix_cases(
    domains: Iterable[str] = DOMAINS,
    threat_models: Iterable[str] = THREAT_MODELS,
) -> tuple[tuple[str, str], ...]:
    """Return validated matrix coordinates in stable manifest order."""
    selected_domains = tuple(domains)
    selected_threat_models = tuple(threat_models)
    unknown_domains = set(selected_domains).difference(ALL_DOMAINS)
    unknown_threat_models = set(selected_threat_models).difference(THREAT_MODELS)
    if unknown_domains or unknown_threat_models:
        raise ValueError(
            "unknown benchmark coordinates: "
            f"domains={sorted(unknown_domains)}, "
            f"threat_models={sorted(unknown_threat_models)}"
        )
    return tuple(
        (domain, threat_model)
        for domain in ALL_DOMAINS
        if domain in selected_domains
        for threat_model in THREAT_MODELS
        if threat_model in selected_threat_models
    )


ALL_CASES = matrix_cases(ALL_DOMAINS, THREAT_MODELS)
DEFAULT_CASES = matrix_cases()
