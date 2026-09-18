"""Generic parser for trusted injection-tool placement metadata.

Argument constraints belong to each MCP tool's JSON Schema. Cross-action
resource semantics that JSON Schema cannot express are advertised through the
tool's MCP ``_meta`` field and parsed fail-closed here. This module deliberately
contains no domain names or tool-name allowlist.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_RESOURCE_KIND = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class PlacementResourceContract:
    kind: str
    role: str  # "provider" or "consumer"


def placement_resource_metadata(*, kind: str, role: str) -> dict[str, Any]:
    """Build the namespaced MCP metadata owned by an injection tool."""

    contract = PlacementResourceContract(kind=kind, role=role)
    _validate(contract)
    return {
        "dtap": {
            "placement": {
                "resource": {"kind": contract.kind, "role": contract.role},
            }
        }
    }


def parse_placement_resource_metadata(meta: Any) -> PlacementResourceContract | None:
    """Parse one exact, bounded resource declaration; malformed data is ignored."""

    if not isinstance(meta, Mapping):
        return None
    dtap = meta.get("dtap")
    placement = dtap.get("placement") if isinstance(dtap, Mapping) else None
    resource = placement.get("resource") if isinstance(placement, Mapping) else None
    if not isinstance(resource, Mapping) or set(resource) != {"kind", "role"}:
        return None
    contract = PlacementResourceContract(
        kind=resource.get("kind"),
        role=resource.get("role"),
    )
    try:
        _validate(contract)
    except (TypeError, ValueError):
        return None
    return contract


def _validate(contract: PlacementResourceContract) -> None:
    if not isinstance(contract.kind, str) or not _RESOURCE_KIND.fullmatch(contract.kind):
        raise ValueError("invalid placement resource kind")
    if contract.role not in {"provider", "consumer"}:
        raise ValueError("invalid placement resource role")
