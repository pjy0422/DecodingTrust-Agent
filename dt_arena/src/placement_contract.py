"""Generic parser for trusted injection-tool placement metadata.

Argument constraints belong to each MCP tool's JSON Schema. Cross-action
resource semantics that JSON Schema cannot express are advertised through the
tool's MCP ``_meta`` field and parsed fail-closed here. This module deliberately
contains no domain names or tool-name allowlist.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence


_RESOURCE_KIND = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_FAILURE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_ARGUMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


@dataclass(frozen=True)
class PlacementResourceContract:
    kind: str
    role: str  # "provider" or "consumer"


@dataclass(frozen=True)
class PlacementFailureContract:
    code: str
    repair_fields: tuple[str, ...]


def placement_resource_metadata(
    *, kind: str, role: str,
    failure_repairs: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Build the namespaced MCP metadata owned by an injection tool."""

    contract = PlacementResourceContract(kind=kind, role=role)
    _validate(contract)
    placement = {
        "resource": {"kind": contract.kind, "role": contract.role},
    }
    if failure_repairs:
        placement["failure_repairs"] = _serialize_failure_repairs(failure_repairs)
    return {"dtap": {"placement": placement}}


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


def placement_failure_result(
    result: Mapping[str, Any], *, code: str, repair_fields: Sequence[str],
) -> dict[str, Any]:
    """Attach a trusted adapter's bounded repair declaration to a result."""

    contract = PlacementFailureContract(code=code, repair_fields=tuple(repair_fields))
    _validate_failure(contract)
    value = dict(result)
    value["dtap"] = {"placement": {"failure": {
        "code": contract.code,
        "repair_fields": list(contract.repair_fields),
    }}}
    return value


def parse_placement_failure_result(value: Any) -> PlacementFailureContract | None:
    """Parse an exact adapter-authored failure declaration, failing closed."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    while isinstance(value, Mapping) and set(value) == {"result"}:
        value = value["result"]
    if not isinstance(value, Mapping):
        return None
    dtap = value.get("dtap")
    placement = dtap.get("placement") if isinstance(dtap, Mapping) else None
    failure = placement.get("failure") if isinstance(placement, Mapping) else None
    if not isinstance(failure, Mapping) or set(failure) != {"code", "repair_fields"}:
        return None
    fields = failure.get("repair_fields")
    if not isinstance(fields, list):
        return None
    contract = PlacementFailureContract(failure.get("code"), tuple(fields))
    try:
        _validate_failure(contract)
    except (TypeError, ValueError):
        return None
    return contract


def parse_placement_failure_metadata(meta: Any) -> dict[str, tuple[str, ...]]:
    """Return exact repair capabilities statically declared by one MCP tool."""

    if not isinstance(meta, Mapping):
        return {}
    dtap = meta.get("dtap")
    placement = dtap.get("placement") if isinstance(dtap, Mapping) else None
    repairs = placement.get("failure_repairs") if isinstance(placement, Mapping) else None
    if not isinstance(repairs, Mapping) or not repairs:
        return {}
    parsed = {}
    try:
        for code, fields in repairs.items():
            if not isinstance(fields, list):
                return {}
            contract = PlacementFailureContract(code, tuple(fields))
            _validate_failure(contract)
            parsed[contract.code] = contract.repair_fields
    except (TypeError, ValueError):
        return {}
    return parsed


def _validate(contract: PlacementResourceContract) -> None:
    if not isinstance(contract.kind, str) or not _RESOURCE_KIND.fullmatch(contract.kind):
        raise ValueError("invalid placement resource kind")
    if contract.role not in {"provider", "consumer"}:
        raise ValueError("invalid placement resource role")


def _validate_failure(contract: PlacementFailureContract) -> None:
    if not isinstance(contract.code, str) or not _FAILURE_CODE.fullmatch(contract.code):
        raise ValueError("invalid placement failure code")
    fields = contract.repair_fields
    if not fields or len(fields) > 8 or len(set(fields)) != len(fields):
        raise ValueError("invalid placement repair fields")
    if any(not isinstance(field, str) or not _ARGUMENT_NAME.fullmatch(field) for field in fields):
        raise ValueError("invalid placement repair field")


def _serialize_failure_repairs(
    repairs: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    serialized = {}
    for code, fields in repairs.items():
        contract = PlacementFailureContract(code, tuple(fields))
        _validate_failure(contract)
        serialized[contract.code] = list(contract.repair_fields)
    if not serialized:
        raise ValueError("placement failure repairs cannot be empty")
    return serialized
