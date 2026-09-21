"""Versioned policy-facing MCP protocol names."""

from __future__ import annotations


HARNESS_PROTOCOL_V1 = "v1"
HARNESS_PROTOCOL_LAZY_SCHEMA_V2 = "lazy-schema-v2"
HARNESS_PROTOCOLS = (HARNESS_PROTOCOL_V1, HARNESS_PROTOCOL_LAZY_SCHEMA_V2)


def validate_harness_protocol(value: str) -> str:
    if value not in HARNESS_PROTOCOLS:
        raise ValueError(f"unsupported policy harness protocol: {value!r}")
    return value
