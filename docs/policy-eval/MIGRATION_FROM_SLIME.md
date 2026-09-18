# Migration from slime

This is the only document where the old milestone names are treated as
primary provenance labels.

| Historical name | DTAP-owned semantic role |
| --- | --- |
| M4 | policy security boundary, submission budget, fresh-attempt isolation |
| M5 | trusted environment route/read-back verification |
| M6 | policy-scoped placement apply + opaque receipt validation |
| M7 | bounded failed-attempt feedback / observability |
| P3 holdout | disjoint holdout policy-evaluation matrix |

Migration source is pinned to `pjy0422/slime@f1e8ee90fe725d394b4e9929947423ecaa01976c`. Runtime
compatibility patches are applied once to the DTAP checkout and become
normal DTAP source changes; the migrated runtime never executes
`dtap_integration/apply.sh` and never imports from slime.

Compatibility strings such as `.m4-verdict.json`, `DTAP_M4_ATTEMPT_INDEX`,
and `m6-placement-v1` are intentionally retained in artifact/helper
boundaries for v1 parity. They are not public Python API names. Rename
them only together with an explicit artifact/protocol version bump.
