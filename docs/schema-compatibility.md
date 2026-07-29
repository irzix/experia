# SQLite schema compatibility

Experia schema version 3 supports forward migration from every schema version in the inclusive window **0 through 3**. Version 0 denotes the recognized unversioned legacy database layout (`PRAGMA user_version = 0`), version 1 is the normalized source-record schema, version 2 adds embedding dimensions and the rebuildable vector-band structure, and version 3 adds versioned readiness and resumable rebuild cursor state for that derived index.

| Schema version | Status | Forward migration |
|---:|---|---|
| 0 | Supported legacy | Migrates through versions 1, 2, and 3 |
| 1 | Supported | Migrates through versions 2 and 3 |
| 2 | Supported | Migrates to version 3 |
| 3 | Current | No migration required |

The machine-readable source of truth is `tests/fixtures/sqlite/schema-support.json`. Its `support_window` must match `SUPPORTED_SCHEMA_VERSIONS` in `experia.memory.migrations`, and it lists a deterministic SQL fixture plus SHA-256 digest for every supported version. Each fixture contains UUID identifiers, nested JSON records, multiple `MemoryType` enum values, ordered embeddings, and timezone-aware timestamps with non-UTC offsets.

Migrations are forward-only. Experia rejects implicit schema downgrades, and a fixture remains in the repository for as long as its version is in the declared support window. Back up a database before upgrading when operational rollback may be required.
