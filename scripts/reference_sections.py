"""Validate and render Experia's generated lifecycle and contract reference.

The lifecycle, typed-failure, outbound-summary, stability, schema-support, and
deprecation-window sections in ``API_REFERENCE.md`` are generated from the
repository's validated machine-readable sources rather than hand-written:

* ``lifecycle-contract.json`` — lifecycle operation and typed-failure semantics,
  each cross-checked against the public methods and exception exports recorded
  in ``api-snapshot.json``.
* ``api-snapshot.json`` — current major version and deprecation metadata.
* ``outbound-data.json`` — per-feature network, credential, and emission summary.
* ``tests/fixtures/sqlite/schema-support.json`` — the supported schema window,
  cross-checked against ``experia.memory.migrations``.
* ``scripts.api_compatibility.MINOR_DEPRECATION_RELEASES`` — the deprecation
  window length.

A drift test fails whenever the documentation no longer matches these sources.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "lifecycle-contract.json"
DEFAULT_API_SNAPSHOT = ROOT / "api-snapshot.json"
DEFAULT_OUTBOUND_CONTRACT = ROOT / "outbound-data.json"
DEFAULT_SCHEMA_SUPPORT = ROOT / "tests" / "fixtures" / "sqlite" / "schema-support.json"
DEFAULT_API_REFERENCE = ROOT / "API_REFERENCE.md"

REFERENCE_BLOCK_START = "<!-- BEGIN GENERATED LIFECYCLE AND CONTRACT REFERENCE -->"
REFERENCE_BLOCK_END = "<!-- END GENERATED LIFECYCLE AND CONTRACT REFERENCE -->"

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PLACEHOLDER_VALUES = {
    "changeme",
    "fixme",
    "n/a",
    "placeholder",
    "replace me",
    "tbd",
    "todo",
    "unknown",
}


class ReferenceContractError(ValueError):
    """Raised when the reference contract is incomplete or inconsistent."""


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReferenceContractError(f"Unable to load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReferenceContractError(f"{path}: expected a JSON object")
    return value


def canonical_json(contract: Mapping[str, Any]) -> str:
    """Serialize a contract to deterministic UTF-8 JSON text."""

    return (
        json.dumps(
            contract,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _method_paths(snapshot: Mapping[str, Any]) -> set[str]:
    """Every documented ``owner.method`` path across import aliases and targets."""

    methods: set[str] = set()
    for export in _sequence(snapshot.get("exports"), "api-snapshot.exports"):
        if not isinstance(export, Mapping):
            continue
        owners = {str(export.get("path")), str(export.get("target"))}
        for member in export.get("members") or ():
            if isinstance(member, Mapping) and member.get("name"):
                for owner in owners:
                    methods.add(f"{owner}.{member['name']}")
    return methods


def _exception_paths(snapshot: Mapping[str, Any]) -> set[str]:
    """Every import alias and target for exported exception types."""

    exceptions: set[str] = set()
    for export in _sequence(snapshot.get("exports"), "api-snapshot.exports"):
        if isinstance(export, Mapping) and export.get("kind") == "exception":
            exceptions.add(str(export.get("path")))
            exceptions.add(str(export.get("target")))
    return exceptions


def validate_contract(
    contract: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> None:
    """Validate the lifecycle/failure contract against the API snapshot."""

    _exact_keys(contract, {"schema_version", "operations", "failures"}, "contract")
    if contract["schema_version"] != SCHEMA_VERSION:
        _fail(
            "contract.schema_version",
            f"expected {SCHEMA_VERSION}, got {contract['schema_version']!r}",
        )

    methods = _method_paths(snapshot)
    exceptions = _exception_paths(snapshot)

    operations = _sequence(contract["operations"], "contract.operations")
    if not operations:
        _fail("contract.operations", "must document at least one lifecycle operation")
    seen_ids: set[str] = set()
    seen_methods: set[str] = set()
    for index, value in enumerate(operations):
        path = f"contract.operations[{index}]"
        operation = _mapping(value, path)
        _exact_keys(
            operation,
            {
                "id",
                "title",
                "method",
                "call_order",
                "postconditions",
                "pending_jobs",
                "idempotence",
            },
            path,
        )
        identifier = _identifier(operation["id"], f"{path}.id")
        if identifier in seen_ids:
            _fail(f"{path}.id", f"duplicate operation id {identifier!r}")
        seen_ids.add(identifier)
        _text(operation["title"], f"{path}.title")
        method = _text(operation["method"], f"{path}.method")
        if method not in methods:
            _fail(
                f"{path}.method",
                f"{method!r} is not a public method in the API snapshot",
            )
        if method in seen_methods:
            _fail(f"{path}.method", f"duplicate lifecycle method {method!r}")
        seen_methods.add(method)
        for key in ("call_order", "postconditions", "pending_jobs", "idempotence"):
            _text(operation[key], f"{path}.{key}")

    _validate_required_operations(seen_methods)

    failures = _sequence(contract["failures"], "contract.failures")
    if not failures:
        _fail("contract.failures", "must document at least one typed failure")
    seen_failure_ids: set[str] = set()
    for index, value in enumerate(failures):
        path = f"contract.failures[{index}]"
        failure = _mapping(value, path)
        _exact_keys(
            failure,
            {"id", "trigger", "typed_error", "resulting_state", "retry"},
            path,
        )
        identifier = _identifier(failure["id"], f"{path}.id")
        if identifier in seen_failure_ids:
            _fail(f"{path}.id", f"duplicate failure id {identifier!r}")
        seen_failure_ids.add(identifier)
        _text(failure["trigger"], f"{path}.trigger")
        typed_error = _text(failure["typed_error"], f"{path}.typed_error")
        if typed_error not in exceptions:
            _fail(
                f"{path}.typed_error",
                f"{typed_error!r} is not an exported Experia exception",
            )
        _text(failure["resulting_state"], f"{path}.resulting_state")
        _text(failure["retry"], f"{path}.retry")


_REQUIRED_LIFECYCLE_METHODS = {
    "experia.memory.store.SQLiteStore.initialize",
    "experia.core.learner.Learner.flush",
    "experia.core.learner.Learner.shutdown",
    "experia.memory.store.SQLiteStore.close",
}


def _validate_required_operations(documented_methods: set[str]) -> None:
    missing = sorted(_REQUIRED_LIFECYCLE_METHODS - documented_methods)
    if missing:
        _fail(
            "contract.operations",
            "missing required lifecycle operations: " + ", ".join(missing),
        )


def validate_schema_support(support: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the schema support window against the migration registry."""

    from experia.memory.migrations import (
        CURRENT_SCHEMA_VERSION,
        MIN_SUPPORTED_SCHEMA_VERSION,
        SUPPORTED_SCHEMA_VERSIONS,
    )

    window = _mapping(support.get("support_window"), "schema-support.support_window")
    supported = list(window.get("supported_schema_versions", []))
    if supported != list(SUPPORTED_SCHEMA_VERSIONS):
        _fail(
            "schema-support.support_window.supported_schema_versions",
            f"{supported!r} does not match migrations {list(SUPPORTED_SCHEMA_VERSIONS)!r}",
        )
    if window.get("current_schema_version") != CURRENT_SCHEMA_VERSION:
        _fail(
            "schema-support.support_window.current_schema_version",
            f"expected {CURRENT_SCHEMA_VERSION}",
        )
    if window.get("minimum_schema_version") != MIN_SUPPORTED_SCHEMA_VERSION:
        _fail(
            "schema-support.support_window.minimum_schema_version",
            f"expected {MIN_SUPPORTED_SCHEMA_VERSION}",
        )
    if window.get("maximum_schema_version") != CURRENT_SCHEMA_VERSION:
        _fail(
            "schema-support.support_window.maximum_schema_version",
            f"expected {CURRENT_SCHEMA_VERSION}",
        )
    return window


def _render_lifecycle_operations(contract: Mapping[str, Any]) -> list[str]:
    lines = [
        "### Lifecycle operations",
        "",
        "Call order, postconditions, pending Background_Job state, and idempotence",
        "for initialization, `flush()`, the shutdown operation, and store close.",
        "",
        "| Operation | Call order | Postconditions | Pending background jobs | Idempotence |",
        "|---|---|---|---|---|",
    ]
    for operation in contract["operations"]:
        lines.append(
            "| "
            f"{_markdown_cell(operation['title'])} (`{operation['method']}`) | "
            f"{_markdown_cell(operation['call_order'])} | "
            f"{_markdown_cell(operation['postconditions'])} | "
            f"{_markdown_cell(operation['pending_jobs'])} | "
            f"{_markdown_cell(operation['idempotence'])} |"
        )
    return lines


def _render_failure_contract(contract: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "### Typed failure contract",
        "",
        "Every documented failure names its trigger, the typed error raised, the",
        "resulting state, and the retry behavior. Typed errors are the exception",
        "classes recorded in [`api-snapshot.json`](api-snapshot.json).",
        "",
        "| Trigger | Typed error | Resulting state | Retry behavior |",
        "|---|---|---|---|",
    ]
    for failure in contract["failures"]:
        lines.append(
            "| "
            f"{_markdown_cell(failure['trigger'])} | "
            f"`{failure['typed_error']}` | "
            f"{_markdown_cell(failure['resulting_state'])} | "
            f"{_markdown_cell(failure['retry'])} |"
        )
    return lines


def _render_outbound_summary(outbound: Mapping[str, Any]) -> list[str]:
    default = outbound.get("default_without_sanitizer")
    features = _sequence(outbound.get("features"), "outbound-data.features")
    lines = [
        "",
        "### Network and credential summary",
        "",
        "Per-feature network, credential, and metadata-emission behavior generated",
        "from [`outbound-data.json`](outbound-data.json). Without a configured",
        f"sanitizer every transmitted and emitted field is **{default}**. The",
        "field-level tables above list each transmitted field and its"
        " sanitized/pass-through classification.",
        "",
        "| Feature | Sink | Network requirement | Credential category | Metadata emission |",
        "|---|---|---|---|---|",
    ]
    for feature in features:
        feature = _mapping(feature, "outbound-data.features[]")
        lines.append(
            "| "
            f"{_markdown_cell(feature['title'])} | "
            f"`{feature['sink']}` | "
            f"`{feature['network_requirement']}` | "
            f"`{feature['credential_category']}` | "
            f"`{feature['metadata_emission']}` |"
        )
    return lines


def _render_stability(snapshot: Mapping[str, Any]) -> list[str]:
    major = snapshot.get("major_version")
    version = snapshot.get("package_version")
    return [
        "",
        "### Public API stability",
        "",
        f"The current major version is **{major}** (package version `{version}`).",
        "Within this major version, every supported import path and compatible"
        " signature recorded in [`api-snapshot.json`](api-snapshot.json) is"
        " preserved. Removals or narrowed signatures ship only under a greater"
        " major version accompanied by a migration guide.",
    ]


def _render_schema_support(window: Mapping[str, Any]) -> list[str]:
    current = int(window["current_schema_version"])
    minimum = int(window["minimum_schema_version"])
    versions = [int(value) for value in window["supported_schema_versions"]]
    legacy_semantics = str(window.get("version_zero_semantics", "")).strip()
    lines = [
        "",
        "### SQLite schema support window",
        "",
        f"Schema version {current} supports forward, upgrade-only migration from"
        f" every version in the inclusive window {minimum} through {current}. The"
        " machine-readable source of truth is"
        " [`schema-support.json`](tests/fixtures/sqlite/schema-support.json), which"
        " must agree with `experia.memory.migrations`.",
        "",
        "| Schema version | Status | Forward migration |",
        "|---:|---|---|",
    ]
    for version in versions:
        if version == current:
            status = "Current"
            migration = "No migration required"
        else:
            targets = ", ".join(
                str(target) for target in range(version + 1, current + 1)
            )
            migration = f"Migrates through version(s) {targets}"
            if version == minimum and legacy_semantics:
                status = "Supported legacy"
            else:
                status = "Supported"
        lines.append(f"| {version} | {status} | {migration} |")
    return lines


def _render_deprecation_window(
    snapshot: Mapping[str, Any],
    minimum_minor_releases: int,
) -> list[str]:
    deprecated = _collect_deprecations(snapshot)
    lines = [
        "",
        "### Deprecation window",
        "",
        "A deprecated public API stays callable for at least"
        f" {minimum_minor_releases} consecutive minor releases while emitting a"
        " deprecation warning that names its replacement. Removal before that"
        " window elapses is permitted only under a greater major version, and the"
        " migration guide identifies the removal and its replacement.",
    ]
    if not deprecated:
        lines.extend(
            [
                "",
                "No public API is deprecated in the current major version.",
            ]
        )
        return lines
    lines.extend(
        [
            "",
            "| Deprecated API | Since | Replacement |",
            "|---|---|---|",
        ]
    )
    for entry in deprecated:
        lines.append(
            "| "
            f"`{entry['path']}` | "
            f"`{_markdown_cell(entry['since'])}` | "
            f"`{_markdown_cell(entry['replacement'])}` |"
        )
    return lines


def _collect_deprecations(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for export in _sequence(snapshot.get("exports"), "api-snapshot.exports"):
        if not isinstance(export, Mapping):
            continue
        path = str(export.get("path"))
        _append_deprecation(entries, path, export.get("deprecation"))
        for member in export.get("members") or ():
            if isinstance(member, Mapping):
                _append_deprecation(
                    entries,
                    f"{path}.{member.get('name')}",
                    member.get("deprecation"),
                )
    entries.sort(key=lambda entry: entry["path"])
    return entries


def _append_deprecation(
    entries: list[dict[str, Any]],
    path: str,
    deprecation: object,
) -> None:
    if isinstance(deprecation, Mapping) and deprecation.get("is_deprecated"):
        entries.append(
            {
                "path": path,
                "since": deprecation.get("since"),
                "replacement": deprecation.get("replacement"),
            }
        )


def render_reference_sections(
    *,
    contract: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    outbound: Mapping[str, Any],
    schema_support: Mapping[str, Any],
    minimum_minor_releases: int,
) -> str:
    """Render every validated reference section as one generated block."""

    validate_contract(contract, snapshot)
    window = validate_schema_support(schema_support)

    lines = [
        REFERENCE_BLOCK_START,
        "## Lifecycle, failure, and contract reference",
        "",
        "Generated from the repository's validated sources"
        " ([`lifecycle-contract.json`](lifecycle-contract.json),"
        " [`api-snapshot.json`](api-snapshot.json),"
        " [`outbound-data.json`](outbound-data.json), and"
        " [`schema-support.json`](tests/fixtures/sqlite/schema-support.json)) so"
        " these tables stay consistent with the installed behavior.",
        "",
    ]
    lines.extend(_render_lifecycle_operations(contract))
    lines.extend(_render_failure_contract(contract))
    lines.extend(_render_outbound_summary(outbound))
    lines.extend(_render_stability(snapshot))
    lines.extend(_render_schema_support(window))
    lines.extend(_render_deprecation_window(snapshot, minimum_minor_releases))
    lines.extend(["", REFERENCE_BLOCK_END])
    return "\n".join(lines)


def _build_reference_block(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    api_snapshot_path: Path = DEFAULT_API_SNAPSHOT,
    outbound_path: Path = DEFAULT_OUTBOUND_CONTRACT,
    schema_support_path: Path = DEFAULT_SCHEMA_SUPPORT,
) -> str:
    from scripts.api_compatibility import MINOR_DEPRECATION_RELEASES

    return render_reference_sections(
        contract=load_json(contract_path),
        snapshot=load_json(api_snapshot_path),
        outbound=load_json(outbound_path),
        schema_support=load_json(schema_support_path),
        minimum_minor_releases=MINOR_DEPRECATION_RELEASES,
    )


def assert_api_reference_synced(
    path: Path = DEFAULT_API_REFERENCE,
    **sources: Path,
) -> None:
    """Raise when the API reference does not contain the rendered block."""

    reference = path.read_text(encoding="utf-8")
    expected = _build_reference_block(**sources)
    actual = _marked_block(reference, path)
    if actual != expected:
        raise ReferenceContractError(
            f"{path} lifecycle/contract section is stale; run this script with --write-docs"
        )


def sync_api_reference(
    path: Path = DEFAULT_API_REFERENCE,
    **sources: Path,
) -> Path:
    """Replace the generated lifecycle/contract block in the API reference."""

    reference = path.read_text(encoding="utf-8")
    _marked_block(reference, path)
    prefix, remainder = reference.split(REFERENCE_BLOCK_START, 1)
    _, suffix = remainder.split(REFERENCE_BLOCK_END, 1)
    path.write_text(
        prefix + _build_reference_block(**sources) + suffix, encoding="utf-8"
    )
    return path


def _marked_block(text: str, path: Path) -> str:
    if text.count(REFERENCE_BLOCK_START) != 1 or text.count(REFERENCE_BLOCK_END) != 1:
        raise ReferenceContractError(
            f"{path} must contain exactly one generated lifecycle/contract block"
        )
    middle = text.split(REFERENCE_BLOCK_START, 1)[1].split(REFERENCE_BLOCK_END, 1)[0]
    return REFERENCE_BLOCK_START + middle + REFERENCE_BLOCK_END


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    keys = set(value)
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected)
    if missing:
        _fail(path, f"missing required keys: {', '.join(missing)}")
    if unexpected:
        _fail(path, f"unexpected keys: {', '.join(unexpected)}")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected an object")
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(path, "expected an array")
    return value


def _identifier(value: object, path: str) -> str:
    text = _text(value, path)
    if _IDENTIFIER_PATTERN.fullmatch(text) is None:
        _fail(path, "must be a lowercase snake_case identifier")
    return text


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    text = value.strip()
    normalized = re.sub(r"[\s_-]+", " ", text.casefold()).strip(" .:<>[]{}")
    if normalized in _PLACEHOLDER_VALUES or re.fullmatch(r"<[^>]+>", text):
        _fail(path, f"placeholder value is not allowed: {value!r}")
    return text


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _fail(path: str, detail: str) -> None:
    raise ReferenceContractError(f"{path}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--api-snapshot", type=Path, default=DEFAULT_API_SNAPSHOT)
    parser.add_argument("--outbound", type=Path, default=DEFAULT_OUTBOUND_CONTRACT)
    parser.add_argument("--schema-support", type=Path, default=DEFAULT_SCHEMA_SUPPORT)
    parser.add_argument("--api-reference", type=Path, default=DEFAULT_API_REFERENCE)
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="synchronize the generated API reference section",
    )
    arguments = parser.parse_args()

    contract = load_json(arguments.contract)
    snapshot = load_json(arguments.api_snapshot)
    validate_contract(contract, snapshot)
    if arguments.contract.read_text(encoding="utf-8") != canonical_json(contract):
        raise ReferenceContractError(
            f"{arguments.contract} is not canonical JSON "
            "(sorted keys and stable formatting required)"
        )

    sources = {
        "contract_path": arguments.contract,
        "api_snapshot_path": arguments.api_snapshot,
        "outbound_path": arguments.outbound,
        "schema_support_path": arguments.schema_support,
    }
    if arguments.write_docs:
        sync_api_reference(arguments.api_reference, **sources)
    else:
        assert_api_reference_synced(arguments.api_reference, **sources)
    print(
        f"Validated {len(contract['operations'])} lifecycle operations and "
        f"{len(contract['failures'])} typed failures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_API_REFERENCE",
    "DEFAULT_CONTRACT",
    "REFERENCE_BLOCK_END",
    "REFERENCE_BLOCK_START",
    "ReferenceContractError",
    "assert_api_reference_synced",
    "canonical_json",
    "load_json",
    "render_reference_sections",
    "sync_api_reference",
    "validate_contract",
    "validate_schema_support",
]
