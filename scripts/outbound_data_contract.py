"""Validate and render Experia's canonical outbound-data contract."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "outbound-data.json"
DEFAULT_API_REFERENCE = ROOT / "API_REFERENCE.md"
DEFAULT_PACKAGE_ROOT = ROOT / "experia"
OUTBOUND_BLOCK_START = "<!-- BEGIN GENERATED OUTBOUND DATA CONTRACT -->"
OUTBOUND_BLOCK_END = "<!-- END GENERATED OUTBOUND DATA CONTRACT -->"

_NETWORK_REQUIREMENTS = {
    "required",
    "not-required",
    "provider-dependent",
    "implementation-dependent",
}
_CREDENTIAL_CATEGORIES = {
    "none",
    "provider-dependent-api-credential",
    "implementation-dependent",
}
_CLASSIFICATIONS = {"sanitized", "pass-through"}
_METADATA_EMISSIONS = {"logging", "not-emitted"}
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_DOTTED_PATH_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*(?:\[\])?(?:\.[a-z][a-z0-9_]*(?:\[\])?)*$"
)
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


class OutboundContractError(ValueError):
    """Raised when the outbound-data contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class ProtectedSink:
    """One source-discovered ``protect_sink`` boundary and its literal fields."""

    path: str
    request_fields: tuple[str, ...]
    metadata_fields: tuple[str, ...]


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Load a JSON outbound-data contract from ``path``."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OutboundContractError(f"Unable to load {path}: {error}") from error
    if not isinstance(value, dict):
        raise OutboundContractError("contract: expected a JSON object")
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


def discover_protected_sinks(
    package_root: Path = DEFAULT_PACKAGE_ROOT,
) -> tuple[ProtectedSink, ...]:
    """Discover every production ``protect_sink`` call and its literal fields."""

    discovered: dict[str, tuple[set[str], set[str]]] = {}
    package_name = package_root.name

    for source_path in sorted(package_root.rglob("*.py")):
        module_parts = (
            package_name,
            *source_path.relative_to(package_root).with_suffix("").parts,
        )
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        module = ".".join(module_parts)
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"), filename=str(source_path)
        )
        visitor = _ProtectionCallVisitor(module, source_path, discovered)
        visitor.visit(tree)

    return tuple(
        ProtectedSink(
            path=path,
            request_fields=tuple(sorted(request_fields)),
            metadata_fields=tuple(sorted(metadata_fields)),
        )
        for path, (request_fields, metadata_fields) in sorted(discovered.items())
    )


class _ProtectionCallVisitor(ast.NodeVisitor):
    def __init__(
        self,
        module: str,
        source_path: Path,
        discovered: dict[str, tuple[set[str], set[str]]],
    ) -> None:
        self._module = module
        self._source_path = source_path
        self._discovered = discovered
        self._scope: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._visit_scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_scope(node)

    def _visit_scope(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr == "protect_sink":
            if not self._scope:
                raise OutboundContractError(
                    f"{self._source_path}:{node.lineno}: protect_sink must be inside a named sink"
                )
            if len(node.args) < 2:
                raise OutboundContractError(
                    f"{self._source_path}:{node.lineno}: protect_sink request and metadata must be positional literal mappings"
                )
            sink_path = f"{self._module}.{'.'.join(self._scope)}"
            request_fields = _literal_mapping_keys(
                node.args[0],
                source_path=self._source_path,
                line=node.lineno,
                label="request",
            )
            metadata_fields = _literal_mapping_keys(
                node.args[1],
                source_path=self._source_path,
                line=node.lineno,
                label="metadata",
            )
            known_request, known_metadata = self._discovered.setdefault(
                sink_path, (set(), set())
            )
            known_request.update(request_fields)
            known_metadata.update(metadata_fields)
        self.generic_visit(node)


def _literal_mapping_keys(
    node: ast.AST,
    *,
    source_path: Path,
    line: int,
    label: str,
) -> set[str]:
    if not isinstance(node, ast.Dict):
        raise OutboundContractError(
            f"{source_path}:{line}: protect_sink {label} must be a literal mapping"
        )
    keys: set[str] = set()
    for key in node.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise OutboundContractError(
                f"{source_path}:{line}: protect_sink {label} keys must be literal strings"
            )
        keys.add(key.value)
    return keys


def validate_contract(
    contract: Mapping[str, Any],
    *,
    package_root: Path = DEFAULT_PACKAGE_ROOT,
) -> None:
    """Validate schema, semantics, pass-through policy, and source coverage."""

    _exact_keys(
        contract,
        {"schema_version", "default_without_sanitizer", "features"},
        "contract",
    )
    if contract["schema_version"] != SCHEMA_VERSION:
        _fail(
            "contract.schema_version",
            f"expected {SCHEMA_VERSION}, got {contract['schema_version']!r}",
        )
    default_classification = contract["default_without_sanitizer"]
    if default_classification != "pass-through":
        _fail(
            "contract.default_without_sanitizer",
            "must be 'pass-through' to preserve the no-sanitizer contract",
        )

    features = _sequence(contract["features"], "contract.features")
    if not features:
        _fail("contract.features", "must contain at least one external feature")

    discovered = {sink.path: sink for sink in discover_protected_sinks(package_root)}
    documented: dict[str, str] = {}
    feature_ids: list[str] = []
    for index, value in enumerate(features):
        path = f"contract.features[{index}]"
        feature = _mapping(value, path)
        _validate_feature(
            feature,
            path=path,
            default_classification=default_classification,
            discovered=discovered,
        )
        feature_id = str(feature["id"])
        sink = str(feature["sink"])
        if feature_id in feature_ids:
            _fail(f"{path}.id", f"duplicate feature id {feature_id!r}")
        if sink in documented:
            _fail(
                f"{path}.sink",
                f"duplicate sink also documented by {documented[sink]!r}",
            )
        feature_ids.append(feature_id)
        documented[sink] = feature_id

    if feature_ids != sorted(feature_ids):
        _fail("contract.features", "features must be sorted by id")

    missing = sorted(set(discovered) - set(documented))
    extra = sorted(set(documented) - set(discovered))
    if missing:
        _fail(
            "contract.features",
            f"missing current protected sink entries: {', '.join(missing)}",
        )
    if extra:
        _fail(
            "contract.features",
            f"declares sinks with no current protection boundary: {', '.join(extra)}",
        )


def _validate_feature(
    feature: Mapping[str, Any],
    *,
    path: str,
    default_classification: object,
    discovered: Mapping[str, ProtectedSink],
) -> None:
    _exact_keys(
        feature,
        {
            "credential_category",
            "credential_details",
            "id",
            "metadata_emission",
            "metadata_fields",
            "network_details",
            "network_requirement",
            "request_fields",
            "service",
            "sink",
            "title",
        },
        path,
    )
    feature_id = _text(feature["id"], f"{path}.id")
    if _IDENTIFIER_PATTERN.fullmatch(feature_id) is None:
        _fail(f"{path}.id", "must be a lowercase snake_case identifier")
    _text(feature["title"], f"{path}.title")
    sink = _text(feature["sink"], f"{path}.sink")
    _text(feature["service"], f"{path}.service")
    _choice(
        feature["network_requirement"],
        _NETWORK_REQUIREMENTS,
        f"{path}.network_requirement",
    )
    _text(feature["network_details"], f"{path}.network_details")
    _choice(
        feature["credential_category"],
        _CREDENTIAL_CATEGORIES,
        f"{path}.credential_category",
    )
    _text(feature["credential_details"], f"{path}.credential_details")
    _choice(
        feature["metadata_emission"],
        _METADATA_EMISSIONS,
        f"{path}.metadata_emission",
    )

    request_fields = _validate_fields(
        feature["request_fields"],
        path=f"{path}.request_fields",
        default_classification=default_classification,
        metadata=False,
    )
    metadata_fields = _validate_fields(
        feature["metadata_fields"],
        path=f"{path}.metadata_fields",
        default_classification=default_classification,
        metadata=True,
    )

    source_sink = discovered.get(sink)
    if source_sink is None:
        return
    protected_request_roots = {
        _root_path(field["protection_path"])
        for field in request_fields
        if field["protection_path"] is not None
    }
    if protected_request_roots != set(source_sink.request_fields):
        _fail(
            f"{path}.request_fields",
            "protected request roots do not match source: "
            f"contract={sorted(protected_request_roots)!r}, "
            f"source={list(source_sink.request_fields)!r}",
        )
    documented_metadata = {str(field["path"]) for field in metadata_fields}
    if documented_metadata != set(source_sink.metadata_fields):
        _fail(
            f"{path}.metadata_fields",
            "metadata fields do not match source: "
            f"contract={sorted(documented_metadata)!r}, "
            f"source={list(source_sink.metadata_fields)!r}",
        )


def _validate_fields(
    value: object,
    *,
    path: str,
    default_classification: object,
    metadata: bool,
) -> list[Mapping[str, Any]]:
    values = _sequence(value, path)
    if not values:
        _fail(path, "must contain at least one field")
    fields: list[Mapping[str, Any]] = []
    field_paths: list[str] = []
    for index, item in enumerate(values):
        field_path = f"{path}[{index}]"
        field = _mapping(item, field_path)
        _exact_keys(
            field,
            {
                "path",
                "protection_path",
                "source",
                "with_sanitizer",
                "without_sanitizer",
            },
            field_path,
        )
        transmitted_path = _text(field["path"], f"{field_path}.path")
        if _DOTTED_PATH_PATTERN.fullmatch(transmitted_path) is None:
            _fail(f"{field_path}.path", "must be a dotted logical field path")
        protection_path = field["protection_path"]
        if protection_path is not None:
            protection_path = _text(protection_path, f"{field_path}.protection_path")
            if _DOTTED_PATH_PATTERN.fullmatch(protection_path) is None:
                _fail(
                    f"{field_path}.protection_path",
                    "must be null or a dotted protection payload path",
                )
        _text(field["source"], f"{field_path}.source")
        with_sanitizer = _choice(
            field["with_sanitizer"],
            _CLASSIFICATIONS,
            f"{field_path}.with_sanitizer",
        )
        without_sanitizer = _choice(
            field["without_sanitizer"],
            _CLASSIFICATIONS,
            f"{field_path}.without_sanitizer",
        )
        if without_sanitizer != default_classification:
            _fail(
                f"{field_path}.without_sanitizer",
                f"must preserve the contract default {default_classification!r}",
            )
        if with_sanitizer == "sanitized" and protection_path is None:
            _fail(
                f"{field_path}.protection_path",
                "is required when the field is classified as sanitized",
            )
        if with_sanitizer == "pass-through" and protection_path is not None:
            _fail(
                f"{field_path}.with_sanitizer",
                "protected payload fields must be classified as sanitized",
            )
        if metadata:
            if with_sanitizer != "sanitized":
                _fail(
                    f"{field_path}.with_sanitizer",
                    "associated metadata values cross the protection boundary",
                )
            if protection_path != transmitted_path:
                _fail(
                    f"{field_path}.protection_path",
                    "metadata protection_path must equal its emitted path",
                )
        if transmitted_path in field_paths:
            _fail(f"{field_path}.path", f"duplicate field {transmitted_path!r}")
        field_paths.append(transmitted_path)
        fields.append(field)

    if field_paths != sorted(field_paths):
        _fail(path, "fields must be sorted by path")
    return fields


def _root_path(value: object) -> str:
    return str(value).split(".", 1)[0].removesuffix("[]")


def render_outbound_contract(contract: Mapping[str, Any]) -> str:
    """Render the validated contract as the canonical API reference section."""

    validate_contract(contract)
    lines = [
        OUTBOUND_BLOCK_START,
        "## Canonical outbound-data contract",
        "",
        "Generated from [`outbound-data.json`](outbound-data.json) and checked",
        "against every current production `protect_sink` call site. Request fields",
        "are logical fields within provider payloads; fixed framing and control values",
        "are listed alongside caller-derived values.",
        "",
        "**No sanitizer configured:** every request and metadata field below is",
        "value-equivalent **pass-through**. Experia still makes a defensive copy before",
        "the sink, but it does not redact or transform values.",
    ]
    for feature in contract["features"]:
        lines.extend(
            [
                "",
                f"### {_markdown_cell(feature['title'])}",
                "",
                f"- Sink: `{feature['sink']}`",
                f"- Service/implementation: {_markdown_cell(feature['service'])}",
                f"- Network: `{feature['network_requirement']}` — "
                f"{_markdown_cell(feature['network_details'])}",
                f"- Credential category: `{feature['credential_category']}` — "
                f"{_markdown_cell(feature['credential_details'])}",
                f"- Associated metadata emission: `{feature['metadata_emission']}`",
                "",
                "| Request field | Source | Sanitizer configured | No sanitizer |",
                "|---|---|---|---|",
            ]
        )
        for field in feature["request_fields"]:
            lines.append(
                f"| `{field['path']}` | {_markdown_cell(field['source'])} | "
                f"`{field['with_sanitizer']}` | `{field['without_sanitizer']}` |"
            )
        lines.extend(
            [
                "",
                "| Associated metadata field | Source | Sanitizer configured | No sanitizer |",
                "|---|---|---|---|",
            ]
        )
        for field in feature["metadata_fields"]:
            lines.append(
                f"| `{field['path']}` | {_markdown_cell(field['source'])} | "
                f"`{field['with_sanitizer']}` | `{field['without_sanitizer']}` |"
            )
    lines.extend(["", OUTBOUND_BLOCK_END])
    return "\n".join(lines)


def assert_api_reference_synced(
    contract: Mapping[str, Any],
    path: Path = DEFAULT_API_REFERENCE,
) -> None:
    """Raise when the API reference does not contain the rendered contract."""

    reference = path.read_text(encoding="utf-8")
    expected = render_outbound_contract(contract)
    actual = _marked_block(reference, path)
    if actual != expected:
        raise OutboundContractError(
            f"{path} outbound-data section is stale; run this script with --write-docs"
        )


def sync_api_reference(
    contract: Mapping[str, Any],
    path: Path = DEFAULT_API_REFERENCE,
) -> Path:
    """Replace the generated outbound-data block in the API reference."""

    reference = path.read_text(encoding="utf-8")
    _marked_block(reference, path)
    prefix, remainder = reference.split(OUTBOUND_BLOCK_START, 1)
    _, suffix = remainder.split(OUTBOUND_BLOCK_END, 1)
    path.write_text(
        prefix + render_outbound_contract(contract) + suffix, encoding="utf-8"
    )
    return path


def _marked_block(text: str, path: Path) -> str:
    if text.count(OUTBOUND_BLOCK_START) != 1 or text.count(OUTBOUND_BLOCK_END) != 1:
        raise OutboundContractError(
            f"{path} must contain exactly one generated outbound-data contract block"
        )
    middle = text.split(OUTBOUND_BLOCK_START, 1)[1].split(OUTBOUND_BLOCK_END, 1)[0]
    return OUTBOUND_BLOCK_START + middle + OUTBOUND_BLOCK_END


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


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    text = value.strip()
    normalized = re.sub(r"[\s_-]+", " ", text.casefold()).strip(" .:<>[]{}")
    if normalized in _PLACEHOLDER_VALUES or re.fullmatch(r"<[^>]+>", text):
        _fail(path, f"placeholder value is not allowed: {value!r}")
    return text


def _choice(value: object, choices: set[str], path: str) -> str:
    if not isinstance(value, str) or value not in choices:
        _fail(path, f"must be one of {sorted(choices)!r}, got {value!r}")
    return value


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _fail(path: str, detail: str) -> None:
    raise OutboundContractError(f"{path}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--api-reference", type=Path, default=DEFAULT_API_REFERENCE)
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="synchronize the generated API reference section",
    )
    arguments = parser.parse_args()

    contract = load_contract(arguments.contract)
    validate_contract(contract)
    if arguments.contract.read_text(encoding="utf-8") != canonical_json(contract):
        raise OutboundContractError(
            f"{arguments.contract} is not canonical JSON (sorted keys and stable formatting required)"
        )
    if arguments.write_docs:
        sync_api_reference(contract, arguments.api_reference)
    else:
        assert_api_reference_synced(contract, arguments.api_reference)
    print(f"Validated {len(contract['features'])} outbound-data features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONTRACT",
    "OUTBOUND_BLOCK_END",
    "OUTBOUND_BLOCK_START",
    "OutboundContractError",
    "ProtectedSink",
    "assert_api_reference_synced",
    "canonical_json",
    "discover_protected_sinks",
    "load_contract",
    "render_outbound_contract",
    "sync_api_reference",
    "validate_contract",
]
