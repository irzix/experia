"""Validate Experia's roadmap ownership/readiness manifest.

Requirement 9.5 requires every planned adapter or integration to carry exactly
one ownership selector -- a public ``owner``, a public ``team``, or an explicit
``unassigned`` value -- together with exactly one readiness status. Requirement
4.7 requires importable planned features to raise an explicit
``UnavailableFeatureError`` rather than appear operational, and Requirements
10.2/10.3 require the documented Project Status to describe planned features as
unavailable. This module parses the committed manifest, enforces the ownership
and readiness schema, connects each importable entry to its live
``UnavailableFeatureError`` behavior, and generates the README Project Status
planned section so the roadmap, runtime, and documentation cannot drift apart.

Owner identities are never invented here: entries default to the explicit
``unassigned`` value until a maintainer records a confirmed public owner or team.
"""

from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experia.core.exceptions import UnavailableFeatureError

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "roadmap-ownership.yml"
DEFAULT_README = ROOT / "README.md"

ROADMAP_BLOCK_START = "<!-- BEGIN GENERATED ROADMAP STATUS -->"
ROADMAP_BLOCK_END = "<!-- END GENERATED ROADMAP STATUS -->"

_READINESS_STATUSES = {"planned", "in_progress", "blocked", "unavailable"}
_KINDS = {"adapter", "integration", "runtime"}
_OWNERSHIP_KEYS = ("owner", "team", "unassigned")
_ITEM_KEYS = {"id", "title", "kind", "readiness", "entrypoint", *_OWNERSHIP_KEYS}
_REQUIRED_ITEM_KEYS = {"id", "title", "kind", "readiness"}
_ENTRYPOINT_KEYS = {"module", "attribute", "feature"}

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MODULE_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*$")
_ATTRIBUTE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
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


class RoadmapManifestError(ValueError):
    """Raised when the roadmap manifest is incomplete or inconsistent."""


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load a YAML roadmap manifest from ``path``."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RoadmapManifestError(f"Unable to load {path}: {error}") from error
    if not isinstance(value, dict):
        raise RoadmapManifestError("manifest: expected a YAML mapping")
    return value


def canonical_yaml(manifest: Mapping[str, Any]) -> str:
    """Serialize a manifest to deterministic block-style YAML text."""

    return yaml.safe_dump(
        dict(manifest),
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
    )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate manifest schema, ownership, readiness, and ordering."""

    _exact_keys(manifest, {"schema_version", "items"}, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        _fail(
            "manifest.schema_version",
            f"expected {SCHEMA_VERSION}, got {manifest['schema_version']!r}",
        )

    items = _sequence(manifest["items"], "manifest.items")
    if not items:
        _fail("manifest.items", "must contain at least one planned item")

    ids: list[str] = []
    for index, value in enumerate(items):
        path = f"manifest.items[{index}]"
        item = _mapping(value, path)
        item_id = validate_entry(item, path=path)
        if item_id in ids:
            _fail(f"{path}.id", f"duplicate item id {item_id!r}")
        ids.append(item_id)

    if ids != sorted(ids):
        _fail("manifest.items", "items must be sorted by id")


def validate_entry(entry: Mapping[str, Any], *, path: str = "item") -> str:
    """Validate one roadmap entry and return its id.

    An entry is valid if and only if it carries exactly one ownership selector
    (``owner``, ``team``, or ``unassigned``) and exactly one readiness status.
    """

    if not isinstance(entry, Mapping):
        _fail(path, "expected an object")

    keys = set(entry)
    unexpected = sorted(keys - _ITEM_KEYS)
    if unexpected:
        _fail(path, f"unexpected keys: {', '.join(unexpected)}")
    missing = sorted(_REQUIRED_ITEM_KEYS - keys)
    if missing:
        _fail(path, f"missing required keys: {', '.join(missing)}")

    item_id = _text(entry["id"], f"{path}.id")
    if _ID_PATTERN.fullmatch(item_id) is None:
        _fail(f"{path}.id", "must be a lowercase snake_case identifier")
    _text(entry["title"], f"{path}.title")
    _choice(entry["kind"], _KINDS, f"{path}.kind")

    _validate_readiness(entry["readiness"], f"{path}.readiness")
    _validate_ownership(entry, path)

    entrypoint = entry.get("entrypoint")
    if entrypoint is not None:
        _validate_entrypoint(entrypoint, f"{path}.entrypoint")

    return item_id


def _validate_readiness(value: object, path: str) -> str:
    """Require exactly one readiness status drawn from the documented set."""

    if isinstance(value, (list, tuple, set)):
        _fail(path, "must be exactly one readiness status, not a collection")
    return _choice(value, _READINESS_STATUSES, path)


def _validate_ownership(entry: Mapping[str, Any], path: str) -> tuple[str, object]:
    """Require exactly one of owner/team/unassigned and validate its value."""

    present = [key for key in _OWNERSHIP_KEYS if key in entry]
    if len(present) != 1:
        _fail(
            path,
            "must declare exactly one ownership selector "
            f"(owner, team, or unassigned); found {present!r}",
        )
    key = present[0]
    value = entry[key]
    if key == "unassigned":
        if value is not True:
            _fail(
                f"{path}.unassigned",
                "must be the literal true when ownership is not established",
            )
    else:
        _text(value, f"{path}.{key}")
    return key, value


def _validate_entrypoint(value: object, path: str) -> Mapping[str, Any]:
    entrypoint = _mapping(value, path)
    _exact_keys(entrypoint, _ENTRYPOINT_KEYS, path)
    module = _text(entrypoint["module"], f"{path}.module")
    if _MODULE_PATTERN.fullmatch(module) is None:
        _fail(f"{path}.module", "must be a dotted importable module path")
    if not module.startswith("experia."):
        _fail(f"{path}.module", "must resolve inside the experia package")
    attribute = _text(entrypoint["attribute"], f"{path}.attribute")
    if _ATTRIBUTE_PATTERN.fullmatch(attribute) is None:
        _fail(f"{path}.attribute", "must be a valid attribute identifier")
    feature = _text(entrypoint["feature"], f"{path}.feature")
    if _ID_PATTERN.fullmatch(feature) is None:
        _fail(f"{path}.feature", "must be a lowercase snake_case identifier")
    return entrypoint


def assert_planned_entrypoints_unavailable(manifest: Mapping[str, Any]) -> int:
    """Connect each importable planned entry to its unavailable-feature check.

    Every entry that declares an ``entrypoint`` is imported and constructed; the
    manifest is only valid if construction raises ``UnavailableFeatureError`` with
    the declared feature name and a status equal to the entry's readiness.
    """

    validate_manifest(manifest)
    checked = 0
    for index, item in enumerate(manifest["items"]):
        entrypoint = item.get("entrypoint")
        if entrypoint is None:
            continue
        path = f"manifest.items[{index}].entrypoint"
        module_name = entrypoint["module"]
        attribute = entrypoint["attribute"]
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            _fail(f"{path}.module", f"cannot import {module_name!r}: {error}")
        placeholder = getattr(module, attribute, None)
        if placeholder is None:
            _fail(
                f"{path}.attribute",
                f"{module_name!r} has no attribute {attribute!r}",
            )
        try:
            placeholder()
        except UnavailableFeatureError as error:
            if error.feature != entrypoint["feature"]:
                _fail(
                    f"{path}.feature",
                    f"raised feature {error.feature!r} does not match "
                    f"declared {entrypoint['feature']!r}",
                )
            if error.status != item["readiness"]:
                _fail(
                    f"manifest.items[{index}].readiness",
                    f"entrypoint reports status {error.status!r} but manifest "
                    f"declares readiness {item['readiness']!r}",
                )
        except Exception as error:  # noqa: BLE001 - report any non-typed failure
            _fail(
                f"{path}",
                f"construction raised {type(error).__name__} instead of "
                "UnavailableFeatureError",
            )
        else:
            _fail(
                f"{path}",
                "construction did not raise UnavailableFeatureError; the planned "
                "feature appears operational",
            )
        checked += 1
    return checked


def render_roadmap_status(manifest: Mapping[str, Any]) -> str:
    """Render the validated manifest as the README Project Status planned block."""

    validate_manifest(manifest)
    lines = [ROADMAP_BLOCK_START]
    for item in manifest["items"]:
        lines.append(
            f"- **{item['title']}** ({item['kind']}) — "
            f"readiness `{item['readiness']}`, "
            f"ownership {_ownership_label(item)}; {_placeholder_clause(item)}"
        )
    lines.append(ROADMAP_BLOCK_END)
    return "\n".join(lines)


def _ownership_label(item: Mapping[str, Any]) -> str:
    if item.get("unassigned") is True:
        return "`unassigned`"
    if "owner" in item:
        return f"owner `{item['owner']}`"
    return f"team `{item['team']}`"


def _placeholder_clause(item: Mapping[str, Any]) -> str:
    entrypoint = item.get("entrypoint")
    if entrypoint is None:
        return "no importable placeholder yet."
    dotted = f"{entrypoint['module']}.{entrypoint['attribute']}"
    return f"placeholder `{dotted}` raises `UnavailableFeatureError`."


def assert_readme_synced(
    manifest: Mapping[str, Any],
    path: Path = DEFAULT_README,
) -> None:
    """Raise when the README does not contain the rendered planned section."""

    readme = path.read_text(encoding="utf-8")
    expected = render_roadmap_status(manifest)
    actual = _marked_block(readme, path)
    if actual != expected:
        raise RoadmapManifestError(
            f"{path} roadmap status section is stale; run this script with --write-docs"
        )


def sync_readme(
    manifest: Mapping[str, Any],
    path: Path = DEFAULT_README,
) -> Path:
    """Replace the generated roadmap status block in the README."""

    readme = path.read_text(encoding="utf-8")
    _marked_block(readme, path)
    prefix, remainder = readme.split(ROADMAP_BLOCK_START, 1)
    _, suffix = remainder.split(ROADMAP_BLOCK_END, 1)
    path.write_text(prefix + render_roadmap_status(manifest) + suffix, encoding="utf-8")
    return path


def _marked_block(text: str, path: Path) -> str:
    if text.count(ROADMAP_BLOCK_START) != 1 or text.count(ROADMAP_BLOCK_END) != 1:
        raise RoadmapManifestError(
            f"{path} must contain exactly one generated roadmap status block"
        )
    middle = text.split(ROADMAP_BLOCK_START, 1)[1].split(ROADMAP_BLOCK_END, 1)[0]
    return ROADMAP_BLOCK_START + middle + ROADMAP_BLOCK_END


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


def _fail(path: str, detail: str) -> None:
    raise RoadmapManifestError(f"{path}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="rewrite the manifest in canonical block-style YAML",
    )
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="synchronize the generated README roadmap status section",
    )
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    validate_manifest(manifest)

    if arguments.write_manifest:
        arguments.manifest.write_text(canonical_yaml(manifest), encoding="utf-8")
    elif arguments.manifest.read_text(encoding="utf-8") != canonical_yaml(manifest):
        raise RoadmapManifestError(
            f"{arguments.manifest} is not canonical YAML "
            "(sorted keys and block style required)"
        )

    checked = assert_planned_entrypoints_unavailable(manifest)

    if arguments.write_docs:
        sync_readme(manifest, arguments.readme)
    else:
        assert_readme_synced(manifest, arguments.readme)

    print(
        f"Validated {len(manifest['items'])} roadmap items "
        f"({checked} with importable placeholders)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST",
    "DEFAULT_README",
    "ROADMAP_BLOCK_END",
    "ROADMAP_BLOCK_START",
    "RoadmapManifestError",
    "assert_planned_entrypoints_unavailable",
    "assert_readme_synced",
    "canonical_yaml",
    "load_manifest",
    "render_roadmap_status",
    "sync_readme",
    "validate_entry",
    "validate_manifest",
]
