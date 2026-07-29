"""Changelog and migration-guide policy automation.

This module hosts credential-free gates that keep the changelog and migration
guide honest across a change set:

* **Optional dependency ranges** (Requirement 8.7): the gate compares two
  ``pyproject.toml`` files and requires an exact machine-readable changelog
  record naming the extra, dependency, previous declaration, and new
  declaration for every changed optional dependency declaration.
* **Same-change-set changelog** (Requirements 8.7, 10.7): whenever a monitored
  documented artifact changes -- the Public API snapshot, documented prompt
  behavior, an optional-dependency range, or documented product status -- the
  changelog must gain a new entry in the same change set.
* **Breaking-change migration guide** (Requirements 4.6, 10.12): an approved
  breaking public API change (a removal or narrowing) is allowed only under a
  greater major version, and every removed public API path must be identified
  with its replacement in the migration guide.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:  # pragma: no cover - import shim mirrors scripts/api_gate.py
    from scripts.api_compatibility import ReleaseVersion, compare_snapshots
except ModuleNotFoundError:  # pragma: no cover
    from api_compatibility import ReleaseVersion, compare_snapshots

_CHANGE_PREFIX = "- Optional dependency range:"
_CHANGE_PATTERN = re.compile(
    r"^- Optional dependency range: "
    r"extra=`(?P<extra>[^`]+)`; "
    r"dependency=`(?P<dependency>[^`]+)`; "
    r"old=`(?P<old>[^`]+)`; "
    r"new=`(?P<new>[^`]+)`\.$"
)
_DEPENDENCY_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


class ChangelogGateError(RuntimeError):
    """Raised when optional dependency range evidence is invalid or missing."""


@dataclass(frozen=True, order=True)
class OptionalDependencyRangeChange:
    """One exact dependency declaration transition within one extra."""

    extra: str
    dependency: str
    old: str
    new: str

    def changelog_record(self) -> str:
        """Return the exact changelog line required for this transition."""
        return (
            f"{_CHANGE_PREFIX} extra=`{self.extra}`; "
            f"dependency=`{self.dependency}`; old=`{self.old}`; "
            f"new=`{self.new}`."
        )


def load_optional_dependencies(path: Path) -> dict[str, tuple[str, ...]]:
    """Load and validate ``project.optional-dependencies`` from TOML."""
    try:
        with path.open("rb") as pyproject_file:
            document = tomllib.load(pyproject_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ChangelogGateError(
            f"Cannot read optional dependencies from {path}: {error}"
        ) from error

    optional = document.get("project", {}).get("optional-dependencies", {})
    if not isinstance(optional, Mapping):
        raise ChangelogGateError(
            f"project.optional-dependencies must be a table in {path}."
        )

    result: dict[str, tuple[str, ...]] = {}
    for extra, declarations in optional.items():
        if not isinstance(extra, str) or not isinstance(declarations, list):
            raise ChangelogGateError(
                f"Invalid optional dependency entry in {path}: extra={extra!r}."
            )
        if any(not isinstance(declaration, str) for declaration in declarations):
            raise ChangelogGateError(
                f"Optional dependency declarations must be strings for extra "
                f"{extra!r} in {path}."
            )
        result[extra] = tuple(declarations)
    return result


def find_optional_dependency_range_changes(
    previous: Mapping[str, Sequence[str]],
    current: Mapping[str, Sequence[str]],
) -> tuple[OptionalDependencyRangeChange, ...]:
    """Return exact changed declarations, excluding package additions/removals."""
    changes: list[OptionalDependencyRangeChange] = []
    for extra in sorted(previous.keys() & current.keys()):
        previous_by_name = _declarations_by_name(extra, previous[extra])
        current_by_name = _declarations_by_name(extra, current[extra])
        for dependency in sorted(previous_by_name.keys() & current_by_name.keys()):
            old = previous_by_name[dependency]
            new = current_by_name[dependency]
            if old != new:
                changes.append(
                    OptionalDependencyRangeChange(
                        extra=extra,
                        dependency=dependency,
                        old=old,
                        new=new,
                    )
                )
    return tuple(changes)


def parse_optional_dependency_range_records(
    changelog_text: str,
) -> tuple[OptionalDependencyRangeChange, ...]:
    """Parse exact optional dependency range records from a changelog."""
    records: list[OptionalDependencyRangeChange] = []
    for line_number, line in enumerate(changelog_text.splitlines(), start=1):
        if not line.startswith(_CHANGE_PREFIX):
            continue
        match = _CHANGE_PATTERN.fullmatch(line)
        if match is None:
            raise ChangelogGateError(
                "Malformed optional dependency range record at changelog line "
                f"{line_number}: {line!r}. Expected format: "
                f"{OptionalDependencyRangeChange('EXTRA', 'PACKAGE', 'OLD', 'NEW').changelog_record()}"
            )
        records.append(OptionalDependencyRangeChange(**match.groupdict()))

    duplicates = sorted(record for record in set(records) if records.count(record) > 1)
    if duplicates:
        rendered = ", ".join(record.changelog_record() for record in duplicates)
        raise ChangelogGateError(
            f"Duplicate optional dependency range records observed: {rendered}"
        )
    return tuple(records)


def validate_optional_dependency_range_changelog(
    previous_pyproject: Path,
    current_pyproject: Path,
    changelog: Path,
) -> tuple[OptionalDependencyRangeChange, ...]:
    """Require exact changelog evidence for every observed optional range change."""
    previous = load_optional_dependencies(previous_pyproject)
    current = load_optional_dependencies(current_pyproject)
    changes = find_optional_dependency_range_changes(previous, current)
    try:
        changelog_text = changelog.read_text(encoding="utf-8")
    except OSError as error:
        raise ChangelogGateError(
            f"Cannot read changelog {changelog}: {error}"
        ) from error
    records = parse_optional_dependency_range_records(changelog_text)
    record_set = set(records)
    missing = tuple(change for change in changes if change not in record_set)
    if missing:
        required = "\n".join(f"- {change.changelog_record()}" for change in missing)
        observed = tuple(
            record
            for record in records
            if any(
                record.extra == change.extra and record.dependency == change.dependency
                for change in missing
            )
        )
        observed_text = (
            "none"
            if not observed
            else " | ".join(record.changelog_record() for record in observed)
        )
        raise ChangelogGateError(
            "Optional dependency range changelog validation failed. "
            f"Observed matching records: {observed_text}. Required exact records:\n"
            f"{required}"
        )
    return changes


def _declarations_by_name(extra: str, declarations: Sequence[str]) -> dict[str, str]:
    indexed: dict[str, str] = {}
    for declaration in declarations:
        match = _DEPENDENCY_NAME_PATTERN.match(declaration)
        if match is None:
            raise ChangelogGateError(
                f"Cannot determine dependency name for {declaration!r} in extra {extra!r}."
            )
        dependency = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if dependency in indexed:
            raise ChangelogGateError(
                f"Duplicate dependency {dependency!r} in extra {extra!r}: "
                f"{indexed[dependency]!r} and {declaration!r}."
            )
        indexed[dependency] = declaration
    return indexed


# ---------------------------------------------------------------------------
# Same-change-set changelog gate (Requirements 8.7, 10.7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoredArtifact:
    """One documented artifact whose change requires a same-change-set entry.

    ``is_json`` compares two revisions of a JSON document by canonical meaning
    (sorted keys) rather than byte identity, so that reformatting or key
    reordering of, for example, the Public API snapshot does not on its own
    demand a changelog entry.
    """

    name: str
    previous: str
    current: str
    is_json: bool = False

    def changed(self) -> bool:
        """Return whether this artifact meaningfully changed in the change set."""
        return _normalize_artifact(self.previous, self.is_json) != _normalize_artifact(
            self.current, self.is_json
        )


def _normalize_artifact(content: str, is_json: bool) -> str:
    if is_json:
        try:
            return json.dumps(json.loads(content), sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    return "\n".join(line.rstrip() for line in content.splitlines()).strip("\n")


def _content_lines(text: str) -> list[str]:
    return [stripped for line in text.splitlines() if (stripped := line.strip())]


def changelog_was_updated(previous_changelog: str, current_changelog: str) -> bool:
    """Return whether the changelog gained at least one new content line."""
    previous_lines = _content_lines(previous_changelog)
    current_lines = _content_lines(current_changelog)
    if current_lines == previous_lines:
        return False
    previous_set = set(previous_lines)
    return any(line not in previous_set for line in current_lines)


def find_changed_artifacts(
    artifacts: Sequence[MonitoredArtifact],
) -> tuple[str, ...]:
    """Return the names of monitored artifacts that changed in this change set."""
    names = tuple(artifact.name for artifact in artifacts if artifact.changed())
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ChangelogGateError(
            f"Duplicate monitored artifact names observed: {', '.join(duplicates)}."
        )
    return names


def validate_change_set_changelog(
    artifacts: Sequence[MonitoredArtifact],
    previous_changelog: str,
    current_changelog: str,
) -> tuple[str, ...]:
    """Require a new changelog entry when any monitored artifact changes.

    Covers the Public API snapshot, documented prompt behavior, optional
    dependency ranges, and documented status: whenever any of them changes in a
    change set, the changelog must gain a new entry in that same change set.
    """
    changed = find_changed_artifacts(artifacts)
    if changed and not changelog_was_updated(previous_changelog, current_changelog):
        raise ChangelogGateError(
            "Change-set changelog validation failed. The following documented "
            f"artifacts changed without a new changelog entry: {', '.join(changed)}. "
            "Update the changelog in the same change set."
        )
    return changed


# ---------------------------------------------------------------------------
# Breaking-change migration-guide gate (Requirements 4.6, 10.12)
# ---------------------------------------------------------------------------

_MIGRATION_PREFIX = "- API removal:"
_MIGRATION_PATTERN = re.compile(
    r"^- API removal: "
    r"path=`(?P<path>[^`]+)`; "
    r"replacement=`(?P<replacement>[^`]+)`\.$"
)

# ``compare_snapshots`` issue codes that represent a breaking removal or a
# narrowing of the accepted call surface within one major version.
_BREAKING_ISSUE_CODES = frozenset(
    {
        "export_removed",
        "member_removed",
        "enum_value_removed",
        "enum_value_changed",
        "parameter_removed",
        "parameter_kind_narrowed",
        "parameter_became_required",
        "parameter_type_narrowed",
        "required_parameter_added",
        "positional_parameter_inserted",
        "positional_order_changed",
        "kind_changed",
        "async_changed",
        "signature_changed",
        "return_changed",
        "default_changed",
        "version_regressed",
    }
)


@dataclass(frozen=True, order=True)
class BreakingChange:
    """One removal or narrowing incompatibility between two API snapshots."""

    code: str
    path: str
    detail: str


@dataclass(frozen=True, order=True)
class MigrationGuideRecord:
    """One exact migration-guide record for a removed public API path."""

    path: str
    replacement: str

    def migration_record(self) -> str:
        """Return the exact migration-guide line required for this removal."""
        return (
            f"{_MIGRATION_PREFIX} path=`{self.path}`; replacement=`{self.replacement}`."
        )


def _exports_by_path(snapshot: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    exports = snapshot.get("exports")
    if not isinstance(exports, Sequence) or isinstance(exports, (str, bytes)):
        return {}
    return {
        str(item["path"]): item
        for item in exports
        if isinstance(item, Mapping) and "path" in item
    }


def _members_by_name(
    export: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    members = export.get("members")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        return {}
    return {
        str(item["name"]): item
        for item in members
        if isinstance(item, Mapping) and "name" in item
    }


def find_removed_api_paths(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[str, ...]:
    """Return every export/member path present in baseline but absent in candidate."""
    removed: list[str] = []
    baseline_exports = _exports_by_path(baseline)
    candidate_exports = _exports_by_path(candidate)
    for path, baseline_export in baseline_exports.items():
        candidate_export = candidate_exports.get(path)
        if candidate_export is None:
            removed.append(path)
            continue
        baseline_members = _members_by_name(baseline_export)
        candidate_members = _members_by_name(candidate_export)
        for name in baseline_members.keys() - candidate_members.keys():
            removed.append(f"{path}.{name}")
    return tuple(sorted(removed))


def find_breaking_changes(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[BreakingChange, ...]:
    """Return the removal/narrowing incompatibilities between two snapshots.

    Within one major version ``compare_snapshots`` reports removals and
    narrowed signatures as incompatibilities; across a greater major version it
    permits them, so this returns an empty tuple for an approved major bump.
    """
    report = compare_snapshots(baseline, candidate)
    return tuple(
        BreakingChange(code=issue.code, path=issue.path, detail=issue.detail)
        for issue in report.issues
        if issue.code in _BREAKING_ISSUE_CODES
    )


def parse_migration_guide_records(
    migration_guide_text: str,
) -> tuple[MigrationGuideRecord, ...]:
    """Parse exact ``- API removal:`` records from a migration guide."""
    records: list[MigrationGuideRecord] = []
    for line_number, line in enumerate(migration_guide_text.splitlines(), start=1):
        if not line.startswith(_MIGRATION_PREFIX):
            continue
        match = _MIGRATION_PATTERN.fullmatch(line)
        if match is None:
            raise ChangelogGateError(
                "Malformed API removal migration-guide record at line "
                f"{line_number}: {line!r}. Expected format: "
                f"{MigrationGuideRecord('experia.Old', 'experia.New').migration_record()}"
            )
        records.append(MigrationGuideRecord(**match.groupdict()))

    duplicates = sorted(record for record in set(records) if records.count(record) > 1)
    if duplicates:
        rendered = ", ".join(record.migration_record() for record in duplicates)
        raise ChangelogGateError(
            f"Duplicate API removal migration-guide records observed: {rendered}"
        )
    return tuple(records)


def _recorded_replacements(snapshot: Mapping[str, object]) -> dict[str, str]:
    """Map each deprecated export/member path to its recorded replacement."""
    replacements: dict[str, str] = {}
    for path, export in _exports_by_path(snapshot).items():
        _record_replacement(replacements, path, export.get("deprecation"))
        for name, member in _members_by_name(export).items():
            _record_replacement(
                replacements, f"{path}.{name}", member.get("deprecation")
            )
    return replacements


def _record_replacement(target: dict[str, str], path: str, deprecation: object) -> None:
    if not isinstance(deprecation, Mapping) or not deprecation.get("is_deprecated"):
        return
    replacement = deprecation.get("replacement")
    if isinstance(replacement, str) and replacement.strip():
        target[path] = replacement


def _parse_snapshot_version(
    snapshot: Mapping[str, object], label: str
) -> ReleaseVersion:
    try:
        return ReleaseVersion.parse(snapshot.get("package_version"))
    except ValueError as error:
        raise ChangelogGateError(
            f"Cannot read {label} package_version: {error}"
        ) from error


def validate_breaking_change_migration_guide(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    migration_guide_text: str,
) -> tuple[MigrationGuideRecord, ...]:
    """Require breaking API changes to ship in a greater major with a guide.

    Within the same (or a lower) major version, any removal or narrowing is
    rejected outright. In a greater major version, every removed public API
    path must be identified with its replacement in the migration guide, and
    that replacement must match the deprecation replacement recorded for the
    path in the baseline snapshot when one exists.
    """
    baseline_version = _parse_snapshot_version(baseline, "baseline")
    candidate_version = _parse_snapshot_version(candidate, "candidate")

    if candidate_version.major <= baseline_version.major:
        breaking = find_breaking_changes(baseline, candidate)
        if breaking:
            rendered = "; ".join(
                f"[{change.code}] {change.path}" for change in breaking
            )
            raise ChangelogGateError(
                "Breaking public API change requires a greater major version. "
                f"Observed baseline={baseline_version}, candidate="
                f"{candidate_version} with breaking changes: {rendered}."
            )
        return ()

    removed = find_removed_api_paths(baseline, candidate)
    records = parse_migration_guide_records(migration_guide_text)
    record_by_path = {record.path: record for record in records}
    recorded_replacements = _recorded_replacements(baseline)

    missing = tuple(path for path in removed if path not in record_by_path)
    if missing:
        required = "\n".join(
            "- "
            + MigrationGuideRecord(
                path, recorded_replacements.get(path, "REPLACEMENT")
            ).migration_record()
            for path in missing
        )
        raise ChangelogGateError(
            "Breaking major migration-guide validation failed. Each removed "
            "public API path must be identified with its replacement in the "
            f"migration guide. Missing records:\n{required}"
        )

    inconsistent = tuple(
        (path, record_by_path[path].replacement, recorded_replacements[path])
        for path in removed
        if path in recorded_replacements
        and record_by_path[path].replacement != recorded_replacements[path]
    )
    if inconsistent:
        rendered = "; ".join(
            f"{path}: guide=`{guide}` != recorded=`{recorded}`"
            for path, guide, recorded in inconsistent
        )
        raise ChangelogGateError(
            "Migration-guide replacement does not match the recorded deprecation "
            f"replacement: {rendered}."
        )

    return tuple(record_by_path[path] for path in removed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-pyproject", type=Path, required=True)
    parser.add_argument("--current-pyproject", type=Path, required=True)
    parser.add_argument("--changelog", type=Path, required=True)
    arguments = parser.parse_args(argv)

    try:
        changes = validate_optional_dependency_range_changelog(
            arguments.previous_pyproject,
            arguments.current_pyproject,
            arguments.changelog,
        )
    except ChangelogGateError as error:
        print(f"FAIL optional-dependency-changelog: observed {error}", file=sys.stderr)
        return 1

    print(
        "PASS optional-dependency-changelog: observed "
        f"documented_range_changes={len(changes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
