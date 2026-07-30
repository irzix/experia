"""Semantic compatibility checks for canonical Experia API snapshots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

MINOR_DEPRECATION_RELEASES = 2
_VERSION_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)$"
)
_POSITIONAL_KINDS = {"positional_only", "positional_or_keyword"}
_PARAMETER_MODES = {
    "positional_only": frozenset({"positional"}),
    "positional_or_keyword": frozenset({"positional", "keyword"}),
    "keyword_only": frozenset({"keyword"}),
    "var_positional": frozenset({"var_positional"}),
    "var_keyword": frozenset({"var_keyword"}),
}


@dataclass(frozen=True, order=True)
class ReleaseVersion:
    """A strict semantic release version used by API snapshots."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: object) -> ReleaseVersion:
        """Parse the canonical ``MAJOR.MINOR.PATCH`` snapshot format."""
        if not isinstance(value, str):
            raise ValueError("Snapshot package_version must be a string.")
        match = _VERSION_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(
                f"Snapshot package_version must be MAJOR.MINOR.PATCH, got {value!r}."
            )
        return cls(*(int(match.group(part)) for part in ("major", "minor", "patch")))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class CompatibilityIssue:
    """One machine-readable incompatibility in a candidate snapshot."""

    code: str
    path: str
    detail: str


@dataclass(frozen=True)
class CompatibilityReport:
    """The semantic result of comparing two canonical snapshots."""

    baseline_version: str
    candidate_version: str
    issues: tuple[CompatibilityIssue, ...]

    @property
    def compatible(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        """Reject an incompatible candidate with deterministic diagnostics."""
        if self.compatible:
            return
        raise SnapshotCompatibilityError(self)


class SnapshotCompatibilityError(ValueError):
    """Raised when a candidate snapshot breaks the enforced API contract."""

    def __init__(self, report: CompatibilityReport) -> None:
        self.report = report
        lines = [
            f"API snapshot {report.candidate_version} is incompatible with "
            f"{report.baseline_version}:"
        ]
        lines.extend(
            f"- [{issue.code}] {issue.path}: {issue.detail}" for issue in report.issues
        )
        super().__init__("\n".join(lines))


def deprecation_window_elapsed(
    since: str,
    through: str,
    *,
    minimum_minor_releases: int = MINOR_DEPRECATION_RELEASES,
) -> bool:
    """Return whether an API survived the required consecutive minor window.

    A deprecation introduced in ``0.6`` remains protected in ``0.6`` and
    ``0.7`` and has completed a two-minor window at ``0.8``. If the API has
    already survived into a later major, the earlier-major window is complete.
    """
    if minimum_minor_releases < 0:
        raise ValueError("minimum_minor_releases must be non-negative.")
    introduced = ReleaseVersion.parse(since)
    current = ReleaseVersion.parse(through)
    if current < introduced:
        return False
    if current.major > introduced.major:
        return True
    return current.minor - introduced.minor >= minimum_minor_releases


def compare_snapshots(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> CompatibilityReport:
    """Compare canonical snapshots by API meaning rather than JSON identity.

    Export and member ordering, package patch changes, new exports, new enum
    values, new members, and optional parameters that do not alter existing
    positional calls are additive. Within one major, existing paths, callable
    modes, enum values, and accepted signatures must remain compatible.
    """
    issues: list[CompatibilityIssue] = []
    baseline_version = str(baseline.get("package_version", "<missing>"))
    candidate_version = str(candidate.get("package_version", "<missing>"))
    old_version = _snapshot_version(baseline, "baseline", issues)
    new_version = _snapshot_version(candidate, "candidate", issues)

    if baseline.get("schema_version") != candidate.get("schema_version"):
        _issue(
            issues,
            "schema_changed",
            "snapshot",
            "schema_version must match the baseline comparator schema",
        )
    if baseline.get("package") != candidate.get("package"):
        _issue(
            issues,
            "package_changed",
            "snapshot.package",
            "candidate package must match the baseline package",
        )

    old_exports = _items_by_key(baseline.get("exports"), "path")
    new_exports = _items_by_key(candidate.get("exports"), "path")
    _validate_deprecations(candidate, new_version, issues)

    if old_version is not None and new_version is not None:
        if new_version < old_version:
            _issue(
                issues,
                "version_regressed",
                "snapshot.package_version",
                f"candidate {new_version} precedes baseline {old_version}",
            )
        same_major = old_version.major == new_version.major
        if same_major:
            _compare_same_major(old_exports, new_exports, issues)
        elif new_version.major > old_version.major:
            _check_early_major_removals(
                old_exports,
                new_exports,
                old_version,
                issues,
            )

    return CompatibilityReport(
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        issues=tuple(issues),
    )


def assert_snapshots_compatible(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    """Raise ``SnapshotCompatibilityError`` unless the candidate is compatible."""
    compare_snapshots(baseline, candidate).raise_for_errors()


def _snapshot_version(
    snapshot: Mapping[str, Any],
    label: str,
    issues: list[CompatibilityIssue],
) -> ReleaseVersion | None:
    try:
        version = ReleaseVersion.parse(snapshot.get("package_version"))
    except ValueError as error:
        _issue(issues, "invalid_version", f"{label}.package_version", str(error))
        return None
    if snapshot.get("major_version") != version.major:
        _issue(
            issues,
            "major_mismatch",
            f"{label}.major_version",
            f"expected {version.major} for package_version {version}",
        )
    return version


def _items_by_key(value: object, key: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return {}
    return {
        str(item[key]): item
        for item in value
        if isinstance(item, Mapping) and key in item
    }


def _compare_same_major(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    issues: list[CompatibilityIssue],
) -> None:
    for path, old_export in baseline.items():
        new_export = candidate.get(path)
        if new_export is None:
            _issue(
                issues,
                "export_removed",
                path,
                "supported import path was removed within the current major",
            )
            continue
        _compare_api_node(path, old_export, new_export, issues)

        old_members = _items_by_key(old_export.get("members"), "name")
        new_members = _items_by_key(new_export.get("members"), "name")
        for name, old_member in old_members.items():
            member_path = f"{path}.{name}"
            new_member = new_members.get(name)
            if new_member is None:
                _issue(
                    issues,
                    "member_removed",
                    member_path,
                    "supported member was removed within the current major",
                )
                continue
            _compare_api_node(member_path, old_member, new_member, issues)

        old_values = _items_by_key(old_export.get("values"), "name")
        new_values = _items_by_key(new_export.get("values"), "name")
        for name, old_value in old_values.items():
            value_path = f"{path}.{name}"
            new_value = new_values.get(name)
            if new_value is None:
                _issue(
                    issues,
                    "enum_value_removed",
                    value_path,
                    "supported enum value was removed",
                )
            elif new_value.get("value") != old_value.get("value"):
                _issue(
                    issues,
                    "enum_value_changed",
                    value_path,
                    "supported enum value changed",
                )


def _compare_api_node(
    path: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    issues: list[CompatibilityIssue],
) -> None:
    if baseline.get("kind") != candidate.get("kind"):
        _issue(
            issues,
            "kind_changed",
            path,
            f"kind changed from {baseline.get('kind')!r} to {candidate.get('kind')!r}",
        )
    if baseline.get("async") != candidate.get("async"):
        _issue(
            issues,
            "async_changed",
            path,
            "sync/async calling convention changed",
        )
    _compare_signature(
        path,
        baseline.get("signature"),
        candidate.get("signature"),
        issues,
    )


def _compare_signature(
    path: str,
    baseline: object,
    candidate: object,
    issues: list[CompatibilityIssue],
) -> None:
    if baseline is None and candidate is None:
        return
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        _issue(
            issues,
            "signature_changed",
            path,
            "inspectable signature availability changed",
        )
        return

    old_parameters = _parameter_list(baseline)
    new_parameters = _parameter_list(candidate)
    old_by_name = {
        str(parameter.get("name")): parameter for parameter in old_parameters
    }
    new_by_name = {
        str(parameter.get("name")): parameter for parameter in new_parameters
    }

    for name, old_parameter in old_by_name.items():
        parameter_path = f"{path}({name})"
        new_parameter = new_by_name.get(name)
        if new_parameter is None:
            _issue(
                issues,
                "parameter_removed",
                parameter_path,
                "accepted parameter was removed",
            )
            continue
        _compare_parameter(parameter_path, old_parameter, new_parameter, issues)

    old_positional = [
        str(parameter.get("name"))
        for parameter in old_parameters
        if parameter.get("kind") in _POSITIONAL_KINDS
    ]
    candidate_old_positional = [
        str(parameter.get("name"))
        for parameter in new_parameters
        if parameter.get("name") in old_by_name
        and old_by_name[str(parameter.get("name"))].get("kind") in _POSITIONAL_KINDS
    ]
    if candidate_old_positional != old_positional:
        _issue(
            issues,
            "positional_order_changed",
            path,
            "existing positional parameter order changed",
        )

    old_has_varargs = any(
        parameter.get("kind") == "var_positional" for parameter in old_parameters
    )
    last_old_position = max(
        (
            index
            for index, parameter in enumerate(new_parameters)
            if parameter.get("name") in old_positional
        ),
        default=-1,
    )
    for index, parameter in enumerate(new_parameters):
        name = str(parameter.get("name"))
        if name in old_by_name:
            continue
        parameter_path = f"{path}({name})"
        if bool(parameter.get("required")):
            _issue(
                issues,
                "required_parameter_added",
                parameter_path,
                "new required parameter narrows accepted calls",
            )
        if parameter.get("kind") in _POSITIONAL_KINDS and (
            index < last_old_position or old_has_varargs
        ):
            _issue(
                issues,
                "positional_parameter_inserted",
                parameter_path,
                "new positional parameter changes an existing positional call",
            )

    if baseline.get("return") != candidate.get("return"):
        _issue(
            issues,
            "return_changed",
            path,
            "return annotation changed",
        )


def _parameter_list(signature: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    parameters = signature.get("parameters")
    if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes)):
        return []
    return [parameter for parameter in parameters if isinstance(parameter, Mapping)]


def _compare_parameter(
    path: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    issues: list[CompatibilityIssue],
) -> None:
    old_kind = str(baseline.get("kind"))
    new_kind = str(candidate.get("kind"))
    old_modes = _PARAMETER_MODES.get(old_kind, frozenset())
    new_modes = _PARAMETER_MODES.get(new_kind, frozenset())
    if not old_modes.issubset(new_modes):
        _issue(
            issues,
            "parameter_kind_narrowed",
            path,
            f"parameter kind changed from {old_kind!r} to {new_kind!r}",
        )

    old_required = bool(baseline.get("required"))
    new_required = bool(candidate.get("required"))
    if not old_required and new_required:
        _issue(
            issues,
            "parameter_became_required",
            path,
            "optional parameter became required",
        )
    if not old_required and baseline.get("default") != candidate.get("default"):
        _issue(
            issues,
            "default_changed",
            path,
            "existing default value changed",
        )
    if not _input_annotation_is_widened_or_equal(
        baseline.get("annotation"), candidate.get("annotation")
    ):
        _issue(
            issues,
            "parameter_type_narrowed",
            path,
            f"accepted type changed from {baseline.get('annotation')!r} "
            f"to {candidate.get('annotation')!r}",
        )


def _input_annotation_is_widened_or_equal(old: object, new: object) -> bool:
    if old == new or new is None or new == "typing.Any":
        return True
    if old is None or old == "typing.Any":
        return False
    if not isinstance(old, str) or not isinstance(new, str):
        return False
    return set(_split_union(old)).issubset(_split_union(new))


def _split_union(annotation: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(annotation):
        if character in "[({":
            depth += 1
        elif character in "])}":
            depth = max(0, depth - 1)
        elif character == "|" and depth == 0:
            parts.append(annotation[start:index].strip())
            start = index + 1
    parts.append(annotation[start:].strip())
    return tuple(parts)


def _validate_deprecations(
    snapshot: Mapping[str, Any],
    version: ReleaseVersion | None,
    issues: list[CompatibilityIssue],
) -> None:
    for export in _items_by_key(snapshot.get("exports"), "path").values():
        path = str(export.get("path"))
        _validate_deprecation(path, export.get("deprecation"), version, issues)
        for member in _items_by_key(export.get("members"), "name").values():
            _validate_deprecation(
                f"{path}.{member.get('name')}",
                member.get("deprecation"),
                version,
                issues,
            )


def _validate_deprecation(
    path: str,
    value: object,
    version: ReleaseVersion | None,
    issues: list[CompatibilityIssue],
) -> None:
    if not isinstance(value, Mapping) or not value.get("is_deprecated"):
        return
    since = value.get("since")
    replacement = value.get("replacement")
    message = value.get("message")
    if not isinstance(replacement, str) or not replacement.strip():
        _issue(
            issues,
            "deprecation_missing_replacement",
            path,
            "deprecated API must name a replacement",
        )
    if not isinstance(message, str) or not message.strip():
        _issue(
            issues,
            "deprecation_missing_message",
            path,
            "deprecated API must define its warning message",
        )
    elif isinstance(replacement, str) and replacement not in message:
        _issue(
            issues,
            "deprecation_message_missing_replacement",
            path,
            "deprecation warning message must contain the replacement path",
        )
    try:
        introduced = ReleaseVersion.parse(since)
    except ValueError as error:
        _issue(issues, "deprecation_invalid_since", path, str(error))
    else:
        if version is not None and introduced > version:
            _issue(
                issues,
                "deprecation_from_future",
                path,
                f"deprecation release {introduced} follows snapshot {version}",
            )


def _check_early_major_removals(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    through: ReleaseVersion,
    issues: list[CompatibilityIssue],
) -> None:
    for path, old_export in baseline.items():
        new_export = candidate.get(path)
        if new_export is None:
            _check_deprecation_removal(path, old_export, through, issues)
            continue
        old_members = _items_by_key(old_export.get("members"), "name")
        new_members = _items_by_key(new_export.get("members"), "name")
        for name, old_member in old_members.items():
            if name not in new_members:
                _check_deprecation_removal(
                    f"{path}.{name}", old_member, through, issues
                )


def _check_deprecation_removal(
    path: str,
    node: Mapping[str, Any],
    through: ReleaseVersion,
    issues: list[CompatibilityIssue],
) -> None:
    deprecation = node.get("deprecation")
    if not isinstance(deprecation, Mapping) or not deprecation.get("is_deprecated"):
        return
    since = deprecation.get("since")
    try:
        elapsed = deprecation_window_elapsed(str(since), str(through))
    except ValueError:
        return
    if not elapsed:
        _issue(
            issues,
            "deprecation_window_incomplete",
            path,
            f"deprecated API must remain callable for at least "
            f"{MINOR_DEPRECATION_RELEASES} minor releases",
        )


def _issue(
    issues: list[CompatibilityIssue],
    code: str,
    path: str,
    detail: str,
) -> None:
    issues.append(CompatibilityIssue(code=code, path=path, detail=detail))


__all__ = [
    "CompatibilityIssue",
    "CompatibilityReport",
    "MINOR_DEPRECATION_RELEASES",
    "ReleaseVersion",
    "SnapshotCompatibilityError",
    "assert_snapshots_compatible",
    "compare_snapshots",
    "deprecation_window_elapsed",
]
