"""Focused tests for exact optional dependency range changelog evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.changelog_gate import (
    ChangelogGateError,
    MigrationGuideRecord,
    MonitoredArtifact,
    OptionalDependencyRangeChange,
    changelog_was_updated,
    find_optional_dependency_range_changes,
    find_removed_api_paths,
    validate_breaking_change_migration_guide,
    validate_change_set_changelog,
    validate_optional_dependency_range_changelog,
)


def _write_pyproject(path: Path, extras: dict[str, list[str]]) -> None:
    lines = ["[project]", 'name = "example"', 'version = "1.0.0"', ""]
    lines.append("[project.optional-dependencies]")
    for extra, declarations in extras.items():
        rendered = ", ".join(f'"{declaration}"' for declaration in declarations)
        lines.append(f"{extra} = [{rendered}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_range_change_requires_exact_old_new_dependency_and_affected_extra(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "previous.toml"
    current = tmp_path / "current.toml"
    changelog = tmp_path / "CHANGELOG.md"
    _write_pyproject(
        previous,
        {"dev": ["tool>=1"], "llm": ["provider>=1,<2"]},
    )
    _write_pyproject(
        current,
        {"dev": ["tool>=2"], "llm": ["provider>=2,<3"]},
    )
    changelog.write_text(
        "# Changelog\n\n"
        "- Optional dependency range: extra=`llm`; dependency=`provider`; "
        "old=`provider>=1,<2`; new=`provider>=2,<3`.\n",
        encoding="utf-8",
    )

    with pytest.raises(ChangelogGateError) as caught:
        validate_optional_dependency_range_changelog(previous, current, changelog)

    message = str(caught.value)
    assert "extra=`dev`" in message
    assert "dependency=`tool`" in message
    assert "old=`tool>=1`" in message
    assert "new=`tool>=2`" in message

    changelog.write_text(
        changelog.read_text(encoding="utf-8")
        + "- Optional dependency range: extra=`dev`; dependency=`tool`; "
        "old=`tool>=1`; new=`tool>=2`.\n",
        encoding="utf-8",
    )

    changes = validate_optional_dependency_range_changelog(previous, current, changelog)
    assert changes == (
        OptionalDependencyRangeChange("dev", "tool", "tool>=1", "tool>=2"),
        OptionalDependencyRangeChange(
            "llm", "provider", "provider>=1,<2", "provider>=2,<3"
        ),
    )


@pytest.mark.parametrize(
    ("record"),
    [
        (
            "- Optional dependency range: extra=`wrong`; dependency=`provider`; "
            "old=`provider>=1`; new=`provider>=2`."
        ),
        (
            "- Optional dependency range: extra=`llm`; dependency=`provider`; "
            "old=`provider>=0`; new=`provider>=2`."
        ),
        (
            "- Optional dependency range: extra=`llm`; dependency=`provider`; "
            "old=`provider>=1`; new=`provider>=3`."
        ),
    ],
    ids=["wrong-extra", "wrong-old", "wrong-new"],
)
def test_near_match_does_not_satisfy_exact_range_record(
    tmp_path: Path, record: str
) -> None:
    previous = tmp_path / "previous.toml"
    current = tmp_path / "current.toml"
    changelog = tmp_path / "CHANGELOG.md"
    _write_pyproject(previous, {"llm": ["provider>=1"]})
    _write_pyproject(current, {"llm": ["provider>=2"]})
    changelog.write_text(f"# Changelog\n\n{record}\n", encoding="utf-8")

    with pytest.raises(ChangelogGateError, match="Required exact records"):
        validate_optional_dependency_range_changelog(previous, current, changelog)


def test_additions_removals_and_reordering_are_not_range_changes() -> None:
    changes = find_optional_dependency_range_changes(
        {
            "llm": ("provider>=1", "removed>=1"),
            "removed-extra": ("unused>=1",),
        },
        {
            "llm": ("added>=1", "provider>=1"),
            "added-extra": ("new>=1",),
        },
    )

    assert changes == ()


def test_dependency_names_are_normalized_before_range_comparison() -> None:
    changes = find_optional_dependency_range_changes(
        {"dev": ("Example_Package>=1",)},
        {"dev": ("example-package>=2",)},
    )

    assert changes == (
        OptionalDependencyRangeChange(
            "dev", "example-package", "Example_Package>=1", "example-package>=2"
        ),
    )


def test_malformed_range_record_is_rejected_with_line_number(tmp_path: Path) -> None:
    previous = tmp_path / "previous.toml"
    current = tmp_path / "current.toml"
    changelog = tmp_path / "CHANGELOG.md"
    _write_pyproject(previous, {"llm": ["provider>=1"]})
    _write_pyproject(current, {"llm": ["provider>=1"]})
    changelog.write_text(
        "# Changelog\n\n- Optional dependency range: llm provider>=1 to provider>=2\n",
        encoding="utf-8",
    )

    with pytest.raises(ChangelogGateError, match="changelog line 3"):
        validate_optional_dependency_range_changelog(previous, current, changelog)


# ---------------------------------------------------------------------------
# Same-change-set changelog gate (Requirements 8.7, 10.7)
# ---------------------------------------------------------------------------

_PREVIOUS_CHANGELOG = "# Changelog\n\n## [0.7.0] - 2026-07-24\n- Initial entry.\n"
_UPDATED_CHANGELOG = (
    "# Changelog\n\n"
    "## [0.8.0] - 2026-08-01\n- Documented the new behavior.\n\n"
    "## [0.7.0] - 2026-07-24\n- Initial entry.\n"
)


def test_changed_artifact_without_new_changelog_entry_is_rejected() -> None:
    artifacts = [
        MonitoredArtifact(
            name="api-snapshot",
            previous='{"exports": [], "package_version": "0.7.0"}',
            current='{"exports": ["experia.New"], "package_version": "0.7.0"}',
            is_json=True,
        ),
        MonitoredArtifact(
            name="documented-status",
            previous="Postgres: planned",
            current="Postgres: planned",
        ),
    ]

    with pytest.raises(ChangelogGateError) as caught:
        validate_change_set_changelog(
            artifacts, _PREVIOUS_CHANGELOG, _PREVIOUS_CHANGELOG
        )

    message = str(caught.value)
    assert "api-snapshot" in message
    assert "documented-status" not in message


def test_changed_artifact_with_new_changelog_entry_reports_changed_names() -> None:
    artifacts = [
        MonitoredArtifact(
            name="prompt-behavior",
            previous="wrap each memory in markers",
            current="wrap each memory in escaped markers",
        ),
    ]

    changed = validate_change_set_changelog(
        artifacts, _PREVIOUS_CHANGELOG, _UPDATED_CHANGELOG
    )

    assert changed == ("prompt-behavior",)


def test_unchanged_artifacts_never_require_a_changelog_entry() -> None:
    artifacts = [
        MonitoredArtifact(
            name="optional-range",
            previous="provider>=1,<2",
            current="provider>=1,<2",
        ),
    ]

    assert (
        validate_change_set_changelog(
            artifacts, _PREVIOUS_CHANGELOG, _PREVIOUS_CHANGELOG
        )
        == ()
    )


def test_json_key_reordering_and_whitespace_do_not_count_as_a_change() -> None:
    artifact = MonitoredArtifact(
        name="api-snapshot",
        previous='{"a": 1, "b": 2}',
        current='{\n  "b": 2,\n  "a": 1\n}\n',
        is_json=True,
    )

    assert artifact.changed() is False
    assert (
        validate_change_set_changelog(
            [artifact], _PREVIOUS_CHANGELOG, _PREVIOUS_CHANGELOG
        )
        == ()
    )


def test_changelog_reordering_without_new_content_is_not_an_update() -> None:
    reordered = "# Changelog\n\n## [0.7.0] - 2026-07-24\n\n- Initial entry.\n"

    assert changelog_was_updated(_PREVIOUS_CHANGELOG, reordered) is False
    assert changelog_was_updated(_PREVIOUS_CHANGELOG, _UPDATED_CHANGELOG) is True


# ---------------------------------------------------------------------------
# Breaking-change migration-guide gate (Requirements 4.6, 10.12)
# ---------------------------------------------------------------------------

_NOT_DEPRECATED = {
    "is_deprecated": False,
    "message": None,
    "replacement": None,
    "since": None,
}


def _member(name: str, *, deprecation: dict | None = None) -> dict:
    return {
        "name": name,
        "kind": "method",
        "async": False,
        "signature": {"parameters": [], "return": "None"},
        "deprecation": deprecation or dict(_NOT_DEPRECATED),
    }


def _export(
    path: str,
    *,
    members: list[dict] | None = None,
    deprecation: dict | None = None,
) -> dict:
    return {
        "path": path,
        "target": path,
        "kind": "class",
        "async": False,
        "signature": {"parameters": [], "return": "None"},
        "members": members or [],
        "deprecation": deprecation or dict(_NOT_DEPRECATED),
    }


def _snapshot(version: str, exports: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "package": "experia",
        "package_version": version,
        "major_version": int(version.split(".", 1)[0]),
        "exports": exports,
    }


def _deprecation(replacement: str, *, since: str = "0.6.0") -> dict:
    return {
        "is_deprecated": True,
        "message": f"Legacy API is deprecated; use {replacement} instead.",
        "replacement": replacement,
        "since": since,
    }


def test_same_major_removal_requires_a_greater_major_version() -> None:
    baseline = _snapshot("0.7.0", [_export("experia.Legacy"), _export("experia.Keep")])
    candidate = _snapshot("0.8.0", [_export("experia.Keep")])

    with pytest.raises(ChangelogGateError, match="requires a greater major version"):
        validate_breaking_change_migration_guide(baseline, candidate, "")


def test_greater_major_removal_without_guide_record_is_rejected() -> None:
    baseline = _snapshot(
        "0.8.0",
        [_export("experia.Legacy", deprecation=_deprecation("experia.NewThing"))],
    )
    candidate = _snapshot("1.0.0", [])

    with pytest.raises(ChangelogGateError) as caught:
        validate_breaking_change_migration_guide(baseline, candidate, "")

    message = str(caught.value)
    assert "Missing records" in message
    # The suggested record names the recorded deprecation replacement.
    assert "path=`experia.Legacy`; replacement=`experia.NewThing`" in message


def test_greater_major_removal_with_matching_guide_record_passes() -> None:
    baseline = _snapshot(
        "0.8.0",
        [
            _export("experia.Legacy", deprecation=_deprecation("experia.NewThing")),
            _export(
                "experia.Store",
                members=[
                    _member("run"),
                    _member(
                        "old_method",
                        deprecation=_deprecation("experia.Store.run"),
                    ),
                ],
            ),
        ],
    )
    candidate = _snapshot("1.0.0", [_export("experia.Store", members=[_member("run")])])

    # A removed export records only the export path; a surviving export that
    # loses a member records the removed member path.
    assert find_removed_api_paths(baseline, candidate) == (
        "experia.Legacy",
        "experia.Store.old_method",
    )

    guide = (
        "# Migration guide\n\n"
        "## 1.0.0\n"
        "- API removal: path=`experia.Legacy`; replacement=`experia.NewThing`.\n"
        "- API removal: path=`experia.Store.old_method`; "
        "replacement=`experia.Store.run`.\n"
    )

    records = validate_breaking_change_migration_guide(baseline, candidate, guide)

    assert records == (
        MigrationGuideRecord("experia.Legacy", "experia.NewThing"),
        MigrationGuideRecord("experia.Store.old_method", "experia.Store.run"),
    )


def test_guide_replacement_must_match_recorded_deprecation_replacement() -> None:
    baseline = _snapshot(
        "0.8.0",
        [_export("experia.Legacy", deprecation=_deprecation("experia.NewThing"))],
    )
    candidate = _snapshot("1.0.0", [])
    guide = "- API removal: path=`experia.Legacy`; replacement=`experia.WrongThing`.\n"

    with pytest.raises(ChangelogGateError, match="does not match the recorded"):
        validate_breaking_change_migration_guide(baseline, candidate, guide)


def test_malformed_migration_guide_record_is_rejected_with_line_number() -> None:
    baseline = _snapshot("0.8.0", [_export("experia.Legacy")])
    candidate = _snapshot("1.0.0", [])
    guide = "intro\n- API removal: experia.Legacy replaced by experia.NewThing\n"

    with pytest.raises(ChangelogGateError, match="line 2"):
        validate_breaking_change_migration_guide(baseline, candidate, guide)
