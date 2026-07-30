"""Focused tests for credential-free release identity validation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scripts.release_identity import (
    ReleaseIdentity,
    ReleaseIdentityError,
    load_changelog_releases,
    load_project_version,
    main,
    parse_release_date,
    parse_release_tag,
    validate_release_identity,
)

ROOT = Path(__file__).resolve().parents[1]


def _release_files(
    tmp_path: Path,
    *,
    project_version: str = "1.2.3",
    changelog: str = "# Changelog\n\n## [1.2.3] - 2026-07-24\n",
) -> tuple[Path, Path]:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f'[build-system]\nrequires = []\n\n[project]\nname = "sample"\n'
        f'version = "{project_version}"\n',
        encoding="utf-8",
    )
    changelog_path = tmp_path / "CHANGELOG.md"
    changelog_path.write_text(changelog, encoding="utf-8")
    return pyproject, changelog_path


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2.3"])
def test_validator_accepts_zero_or_one_v_for_exact_release_identity(
    tmp_path: Path, tag: str
):
    pyproject, changelog = _release_files(tmp_path)

    identity = validate_release_identity(
        tag,
        date(2026, 7, 24),
        pyproject_path=pyproject,
        changelog_path=changelog,
    )

    assert identity == ReleaseIdentity(
        tag=tag,
        version="1.2.3",
        release_date="2026-07-24",
        changelog_line=3,
    )


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "v",
        "1.2",
        "1.2.3.4",
        "vv1.2.3",
        "V1.2.3",
        "01.2.3",
        "1.02.3",
        "1.2.03",
        "1.2.3rc1",
        "1.2.3+build",
        " 1.2.3",
        "1.2.3\n",
    ],
)
def test_tag_parser_rejects_every_non_exact_tag_with_observed_value(tag: str):
    with pytest.raises(ReleaseIdentityError) as caught:
        parse_release_tag(tag)

    assert caught.value.check == "tag-syntax"
    assert caught.value.observed == {"tag": tag}
    assert f"tag={tag!r}" in str(caught.value)


def test_project_version_mismatch_names_both_observed_versions(tmp_path: Path):
    pyproject, changelog = _release_files(tmp_path, project_version="1.2.4")

    with pytest.raises(ReleaseIdentityError) as caught:
        validate_release_identity(
            "v1.2.3",
            "2026-07-24",
            pyproject_path=pyproject,
            changelog_path=changelog,
        )

    assert caught.value.check == "tag-project-version-identity"
    assert caught.value.observed == {
        "tag_version": "1.2.3",
        "project_version": "1.2.4",
    }
    assert "tag_version='1.2.3'" in str(caught.value)
    assert "project_version='1.2.4'" in str(caught.value)


def test_missing_changelog_version_names_observed_release_versions(tmp_path: Path):
    pyproject, changelog = _release_files(
        tmp_path,
        changelog="# Changelog\n\n## [1.2.2] - 2026-07-23\n",
    )

    with pytest.raises(ReleaseIdentityError) as caught:
        validate_release_identity(
            "1.2.3",
            "2026-07-24",
            pyproject_path=pyproject,
            changelog_path=changelog,
        )

    assert caught.value.check == "changelog-version-identity"
    assert caught.value.observed["project_version"] == "1.2.3"
    assert caught.value.observed["matching_entries"] == ()
    assert caught.value.observed["changelog_versions"] == ("1.2.2",)
    assert "changelog_versions=('1.2.2',)" in str(caught.value)


def test_duplicate_changelog_version_is_rejected_as_ambiguous(tmp_path: Path):
    pyproject, changelog = _release_files(
        tmp_path,
        changelog=("# Changelog\n\n## [1.2.3] - 2026-07-24\n## [1.2.3] - 2026-07-25\n"),
    )

    with pytest.raises(ReleaseIdentityError) as caught:
        validate_release_identity(
            "1.2.3",
            "2026-07-24",
            pyproject_path=pyproject,
            changelog_path=changelog,
        )

    assert caught.value.check == "changelog-version-identity"
    assert caught.value.observed["matching_entries"] == (
        ("2026-07-24", 3),
        ("2026-07-25", 4),
    )


def test_changelog_date_mismatch_names_release_and_observed_changelog_dates(
    tmp_path: Path,
):
    pyproject, changelog = _release_files(
        tmp_path,
        changelog="# Changelog\n\n## [1.2.3] - 2026-07-23\n",
    )

    with pytest.raises(ReleaseIdentityError) as caught:
        validate_release_identity(
            "v1.2.3",
            "2026-07-24",
            pyproject_path=pyproject,
            changelog_path=changelog,
        )

    assert caught.value.check == "changelog-date-identity"
    assert caught.value.observed == {
        "project_version": "1.2.3",
        "release_date": "2026-07-24",
        "changelog_date": "2026-07-23",
        "changelog_line": 3,
    }
    assert "release_date='2026-07-24'" in str(caught.value)
    assert "changelog_date='2026-07-23'" in str(caught.value)


@pytest.mark.parametrize("value", ["2026-7-24", "2026-02-30", "24-07-2026"])
def test_release_date_requires_exact_calendar_valid_iso_date(value: str):
    with pytest.raises(ReleaseIdentityError) as caught:
        parse_release_date(value)

    assert caught.value.check == "release-date-syntax"
    assert caught.value.observed == {"release_date": value}


def test_loaders_read_current_project_and_release_evidence():
    assert load_project_version(ROOT / "pyproject.toml") == "0.8.0"
    releases = load_changelog_releases(ROOT / "CHANGELOG.md")
    current = next(release for release in releases if release.version == "0.8.0")
    assert current.release_date == "2026-07-30"


def test_cli_reports_named_observed_values_on_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    pyproject, changelog = _release_files(tmp_path)
    common = [
        "--release-date",
        "2026-07-24",
        "--pyproject",
        str(pyproject),
        "--changelog",
        str(changelog),
    ]

    assert main(["v1.2.3", *common]) == 0
    success = capsys.readouterr()
    assert "PASS tag-syntax: observed tag='v1.2.3', version='1.2.3'" in success.out
    assert "PASS changelog-version-date-identity" in success.out
    assert success.err == ""

    assert main(["v1.2.4", *common]) == 1
    failure = capsys.readouterr()
    assert failure.out == ""
    assert "FAIL tag-project-version-identity" in failure.err
    assert "tag_version='1.2.4'" in failure.err
    assert "project_version='1.2.3'" in failure.err
