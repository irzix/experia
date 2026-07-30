"""Credential-free release tag, project version, and changelog validation."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYPROJECT = PROJECT_ROOT / "pyproject.toml"
DEFAULT_CHANGELOG = PROJECT_ROOT / "CHANGELOG.md"

_SEMANTIC_VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
_BARE_VERSION_PATTERN = re.compile(rf"(?P<version>{_SEMANTIC_VERSION})", re.ASCII)
_TAG_PATTERN = re.compile(rf"v?(?P<version>{_SEMANTIC_VERSION})", re.ASCII)
_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", re.ASCII)
_TABLE_PATTERN = re.compile(r"^\s*\[([^\[\]\r\n]+)\]\s*(?:#.*)?$")
_VERSION_ASSIGNMENT_PATTERN = re.compile(
    r"^\s*version\s*=\s*(?P<quote>[\"'])(?P<value>[^\"'\r\n]*)(?P=quote)"
    r"\s*(?:#.*)?$"
)
_CHANGELOG_RELEASE_PATTERN = re.compile(
    r"^## \[(?P<version>[^\]\r\n]+)\] - (?P<date>[^\r\n]+)$"
)


class ReleaseIdentityError(ValueError):
    """A named release validation failure with explicit observed values."""

    def __init__(self, check: str, expected: str, **observed: object) -> None:
        self.check = check
        self.expected = expected
        self.observed = dict(observed)
        observed_text = ", ".join(
            f"{name}={value!r}" for name, value in self.observed.items()
        )
        super().__init__(f"FAIL {check}: expected {expected}; observed {observed_text}")


@dataclass(frozen=True)
class ChangelogRelease:
    """One exact Keep a Changelog release heading."""

    version: str
    release_date: str
    line_number: int


@dataclass(frozen=True)
class ReleaseIdentity:
    """Validated source evidence for one release."""

    tag: str
    version: str
    release_date: str
    changelog_line: int


def parse_release_tag(tag: str) -> str:
    """Return the bare version from an exact ``v?MAJOR.MINOR.PATCH`` tag."""
    if not isinstance(tag, str) or (match := _TAG_PATTERN.fullmatch(tag)) is None:
        raise ReleaseIdentityError(
            "tag-syntax",
            "tag matching v?MAJOR.MINOR.PATCH with no leading zeroes",
            tag=tag,
        )
    return match.group("version")


def parse_release_date(value: str | date) -> str:
    """Return an exact, calendar-valid ISO release date."""
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or _DATE_PATTERN.fullmatch(value) is None:
        raise ReleaseIdentityError(
            "release-date-syntax",
            "release date in YYYY-MM-DD format",
            release_date=value,
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ReleaseIdentityError(
            "release-date-syntax",
            "calendar-valid release date in YYYY-MM-DD format",
            release_date=value,
        ) from error
    if parsed.isoformat() != value:
        raise ReleaseIdentityError(
            "release-date-syntax",
            "release date in canonical YYYY-MM-DD format",
            release_date=value,
        )
    return value


def load_project_version(path: Path) -> str:
    """Read the static version declared directly in ``[project]``."""
    text = _read_text(path, check="project-version-read")
    in_project_table = False
    assignments: list[str] = []
    values: list[str] = []

    for line in text.splitlines():
        if table := _TABLE_PATTERN.fullmatch(line):
            in_project_table = table.group(1).strip() == "project"
            continue
        if not in_project_table or re.match(r"^\s*version\s*=", line) is None:
            continue
        assignments.append(line.strip())
        assignment = _VERSION_ASSIGNMENT_PATTERN.fullmatch(line)
        if assignment is not None:
            values.append(assignment.group("value"))

    if len(assignments) != 1 or len(values) != 1:
        raise ReleaseIdentityError(
            "project-version-declaration",
            "exactly one quoted version assignment in [project]",
            path=str(path),
            assignments=tuple(assignments),
        )

    version = values[0]
    if _BARE_VERSION_PATTERN.fullmatch(version) is None:
        raise ReleaseIdentityError(
            "project-version-syntax",
            "project version matching MAJOR.MINOR.PATCH with no leading zeroes",
            project_version=version,
            path=str(path),
        )
    return version


def load_changelog_releases(path: Path) -> tuple[ChangelogRelease, ...]:
    """Read exact ``## [VERSION] - DATE`` release headings from a changelog."""
    text = _read_text(path, check="changelog-read")
    releases: list[ChangelogRelease] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _CHANGELOG_RELEASE_PATTERN.fullmatch(line)
        if match is not None:
            releases.append(
                ChangelogRelease(
                    version=match.group("version"),
                    release_date=match.group("date"),
                    line_number=line_number,
                )
            )
    return tuple(releases)


def validate_release_identity(
    tag: str,
    release_date: str | date,
    *,
    pyproject_path: Path = DEFAULT_PYPROJECT,
    changelog_path: Path = DEFAULT_CHANGELOG,
) -> ReleaseIdentity:
    """Validate exact tag, project version, and changelog release identity."""
    tag_version = parse_release_tag(tag)
    expected_date = parse_release_date(release_date)
    project_version = load_project_version(pyproject_path)
    if tag_version != project_version:
        raise ReleaseIdentityError(
            "tag-project-version-identity",
            "tag version exactly equal to project version",
            tag_version=tag_version,
            project_version=project_version,
        )

    releases = load_changelog_releases(changelog_path)
    matching = tuple(release for release in releases if release.version == tag_version)
    if len(matching) != 1:
        raise ReleaseIdentityError(
            "changelog-version-identity",
            "exactly one changelog release entry for the project version",
            project_version=project_version,
            matching_entries=tuple(
                (release.release_date, release.line_number) for release in matching
            ),
            changelog_versions=tuple(release.version for release in releases),
        )

    changelog_release = matching[0]
    if changelog_release.release_date != expected_date:
        raise ReleaseIdentityError(
            "changelog-date-identity",
            "changelog date exactly equal to release date",
            project_version=project_version,
            release_date=expected_date,
            changelog_date=changelog_release.release_date,
            changelog_line=changelog_release.line_number,
        )

    return ReleaseIdentity(
        tag=tag,
        version=project_version,
        release_date=expected_date,
        changelog_line=changelog_release.line_number,
    )


def format_success(identity: ReleaseIdentity) -> tuple[str, ...]:
    """Render stable, named success evidence for release automation."""
    return (
        f"PASS tag-syntax: observed tag={identity.tag!r}, version={identity.version!r}",
        "PASS tag-project-version-identity: "
        f"observed tag_version={identity.version!r}, "
        f"project_version={identity.version!r}",
        "PASS changelog-version-date-identity: "
        f"observed version={identity.version!r}, date={identity.release_date!r}, "
        f"line={identity.changelog_line}",
    )


def _read_text(path: Path, *, check: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseIdentityError(
            check,
            "readable UTF-8 release evidence",
            path=str(path),
            error_type=type(error).__name__,
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="release tag to validate")
    parser.add_argument(
        "--release-date",
        required=True,
        help="expected UTC release date in YYYY-MM-DD format",
    )
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    parser.add_argument("--changelog", type=Path, default=DEFAULT_CHANGELOG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        identity = validate_release_identity(
            arguments.tag,
            arguments.release_date,
            pyproject_path=arguments.pyproject,
            changelog_path=arguments.changelog,
        )
    except ReleaseIdentityError as error:
        print(error, file=sys.stderr)
        return 1
    for message in format_success(identity):
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
