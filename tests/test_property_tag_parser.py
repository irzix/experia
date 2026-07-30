"""Property tests for exact release-tag acceptance against the project version."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.release_identity import (
    ReleaseIdentity,
    ReleaseIdentityError,
    validate_release_identity,
)

_PROJECT_VERSION = "1.2.3"
_RELEASE_DATE = "2026-07-24"
_MAJOR, _MINOR, _PATCH = _PROJECT_VERSION.split(".")

# Prefixes that exercise the single optional leading ``v`` and its near misses.
_PREFIXES = st.sampled_from(
    ["", "v", "V", "vv", "vV", " v", "v ", " ", "\t", "release-", "refs/tags/v"]
)
# Suffixes that turn an otherwise exact version into a rejected form.
_SUFFIXES = st.sampled_from(
    ["", " ", "\n", "\t", "rc1", "-alpha", "+build", ".4", ".0", "-"]
)
# Numeric segments including the exact parts and leading-zero / larger variants.
_SEGMENT = st.sampled_from(
    [_MAJOR, _MINOR, _PATCH, "0", "1", "9", "10", "00", "01", "007", "4", "123"]
)
_SEMVER_LIKE = st.tuples(_SEGMENT, _SEGMENT, _SEGMENT).map(".".join)
_VERSION_BODIES = st.one_of(
    st.just(_PROJECT_VERSION),
    _SEMVER_LIKE,
    st.sampled_from(["1.2", "1.2.3.4", "1", "", "1..3", "1.2.", ".2.3"]),
    st.text(max_size=12),
)
_COMPOSED_TAGS = st.tuples(_PREFIXES, _VERSION_BODIES, _SUFFIXES).map(
    lambda parts: "".join(parts)
)
# Arbitrary tag strings biased toward the acceptance boundary of ``v?MAJOR.MINOR.PATCH``.
_TAG_STRATEGY = st.one_of(
    st.sampled_from([_PROJECT_VERSION, f"v{_PROJECT_VERSION}"]),
    _COMPOSED_TAGS,
    st.text(max_size=24),
)


@pytest.fixture(scope="module")
def release_evidence(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Write one controlled project version and matching changelog release entry."""
    directory = tmp_path_factory.mktemp("release_identity")
    pyproject = directory / "pyproject.toml"
    pyproject.write_text(
        '[build-system]\nrequires = []\n\n[project]\nname = "sample"\n'
        f'version = "{_PROJECT_VERSION}"\n',
        encoding="utf-8",
    )
    changelog = directory / "CHANGELOG.md"
    changelog.write_text(
        f"# Changelog\n\n## [{_PROJECT_VERSION}] - {_RELEASE_DATE}\n",
        encoding="utf-8",
    )
    return pyproject, changelog


# Feature: open-source-project-improvements, Property 22: Tag parser accepts only the exact project version
@settings(max_examples=100, deadline=None)
@given(tag=_TAG_STRATEGY)
def test_tag_acceptance_is_exactly_the_project_version_with_optional_single_v(
    tag: str,
    release_evidence: tuple[Path, Path],
) -> None:
    """**Validates: Requirements 8.1**"""
    pyproject, changelog = release_evidence
    expected_accept = tag in {_PROJECT_VERSION, f"v{_PROJECT_VERSION}"}

    if expected_accept:
        identity = validate_release_identity(
            tag,
            _RELEASE_DATE,
            pyproject_path=pyproject,
            changelog_path=changelog,
        )
        assert identity == ReleaseIdentity(
            tag=tag,
            version=_PROJECT_VERSION,
            release_date=_RELEASE_DATE,
            changelog_line=3,
        )
        return

    with pytest.raises(ReleaseIdentityError):
        validate_release_identity(
            tag,
            _RELEASE_DATE,
            pyproject_path=pyproject,
            changelog_path=changelog,
        )
