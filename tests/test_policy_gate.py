"""Focused tests for machine-validated security and conduct policies."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.policy_gate import (
    CodeOfConduct,
    PolicyGateError,
    SecurityPolicy,
    ensure_confirmable_contact,
    main,
    parse_acknowledgement_target,
    parse_labeled_fields,
    parse_supported_lines,
    reject_placeholder,
    validate_code_of_conduct,
    validate_policies,
    validate_security_policy,
)

ROOT = Path(__file__).resolve().parents[1]

_CHANNEL = "https://github.com/irzix/experia/security/advisories/new"

_VALID_SECURITY = "\n".join(
    [
        "# Security Policy",
        "",
        f"- **Private reporting channel:** {_CHANNEL}",
        "- **Supported release lines:** 0.7.x: supported; 0.1.x: end-of-life",
        "- **Acknowledgement target:** at most 3 UTC business days",
        "- **Scope:** project-wide",
        "",
    ]
)

_VALID_CODE_OF_CONDUCT = "\n".join(
    [
        "# Code of Conduct",
        "",
        f"- **Private enforcement contact:** {_CHANNEL}",
        "- **Scope:** project-wide",
        "- **Enforcement flow:** Reports are received privately by the maintainers.",
        "",
    ]
)


def _pyproject(tmp_path: Path, *, version: str = "0.7.0") -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        f'[project]\nname = "experia"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    return path


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- Repository policies validate ------------------------------------------


def test_repository_policies_validate_without_placeholders() -> None:
    security, code_of_conduct = validate_policies()

    assert isinstance(security, SecurityPolicy)
    assert isinstance(code_of_conduct, CodeOfConduct)
    assert security.private_reporting_channel == _CHANNEL
    assert security.acknowledgement_business_days == 3
    assert ("0.7.x", "supported") in security.supported_lines
    assert code_of_conduct.private_enforcement_contact == _CHANNEL
    assert "project-wide" in code_of_conduct.scope.lower()


def test_valid_generated_policies_round_trip(tmp_path: Path) -> None:
    security_path = _write(tmp_path, "SECURITY.md", _VALID_SECURITY)
    coc_path = _write(tmp_path, "CODE_OF_CONDUCT.md", _VALID_CODE_OF_CONDUCT)
    pyproject = _pyproject(tmp_path)

    security, code_of_conduct = validate_policies(
        security_path=security_path,
        code_of_conduct_path=coc_path,
        pyproject_path=pyproject,
    )

    assert security.acknowledgement_business_days == 3
    assert code_of_conduct.enforcement_flow.startswith("Reports are received")


# --- Field parsing ----------------------------------------------------------


def test_parse_labeled_fields_normalizes_labels() -> None:
    fields = parse_labeled_fields("- **Private Reporting Channel:**   value here  ")
    assert fields == {"private reporting channel": "value here"}


def test_duplicate_field_is_rejected() -> None:
    text = "- **Scope:** project-wide\n- **Scope:** project-wide\n"
    with pytest.raises(PolicyGateError) as caught:
        parse_labeled_fields(text)
    assert caught.value.check == "duplicate-field"


# --- Missing required fields ------------------------------------------------


def test_missing_required_security_field_is_named(tmp_path: Path) -> None:
    text = _VALID_SECURITY.replace(
        "- **Acknowledgement target:** at most 3 UTC business days\n", ""
    )
    security_path = _write(tmp_path, "SECURITY.md", text)
    with pytest.raises(PolicyGateError) as caught:
        validate_security_policy(security_path, pyproject_path=_pyproject(tmp_path))
    assert caught.value.check == "missing-required-field"
    assert caught.value.observed["field"] == "Acknowledgement target"


def test_missing_enforcement_flow_is_named(tmp_path: Path) -> None:
    text = _VALID_CODE_OF_CONDUCT.replace(
        "- **Enforcement flow:** Reports are received privately by the maintainers.\n",
        "",
    )
    coc_path = _write(tmp_path, "CODE_OF_CONDUCT.md", text)
    with pytest.raises(PolicyGateError) as caught:
        validate_code_of_conduct(coc_path)
    assert caught.value.check == "missing-required-field"
    assert caught.value.observed["field"] == "Enforcement flow"


# --- Placeholder rejection --------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "TODO",
        "TBD",
        "FIXME",
        "placeholder contact",
        "<your-email-here>",
        "security@example.com",
        "your email address",
        "reach us at ...",
        "change me",
        "contact@domain",
    ],
)
def test_reject_placeholder_flags_unconfirmed_values(value: str) -> None:
    with pytest.raises(PolicyGateError) as caught:
        reject_placeholder(
            value, source="SECURITY.md", field="Private reporting channel"
        )
    assert caught.value.check == "placeholder-value"


def test_placeholder_channel_fails_full_validation(tmp_path: Path) -> None:
    text = _VALID_SECURITY.replace(_CHANNEL, "TODO: confirm with maintainer")
    security_path = _write(tmp_path, "SECURITY.md", text)
    with pytest.raises(PolicyGateError) as caught:
        validate_security_policy(security_path, pyproject_path=_pyproject(tmp_path))
    assert caught.value.check == "placeholder-value"


# --- Confirmable contact ----------------------------------------------------


def test_confirmable_contact_accepts_url_and_real_email() -> None:
    assert ensure_confirmable_contact(_CHANNEL, source="s", field="f") == _CHANNEL
    email = "security@irzix.dev"
    assert ensure_confirmable_contact(email, source="s", field="f") == email


@pytest.mark.parametrize(
    "value",
    [
        "call the maintainers",
        "http://insecure.example",  # not https
        "https://",  # no domain
        "https://host with space",
        "maintainer@example.com",  # placeholder domain
    ],
)
def test_confirmable_contact_rejects_non_contactable_values(value: str) -> None:
    with pytest.raises(PolicyGateError):
        ensure_confirmable_contact(value, source="s", field="f")


# --- Acknowledgement target -------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("at most 3 UTC business days", 3),
        ("within 2 UTC business days", 2),
        ("1 UTC business day", 1),
    ],
)
def test_acknowledgement_target_parses_bounded_days(value: str, expected: int) -> None:
    assert parse_acknowledgement_target(value, source="SECURITY.md") == expected


@pytest.mark.parametrize("days", [4, 5, 10])
def test_acknowledgement_target_over_three_days_is_rejected(days: int) -> None:
    with pytest.raises(PolicyGateError) as caught:
        parse_acknowledgement_target(f"{days} UTC business days", source="SECURITY.md")
    assert caught.value.check == "acknowledgement-target-out-of-range"


def test_acknowledgement_target_without_utc_business_days_is_rejected() -> None:
    with pytest.raises(PolicyGateError) as caught:
        parse_acknowledgement_target("3 days", source="SECURITY.md")
    assert caught.value.check == "malformed-acknowledgement-target"


# --- Supported release lines ------------------------------------------------


def test_supported_lines_parse_line_and_status_pairs() -> None:
    entries = parse_supported_lines(
        "0.7.x: supported; 0.2.x: end-of-life", source="SECURITY.md"
    )
    assert entries == (("0.7.x", "supported"), ("0.2.x", "end-of-life"))


def test_supported_lines_reject_malformed_line() -> None:
    with pytest.raises(PolicyGateError) as caught:
        parse_supported_lines("latest: supported", source="SECURITY.md")
    assert caught.value.check == "malformed-supported-line"


def test_current_release_line_must_be_present(tmp_path: Path) -> None:
    text = _VALID_SECURITY.replace("0.7.x: supported; ", "")
    security_path = _write(tmp_path, "SECURITY.md", text)
    with pytest.raises(PolicyGateError) as caught:
        validate_security_policy(security_path, pyproject_path=_pyproject(tmp_path))
    assert caught.value.check == "current-line-missing"
    assert caught.value.observed["current_line"] == "0.7.x"


def test_current_release_line_must_be_supported(tmp_path: Path) -> None:
    text = _VALID_SECURITY.replace("0.7.x: supported", "0.7.x: end-of-life")
    security_path = _write(tmp_path, "SECURITY.md", text)
    with pytest.raises(PolicyGateError) as caught:
        validate_security_policy(security_path, pyproject_path=_pyproject(tmp_path))
    assert caught.value.check == "current-line-unsupported"


# --- Scope ------------------------------------------------------------------


def test_scope_must_be_project_wide(tmp_path: Path) -> None:
    text = _VALID_CODE_OF_CONDUCT.replace(
        "- **Scope:** project-wide", "- **Scope:** core only"
    )
    coc_path = _write(tmp_path, "CODE_OF_CONDUCT.md", text)
    with pytest.raises(PolicyGateError) as caught:
        validate_code_of_conduct(coc_path)
    assert caught.value.check == "scope-not-project-wide"


# --- CLI --------------------------------------------------------------------


def test_cli_reports_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    security_path = _write(tmp_path, "SECURITY.md", _VALID_SECURITY)
    coc_path = _write(tmp_path, "CODE_OF_CONDUCT.md", _VALID_CODE_OF_CONDUCT)
    pyproject = _pyproject(tmp_path)
    argv = [
        "--security",
        str(security_path),
        "--code-of-conduct",
        str(coc_path),
        "--pyproject",
        str(pyproject),
    ]

    assert main(argv) == 0
    success = capsys.readouterr()
    assert "PASS security-policy" in success.out
    assert "PASS code-of-conduct" in success.out
    assert success.err == ""

    broken = _write(
        tmp_path,
        "SECURITY.md",
        _VALID_SECURITY.replace(_CHANNEL, "TODO"),
    )
    argv[1] = str(broken)
    assert main(argv) == 1
    failure = capsys.readouterr()
    assert failure.out == ""
    assert "FAIL placeholder-value" in failure.err
