"""Machine validation for the Experia security policy and code of conduct.

This gate reads the project-wide ``SECURITY.md`` and ``CODE_OF_CONDUCT.md``
policies, extracts their machine-readable labeled fields, and rejects any
missing required field or unconfirmed placeholder value.  It never invents,
completes, or publishes contact data; its only job is to prove that whatever a
maintainer committed is present, confirmable, and non-placeholder.

Required security policy fields (Requirement 9.1):

* ``Private reporting channel`` -- a confirmable URL or email (never a placeholder).
* ``Supported release lines`` -- a support status for every released line.
* ``Acknowledgement target`` -- at most three UTC business days.
* ``Scope`` -- project-wide.

Required code of conduct fields (Requirement 9.2):

* ``Private enforcement contact`` -- a confirmable URL or email.
* ``Scope`` -- project-wide.
* ``Enforcement flow`` -- a private enforcement flow description.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from scripts.release_identity import (
    ReleaseIdentityError,
    load_project_version,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECURITY = PROJECT_ROOT / "SECURITY.md"
DEFAULT_CODE_OF_CONDUCT = PROJECT_ROOT / "CODE_OF_CONDUCT.md"
DEFAULT_PYPROJECT = PROJECT_ROOT / "pyproject.toml"

MAX_ACKNOWLEDGEMENT_BUSINESS_DAYS = 3

_FIELD_PATTERN = re.compile(
    r"^\s*[-*]\s+\*\*(?P<label>[^:*][^:]*):\*\*\s*(?P<value>.+?)\s*$"
)
_ACKNOWLEDGEMENT_PATTERN = re.compile(
    r"(?:at most|within|<=|no more than)?\s*(?P<days>\d+)\s+utc business days?\b",
    re.IGNORECASE,
)
_RELEASE_LINE_PATTERN = re.compile(r"^\d+\.\d+\.(?:x|\d+)$|^\d+\.\d+$")
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_SUPPORTED_STATUS = re.compile(r"support", re.IGNORECASE)

# Tokens that indicate an unconfirmed, invented, or fill-in value.  A required
# field containing any of these is treated as missing rather than published.
_PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\btodo\b", re.IGNORECASE),
    re.compile(r"\btbd\b", re.IGNORECASE),
    re.compile(r"\btbc\b", re.IGNORECASE),
    re.compile(r"\bfixme\b", re.IGNORECASE),
    re.compile(r"\bxxx+\b", re.IGNORECASE),
    re.compile(r"placeholder", re.IGNORECASE),
    re.compile(r"change[\s_-]?me", re.IGNORECASE),
    re.compile(r"\byour[\s_-]", re.IGNORECASE),
    re.compile(r"\binsert[\s_-]", re.IGNORECASE),
    re.compile(r"<[^>]*>"),
    re.compile(r"\bexample\.(?:com|org|net)\b", re.IGNORECASE),
    re.compile(r"@example\.", re.IGNORECASE),
    re.compile(r"@(?:domain|host|email)\b", re.IGNORECASE),
    re.compile(r"\bfoo@bar\b", re.IGNORECASE),
    re.compile(r"\bn/a\b", re.IGNORECASE),
    re.compile(r"\.\.\."),
    re.compile(r"_{3,}"),
)
# Placeholder-only domains that must never appear inside an email contact.
_PLACEHOLDER_EMAIL_DOMAINS = frozenset(
    {"example.com", "example.org", "example.net", "domain.tld", "email.com"}
)


class PolicyGateError(ValueError):
    """A named policy validation failure with explicit observed values."""

    def __init__(self, check: str, expected: str, **observed: object) -> None:
        self.check = check
        self.expected = expected
        self.observed = dict(observed)
        observed_text = ", ".join(
            f"{name}={value!r}" for name, value in self.observed.items()
        )
        super().__init__(f"FAIL {check}: expected {expected}; observed {observed_text}")


@dataclass(frozen=True)
class SecurityPolicy:
    """Validated required content of the security policy."""

    private_reporting_channel: str
    supported_lines: tuple[tuple[str, str], ...]
    acknowledgement_business_days: int
    scope: str


@dataclass(frozen=True)
class CodeOfConduct:
    """Validated required content of the code of conduct."""

    private_enforcement_contact: str
    scope: str
    enforcement_flow: str


def parse_labeled_fields(text: str) -> dict[str, str]:
    """Parse ``- **Label:** value`` lines into normalized label/value pairs."""
    fields: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _FIELD_PATTERN.fullmatch(line)
        if match is None:
            continue
        label = _normalize_label(match.group("label"))
        value = match.group("value").strip()
        if label in fields:
            raise PolicyGateError(
                "duplicate-field",
                "exactly one entry per labeled field",
                label=label,
                line=line_number,
            )
        fields[label] = value
    return fields


def require_field(fields: Mapping[str, str], label: str, *, source: str) -> str:
    """Return a present, non-empty, non-placeholder field value."""
    normalized = _normalize_label(label)
    if normalized not in fields:
        raise PolicyGateError(
            "missing-required-field",
            f"a `{label}` field in {source}",
            source=source,
            field=label,
            present_fields=tuple(sorted(fields)),
        )
    value = fields[normalized]
    if not value:
        raise PolicyGateError(
            "empty-required-field",
            f"a non-empty `{label}` field in {source}",
            source=source,
            field=label,
        )
    reject_placeholder(value, source=source, field=label)
    return value


def reject_placeholder(value: str, *, source: str, field: str) -> None:
    """Raise when a value contains any unconfirmed placeholder token."""
    for pattern in _PLACEHOLDER_PATTERNS:
        match = pattern.search(value)
        if match is not None:
            raise PolicyGateError(
                "placeholder-value",
                f"a confirmed, non-placeholder `{field}` value in {source}",
                source=source,
                field=field,
                value=value,
                placeholder=match.group(0),
            )


def ensure_confirmable_contact(value: str, *, source: str, field: str) -> str:
    """Require a confirmable ``https`` URL or real email, never a placeholder."""
    if value.startswith("https://"):
        remainder = value[len("https://") :]
        if not remainder or " " in value or "." not in remainder.split("/", 1)[0]:
            raise PolicyGateError(
                "unconfirmable-contact",
                f"a well-formed https URL for `{field}` in {source}",
                source=source,
                field=field,
                value=value,
            )
        return value
    if _EMAIL_PATTERN.fullmatch(value):
        domain = value.rsplit("@", 1)[1].lower()
        if domain in _PLACEHOLDER_EMAIL_DOMAINS:
            raise PolicyGateError(
                "unconfirmable-contact",
                f"a non-placeholder email domain for `{field}` in {source}",
                source=source,
                field=field,
                value=value,
            )
        return value
    raise PolicyGateError(
        "unconfirmable-contact",
        f"an https URL or email address for `{field}` in {source}",
        source=source,
        field=field,
        value=value,
    )


def parse_supported_lines(value: str, *, source: str) -> tuple[tuple[str, str], ...]:
    """Parse ``line: status; line: status`` release support entries."""
    entries: list[tuple[str, str]] = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise PolicyGateError(
                "malformed-supported-line",
                f"`RELEASE_LINE: STATUS` entries in {source}",
                source=source,
                entry=part,
            )
        line, status = (segment.strip() for segment in part.split(":", 1))
        if _RELEASE_LINE_PATTERN.fullmatch(line) is None:
            raise PolicyGateError(
                "malformed-supported-line",
                f"a semantic release line such as `0.7.x` in {source}",
                source=source,
                release_line=line,
            )
        if not status:
            raise PolicyGateError(
                "empty-support-status",
                f"a support status for release line `{line}` in {source}",
                source=source,
                release_line=line,
            )
        reject_placeholder(status, source=source, field=f"support status ({line})")
        entries.append((line, status))
    if not entries:
        raise PolicyGateError(
            "missing-supported-lines",
            f"at least one released-line support status in {source}",
            source=source,
            value=value,
        )
    return tuple(entries)


def parse_acknowledgement_target(value: str, *, source: str) -> int:
    """Parse and bound the acknowledgement target in UTC business days."""
    match = _ACKNOWLEDGEMENT_PATTERN.search(value)
    if match is None:
        raise PolicyGateError(
            "malformed-acknowledgement-target",
            f"an acknowledgement target measured in UTC business days in {source}",
            source=source,
            value=value,
        )
    days = int(match.group("days"))
    if days < 1 or days > MAX_ACKNOWLEDGEMENT_BUSINESS_DAYS:
        raise PolicyGateError(
            "acknowledgement-target-out-of-range",
            f"an acknowledgement target of 1 to {MAX_ACKNOWLEDGEMENT_BUSINESS_DAYS} "
            f"UTC business days in {source}",
            source=source,
            business_days=days,
        )
    return days


def require_project_wide_scope(value: str, *, source: str) -> str:
    """Require an explicit project-wide scope statement."""
    if "project-wide" not in value.lower():
        raise PolicyGateError(
            "scope-not-project-wide",
            f"an explicit project-wide scope in {source}",
            source=source,
            value=value,
        )
    return value


def validate_security_policy(
    path: Path = DEFAULT_SECURITY,
    *,
    pyproject_path: Path = DEFAULT_PYPROJECT,
) -> SecurityPolicy:
    """Validate the required, non-placeholder security policy content."""
    source = path.name
    fields = parse_labeled_fields(_read_text(path, check="security-policy-read"))

    channel = require_field(fields, "Private reporting channel", source=source)
    ensure_confirmable_contact(
        channel, source=source, field="Private reporting channel"
    )

    supported_raw = require_field(fields, "Supported release lines", source=source)
    supported_lines = parse_supported_lines(supported_raw, source=source)
    _require_current_line_supported(
        supported_lines, source=source, pyproject_path=pyproject_path
    )

    acknowledgement_raw = require_field(fields, "Acknowledgement target", source=source)
    business_days = parse_acknowledgement_target(acknowledgement_raw, source=source)

    scope = require_project_wide_scope(
        require_field(fields, "Scope", source=source), source=source
    )

    return SecurityPolicy(
        private_reporting_channel=channel,
        supported_lines=supported_lines,
        acknowledgement_business_days=business_days,
        scope=scope,
    )


def validate_code_of_conduct(path: Path = DEFAULT_CODE_OF_CONDUCT) -> CodeOfConduct:
    """Validate the required, non-placeholder code of conduct content."""
    source = path.name
    fields = parse_labeled_fields(_read_text(path, check="code-of-conduct-read"))

    contact = require_field(fields, "Private enforcement contact", source=source)
    ensure_confirmable_contact(
        contact, source=source, field="Private enforcement contact"
    )

    scope = require_project_wide_scope(
        require_field(fields, "Scope", source=source), source=source
    )
    enforcement_flow = require_field(fields, "Enforcement flow", source=source)

    return CodeOfConduct(
        private_enforcement_contact=contact,
        scope=scope,
        enforcement_flow=enforcement_flow,
    )


def validate_policies(
    *,
    security_path: Path = DEFAULT_SECURITY,
    code_of_conduct_path: Path = DEFAULT_CODE_OF_CONDUCT,
    pyproject_path: Path = DEFAULT_PYPROJECT,
) -> tuple[SecurityPolicy, CodeOfConduct]:
    """Validate both project-wide policies and return their parsed content."""
    security = validate_security_policy(security_path, pyproject_path=pyproject_path)
    code_of_conduct = validate_code_of_conduct(code_of_conduct_path)
    return security, code_of_conduct


def format_success(
    security: SecurityPolicy, code_of_conduct: CodeOfConduct
) -> tuple[str, ...]:
    """Render stable, named success evidence for automation."""
    return (
        "PASS security-policy: observed "
        f"private_channel={security.private_reporting_channel!r}, "
        f"supported_lines={len(security.supported_lines)}, "
        f"acknowledgement_business_days={security.acknowledgement_business_days}",
        "PASS code-of-conduct: observed "
        f"private_contact={code_of_conduct.private_enforcement_contact!r}, "
        f"scope={code_of_conduct.scope!r}",
    )


def _require_current_line_supported(
    supported_lines: Sequence[tuple[str, str]],
    *,
    source: str,
    pyproject_path: Path,
) -> None:
    try:
        version = load_project_version(pyproject_path)
    except ReleaseIdentityError as error:
        raise PolicyGateError(
            "project-version-read",
            "a readable project version for released-line validation",
            pyproject=str(pyproject_path),
            error=str(error),
        ) from error
    major, minor, _ = version.split(".", 2)
    current_line = f"{major}.{minor}.x"
    status_by_line = {line: status for line, status in supported_lines}
    status = status_by_line.get(current_line)
    if status is None:
        raise PolicyGateError(
            "current-line-missing",
            f"support status for the current release line `{current_line}` in {source}",
            source=source,
            current_line=current_line,
            documented_lines=tuple(status_by_line),
        )
    if _SUPPORTED_STATUS.search(status) is None:
        raise PolicyGateError(
            "current-line-unsupported",
            f"the current release line `{current_line}` marked supported in {source}",
            source=source,
            current_line=current_line,
            status=status,
        )


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip().lower()


def _read_text(path: Path, *, check: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise PolicyGateError(
            check,
            "a readable UTF-8 policy document",
            path=str(path),
            error_type=type(error).__name__,
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--security", type=Path, default=DEFAULT_SECURITY)
    parser.add_argument("--code-of-conduct", type=Path, default=DEFAULT_CODE_OF_CONDUCT)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        security, code_of_conduct = validate_policies(
            security_path=arguments.security,
            code_of_conduct_path=arguments.code_of_conduct,
            pyproject_path=arguments.pyproject,
        )
    except PolicyGateError as error:
        print(error, file=sys.stderr)
        return 1
    for message in format_success(security, code_of_conduct):
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
