"""Validate Experia's structured bug-report and pull-request templates.

Requirement 9.3 requires the bug report to request separately labeled fields
for the Experia version, Python version, operating system, reproduction steps,
expected behavior, and actual behavior. Requirement 9.4 requires the pull
request template to request separately labeled fields for tests, documentation,
changelog, backward compatibility, and security impact. This module parses the
committed templates and asserts every required labeled field is present so the
contributor experience cannot silently regress.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUG_REPORT = ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
DEFAULT_PULL_REQUEST = ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"

# Form element types that carry a contributor-visible label.
_LABELED_ELEMENT_TYPES = frozenset({"input", "textarea", "dropdown", "checkboxes"})

# Each required bug field maps a stable key to the accepted normalized labels.
REQUIRED_BUG_FIELDS: dict[str, tuple[str, ...]] = {
    "experia_version": ("experia version",),
    "python_version": ("python version",),
    "operating_system": ("operating system", "os"),
    "reproduction_steps": ("reproduction steps", "steps to reproduce"),
    "expected_behavior": ("expected behavior", "expected behaviour"),
    "actual_behavior": ("actual behavior", "actual behaviour"),
}

# Each required pull-request section maps a stable key to accepted headings.
REQUIRED_PULL_REQUEST_SECTIONS: dict[str, tuple[str, ...]] = {
    "tests": ("tests", "testing"),
    "documentation": ("documentation", "docs"),
    "changelog": ("changelog",),
    "backward_compatibility": (
        "backward compatibility",
        "backwards compatibility",
        "compatibility",
    ),
    "security_impact": ("security impact", "security"),
}

_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*#*\s*$")


class ContributorTemplateError(ValueError):
    """Raised when a contributor template is missing a required labeled field."""


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace/punctuation for tolerant matching."""

    return re.sub(r"[\s_/-]+", " ", str(text).casefold()).strip(" .:*#")


def load_issue_form(path: Path = DEFAULT_BUG_REPORT) -> Mapping[str, Any]:
    """Load and shallowly validate a GitHub issue-form YAML document."""

    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ContributorTemplateError(f"Unable to load {path}: {error}") from error
    if not isinstance(parsed, Mapping):
        raise ContributorTemplateError(f"{path}: expected a YAML mapping")
    for key in ("name", "description", "body"):
        if key not in parsed:
            raise ContributorTemplateError(f"{path}: missing required key {key!r}")
    if not isinstance(parsed["body"], Sequence) or isinstance(parsed["body"], str):
        raise ContributorTemplateError(f"{path}: 'body' must be a list of elements")
    return parsed


def labeled_required_fields(form: Mapping[str, Any]) -> dict[str, bool]:
    """Return each labeled form field's normalized label and required status."""

    fields: dict[str, bool] = {}
    for index, element in enumerate(form["body"]):
        if not isinstance(element, Mapping):
            raise ContributorTemplateError(f"body[{index}]: expected a mapping")
        if element.get("type") not in _LABELED_ELEMENT_TYPES:
            continue
        attributes = element.get("attributes")
        if not isinstance(attributes, Mapping) or "label" not in attributes:
            raise ContributorTemplateError(
                f"body[{index}]: labeled element is missing 'attributes.label'"
            )
        label = _normalize(attributes["label"])
        if not label:
            raise ContributorTemplateError(f"body[{index}]: label must be non-empty")
        validations = element.get("validations")
        required = bool(
            isinstance(validations, Mapping) and validations.get("required") is True
        )
        # Separate elements must not collide on the same label.
        if label in fields:
            raise ContributorTemplateError(
                f"body[{index}]: duplicate labeled field {label!r}"
            )
        fields[label] = required
    return fields


def validate_bug_report(form: Mapping[str, Any]) -> None:
    """Assert every Requirement 9.3 field is separately labeled and required."""

    fields = labeled_required_fields(form)
    for key, accepted in REQUIRED_BUG_FIELDS.items():
        match = next((label for label in accepted if label in fields), None)
        if match is None:
            raise ContributorTemplateError(
                f"bug report is missing a separately labeled field for {key!r} "
                f"(expected one of {list(accepted)!r})"
            )
        if not fields[match]:
            raise ContributorTemplateError(
                f"bug report field {match!r} for {key!r} must be marked required"
            )


def pull_request_headings(text: str) -> tuple[str, ...]:
    """Return the normalized Markdown headings from a pull-request template."""

    headings: list[str] = []
    for line in text.splitlines():
        match = _HEADING_PATTERN.match(line)
        if match is not None:
            headings.append(_normalize(match.group("title")))
    return tuple(headings)


def validate_pull_request_template(text: str) -> None:
    """Assert every Requirement 9.4 section has a separately labeled heading."""

    headings = pull_request_headings(text)
    counts: dict[str, int] = {}
    for heading in headings:
        counts[heading] = counts.get(heading, 0) + 1
    for key, accepted in REQUIRED_PULL_REQUEST_SECTIONS.items():
        matches = [heading for heading in accepted if heading in headings]
        if not matches:
            raise ContributorTemplateError(
                f"pull-request template is missing a labeled section for {key!r} "
                f"(expected one of {list(accepted)!r})"
            )
        if any(counts[heading] > 1 for heading in matches):
            raise ContributorTemplateError(
                f"pull-request template repeats the section for {key!r}"
            )


def validate_templates(
    *,
    bug_report_path: Path = DEFAULT_BUG_REPORT,
    pull_request_path: Path = DEFAULT_PULL_REQUEST,
) -> None:
    """Validate both contributor templates against their requirements."""

    validate_bug_report(load_issue_form(bug_report_path))
    try:
        pull_request_text = pull_request_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ContributorTemplateError(
            f"Unable to load {pull_request_path}: {error}"
        ) from error
    validate_pull_request_template(pull_request_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bug-report", type=Path, default=DEFAULT_BUG_REPORT)
    parser.add_argument("--pull-request", type=Path, default=DEFAULT_PULL_REQUEST)
    arguments = parser.parse_args()

    validate_templates(
        bug_report_path=arguments.bug_report,
        pull_request_path=arguments.pull_request,
    )
    print(
        f"Validated {len(REQUIRED_BUG_FIELDS)} bug-report fields and "
        f"{len(REQUIRED_PULL_REQUEST_SECTIONS)} pull-request sections."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REQUIRED_BUG_FIELDS",
    "REQUIRED_PULL_REQUEST_SECTIONS",
    "ContributorTemplateError",
    "DEFAULT_BUG_REPORT",
    "DEFAULT_PULL_REQUEST",
    "labeled_required_fields",
    "load_issue_form",
    "pull_request_headings",
    "validate_bug_report",
    "validate_pull_request_template",
    "validate_templates",
]
