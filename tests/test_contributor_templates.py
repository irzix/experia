"""Tests for the structured bug-report and pull-request templates.

Validates Requirements 9.3 and 9.4.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.contributor_templates import (
    DEFAULT_PULL_REQUEST,
    REQUIRED_BUG_FIELDS,
    REQUIRED_PULL_REQUEST_SECTIONS,
    ContributorTemplateError,
    labeled_required_fields,
    load_issue_form,
    pull_request_headings,
    validate_bug_report,
    validate_pull_request_template,
    validate_templates,
)


def test_committed_templates_validate() -> None:
    validate_templates()


def test_bug_report_requests_every_required_field_as_required() -> None:
    form = load_issue_form()
    fields = labeled_required_fields(form)

    for accepted in REQUIRED_BUG_FIELDS.values():
        match = next((label for label in accepted if label in fields), None)
        assert match is not None, f"missing labeled field for {accepted!r}"
        assert fields[match] is True, f"field {match!r} must be required"


def test_pull_request_template_requests_every_required_section() -> None:
    text = DEFAULT_PULL_REQUEST.read_text(encoding="utf-8")
    headings = pull_request_headings(text)

    for accepted in REQUIRED_PULL_REQUEST_SECTIONS.values():
        assert any(heading in headings for heading in accepted), (
            f"missing labeled section for {accepted!r}"
        )


@pytest.mark.parametrize("field_key", sorted(REQUIRED_BUG_FIELDS))
def test_bug_report_rejects_missing_required_field(field_key: str) -> None:
    form = deepcopy(load_issue_form())
    accepted = {label for label in REQUIRED_BUG_FIELDS[field_key]}
    remaining = []
    for element in form["body"]:
        attributes = element.get("attributes") if isinstance(element, dict) else None
        label = attributes.get("label") if isinstance(attributes, dict) else None
        normalized = None
        if isinstance(label, str):
            normalized = label.strip().casefold().replace("_", " ").replace("-", " ")
        if normalized in accepted:
            continue
        remaining.append(element)
    form["body"] = remaining

    with pytest.raises(ContributorTemplateError, match=field_key):
        validate_bug_report(form)


def test_bug_report_rejects_optional_required_field() -> None:
    form = deepcopy(load_issue_form())
    for element in form["body"]:
        attributes = element.get("attributes") if isinstance(element, dict) else None
        label = attributes.get("label") if isinstance(attributes, dict) else None
        if isinstance(label, str) and label.strip().casefold() == "python version":
            element["validations"] = {"required": False}

    with pytest.raises(ContributorTemplateError, match="must be marked required"):
        validate_bug_report(form)


@pytest.mark.parametrize("section_key", sorted(REQUIRED_PULL_REQUEST_SECTIONS))
def test_pull_request_rejects_missing_section(section_key: str) -> None:
    accepted = {heading for heading in REQUIRED_PULL_REQUEST_SECTIONS[section_key]}
    text = DEFAULT_PULL_REQUEST.read_text(encoding="utf-8")
    kept_lines = []
    for line in text.splitlines():
        stripped = line.lstrip("#").strip().casefold()
        if line.lstrip().startswith("#") and stripped in accepted:
            continue
        kept_lines.append(line)

    with pytest.raises(ContributorTemplateError, match=section_key):
        validate_pull_request_template("\n".join(kept_lines))
