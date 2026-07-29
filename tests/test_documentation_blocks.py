"""Focused tests for classified documentation block validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.documentation_blocks import (
    BlockClassification,
    DocumentationValidationError,
    parse_document,
    validate_documentation,
)


def _write(tmp_path: Path, source: str) -> Path:
    document = tmp_path / "example.md"
    document.write_text(source, encoding="utf-8")
    return document


def test_parser_requires_and_preserves_explicit_block_classification(tmp_path):
    document = _write(
        tmp_path,
        """# Classified examples

```python executable
print("installed example")
```

```json illustrative
{"status": "documented"}
```
""",
    )

    blocks = parse_document(document)

    assert [block.line for block in blocks] == [3, 7]
    assert [block.language for block in blocks] == ["python", "json"]
    assert [block.classification for block in blocks] == [
        BlockClassification.EXECUTABLE,
        BlockClassification.ILLUSTRATIVE,
    ]
    assert blocks[0].source == 'print("installed example")'


@pytest.mark.parametrize(
    ("fence", "error_code"),
    [
        ("```python\npass\n```\n", "invalid_classification"),
        ("```python example\npass\n```\n", "invalid_classification"),
        ("```python executable extra\npass\n```\n", "invalid_classification"),
        ("```python illustrative\npass\n", "unclosed_fence"),
    ],
)
def test_parser_rejects_missing_invalid_or_unclosed_classification(
    tmp_path,
    fence,
    error_code,
):
    document = _write(tmp_path, fence)

    with pytest.raises(DocumentationValidationError) as captured:
        parse_document(document)

    assert captured.value.code == error_code
    assert captured.value.path == document
    assert captured.value.line == 1


def test_illustrative_blocks_are_syntax_checked_in_their_declared_languages(
    tmp_path,
):
    document = _write(
        tmp_path,
        """```python illustrative
value: int = 3
```
```json illustrative
{"value": 3}
```
```bash illustrative
if true; then
  printf '%s\\n' ok
fi
```
```text illustrative
output has no executable semantics
```
""",
    )

    results = validate_documentation([document])

    assert len(results) == 4
    assert {result.action for result in results} == {"syntax-checked"}


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("python", 'if True print("invalid")\n'),
        ("json", '{"value": 3\n'),
        ("bash", "if true; then\n  printf '%s\\n' missing-fi\n"),
    ],
)
def test_invalid_illustrative_syntax_has_a_named_location_diagnostic(
    tmp_path,
    language,
    source,
):
    document = _write(
        tmp_path,
        f"# Broken example\n```{language} illustrative\n{source}```\n",
    )

    with pytest.raises(DocumentationValidationError) as captured:
        validate_documentation([document])

    assert captured.value.code == "invalid_syntax"
    assert captured.value.path == document
    assert captured.value.line == 2


def test_supported_executable_block_requires_installed_artifact_context(tmp_path):
    document = _write(
        tmp_path,
        """```python executable
import experia
```
""",
    )

    with pytest.raises(DocumentationValidationError) as captured:
        validate_documentation([document])

    assert captured.value.code == "missing_installed_context"


def test_executable_context_rejects_a_package_root_inside_the_source_tree(tmp_path):
    document = _write(
        tmp_path,
        """```python executable
import experia
```
""",
    )

    with pytest.raises(DocumentationValidationError) as captured:
        validate_documentation(
            [document],
            python_executable=Path(__import__("sys").executable),
            installed_package_root=tmp_path,
            source_root=tmp_path,
        )

    assert captured.value.code == "source_tree_install"


def test_unsupported_executable_language_fails_before_artifact_setup(tmp_path):
    document = _write(
        tmp_path,
        """```ruby executable
puts "unchecked"
```
""",
    )

    with pytest.raises(DocumentationValidationError) as captured:
        validate_documentation([document])

    assert captured.value.code == "unsupported_executable_language"
    assert "is not supported" in captured.value.detail


def test_unsupported_illustrative_language_fails_instead_of_being_skipped(tmp_path):
    document = _write(
        tmp_path,
        """```ruby illustrative
puts "unchecked"
```
""",
    )

    with pytest.raises(DocumentationValidationError) as captured:
        validate_documentation([document])

    assert captured.value.code == "unsupported_illustrative_language"
    assert "has no syntax checker" in captured.value.detail
