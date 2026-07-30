import os
import re
import subprocess
import sys
from pathlib import Path

from scripts.documentation_blocks import validate_documentation
from scripts.generate_api_snapshot import build_snapshot, render_constructor_contracts
from scripts.outbound_data_contract import (
    assert_api_reference_synced,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
API_REFERENCE_PATH = ROOT / "API_REFERENCE.md"
QUICKSTART_PATH = ROOT / "examples" / "quickstart.py"

# Fence info strings that carry an explicit executable/illustrative classification
# understood by the Quality Gate documentation-block classifier.
_CLASSIFIED_BLOCK = re.compile(r"^(bash|json|python|text) (executable|illustrative)$")
# Rendered-diagram languages are visualizations validated by Markdown rendering,
# not code executed or syntax-checked by the classifier, so they are exempt from
# the executable/illustrative classification contract.
_RENDERED_DIAGRAM_LANGUAGES = frozenset({"mermaid"})


def _between_markers(text: str, start: str, end: str) -> str:
    assert text.count(start) == 1
    assert text.count(end) == 1
    return text.split(start, 1)[1].split(end, 1)[0]


def _fenced_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``(info_string, body)`` for every top-level fenced code block."""
    blocks: list[tuple[str, str]] = []
    info: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if info is None:
            if line.startswith("```"):
                info = line[3:].strip()
                body = []
        elif line.strip() == "```":
            blocks.append((info, "\n".join(body)))
            info = None
        else:
            body.append(line)
    assert info is None, "README has an unterminated fenced code block"
    return blocks


def test_api_reference_constructor_contracts_match_canonical_snapshot():
    reference = API_REFERENCE_PATH.read_text(encoding="utf-8")
    documented = (
        "<!-- BEGIN GENERATED CONSTRUCTOR CONTRACTS -->"
        + _between_markers(
            reference,
            "<!-- BEGIN GENERATED CONSTRUCTOR CONTRACTS -->",
            "<!-- END GENERATED CONSTRUCTOR CONTRACTS -->",
        )
        + "<!-- END GENERATED CONSTRUCTOR CONTRACTS -->"
    )

    assert documented == render_constructor_contracts(build_snapshot())
    assert "`evaluator` | `positional_or_keyword`" in documented
    assert "`experia.core.interfaces.Evaluator` | yes | `—`" in documented


def test_api_reference_consumes_the_validated_outbound_data_contract():
    contract = load_contract()

    assert_api_reference_synced(contract, API_REFERENCE_PATH)
    assert contract["default_without_sanitizer"] == "pass-through"


def test_readme_quickstart_is_the_checked_example_source():
    readme = README_PATH.read_text(encoding="utf-8")
    fenced = _between_markers(
        readme,
        "<!-- BEGIN EXECUTABLE QUICKSTART -->",
        "<!-- END EXECUTABLE QUICKSTART -->",
    ).strip()
    opening = "```python executable\n"
    assert fenced.startswith(opening)
    assert fenced.endswith("\n```")

    documented_source = fenced[len(opening) : -len("\n```")]
    assert documented_source == QUICKSTART_PATH.read_text(encoding="utf-8").rstrip()


def test_offline_quickstart_runs_outside_the_source_tree(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(QUICKSTART_PATH)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_planned_features_are_unavailable_and_absent_from_quickstart():
    readme = README_PATH.read_text(encoding="utf-8")
    quickstart = QUICKSTART_PATH.read_text(encoding="utf-8").casefold()

    assert "Planned (unavailable in the current version)" in readme
    assert "UnavailableFeatureError" in readme
    for planned_name in ("postgres", "pgvector", "mem0", "zep", "crewai", "redis"):
        assert planned_name not in quickstart


def test_each_implemented_status_entry_links_executable_evidence():
    readme = README_PATH.read_text(encoding="utf-8")
    implemented = readme.split("**Implemented today**", 1)[1].split(
        "**Planned (unavailable in the current version)**", 1
    )[0]
    rows = [line for line in implemented.splitlines() if line.startswith("| ")][1:]

    assert len(rows) == 5
    assert all("](" in row and "tests/" in row for row in rows)


def test_every_readme_code_block_is_explicitly_classified():
    blocks = _fenced_code_blocks(README_PATH.read_text(encoding="utf-8"))
    assert blocks, "expected README to contain fenced code blocks"

    for info, _ in blocks:
        language = info.split()[0] if info else ""
        if language in _RENDERED_DIAGRAM_LANGUAGES:
            assert info == language, f"rendered diagram fence must stay bare: {info!r}"
            continue
        assert _CLASSIFIED_BLOCK.fullmatch(info), (
            f"code block {info!r} is not classified as executable or illustrative"
        )


def test_readme_illustrative_blocks_pass_the_classifier_syntax_check(tmp_path):
    illustrative = [
        (info, body)
        for info, body in _fenced_code_blocks(README_PATH.read_text(encoding="utf-8"))
        if info.endswith(" illustrative")
    ]
    assert illustrative, "expected classified illustrative blocks in README"

    document = tmp_path / "readme-illustrative.md"
    document.write_text(
        "\n\n".join(f"```{info}\n{body}\n```" for info, body in illustrative) + "\n",
        encoding="utf-8",
    )

    results = validate_documentation([document])

    assert len(results) == len(illustrative)
    assert {result.action for result in results} == {"syntax-checked"}
