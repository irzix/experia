"""Installed-wheel documentation tests runnable on every supported Python.

Task 15.4 consolidates the executable-documentation contract against a freshly
built and installed wheel:

* documented quickstarts and executable code blocks execute against the
  installed wheel from a working directory outside the source tree,
* illustrative blocks are syntax-checked in their declared language through the
  shared classifier, and
* the generated lifecycle, typed-failure, outbound-data, and stability
  reference sections are validated against their machine-readable sources.

The suite runs against whatever interpreter executes it (``sys.executable``).
The CI matrix supplies Python 3.10/3.11/3.12, so per-version coverage comes
from the matrix rather than this module ever launching an interpreter that is
not installed.

Validates: Requirements 4.2, 9.6, 10.1-10.10
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.api_compatibility import MINOR_DEPRECATION_RELEASES
from scripts.documentation_blocks import validate_documentation
from scripts.reference_sections import (
    DEFAULT_API_SNAPSHOT,
    DEFAULT_CONTRACT,
    DEFAULT_OUTBOUND_CONTRACT,
    DEFAULT_SCHEMA_SUPPORT,
    assert_api_reference_synced,
    load_json,
    render_reference_sections,
)

try:  # Python 3.11+ ships tomllib in the standard library.
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.10 uses the pinned dev-only tomli backport.
    import tomli as _toml

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
API_REFERENCE_PATH = ROOT / "API_REFERENCE.md"
PYPROJECT_PATH = ROOT / "pyproject.toml"
API_GATE = ROOT / "scripts" / "api_gate.py"
BASELINE_PATH = ROOT / "api-snapshot.json"
MANIFEST_PATH = ROOT / "examples" / "installed-examples.json"

# The documented Supported_Python_Matrix the documentation suite targets.
DECLARED_SUPPORTED_PYTHONS = ((3, 10), (3, 11), (3, 12))

# Rendered-diagram fences (e.g. mermaid) are validated by Markdown rendering and
# are exempt from the executable/illustrative classifier contract.
_RENDERED_DIAGRAM_LANGUAGES = frozenset({"mermaid"})
_CLASSIFIED_INFO = re.compile(r"^(bash|json|python|text) (executable|illustrative)$")

_REQUIRED_LIFECYCLE_METHODS = (
    "experia.memory.store.SQLiteStore.initialize",
    "experia.core.learner.Learner.flush",
    "experia.core.learner.Learner.shutdown",
    "experia.memory.store.SQLiteStore.close",
)


@dataclass(frozen=True)
class InstalledWheel:
    """A wheel built and installed outside the source tree for this interpreter."""

    python: Path
    site_packages: Path
    workspace: Path


def _environment_without_source_path() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def _run(
    command: list[str | Path],
    *,
    cwd: Path,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=_environment_without_source_path(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Command failed: {' '.join(str(part) for part in command)}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return completed


def _installed_script_command(
    wheel: InstalledWheel,
    script: Path,
    *arguments: str | Path,
) -> list[str | Path]:
    """Run ``script`` in the current interpreter with the installed wheel first."""
    argv = [str(script), *(str(argument) for argument in arguments)]
    code = (
        "import runpy; "
        f"sys.argv = {argv!r}; "
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    bootstrap = f"import sys; sys.path.insert(0, {str(wheel.site_packages)!r}); {code}"
    return [wheel.python, "-I", "-c", bootstrap]


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
    assert info is None, "document has an unterminated fenced code block"
    return blocks


def _classified_blocks(text: str) -> list[tuple[str, str]]:
    """Collect the classifier-owned fenced blocks, requiring explicit classification.

    Rendered-diagram fences are skipped; every other fenced block must carry an
    explicit ``LANGUAGE executable``/``LANGUAGE illustrative`` classification so
    documentation content is never silently dropped from validation.
    """
    classified: list[tuple[str, str]] = []
    for info, body in _fenced_code_blocks(text):
        language = info.split()[0] if info else ""
        if language in _RENDERED_DIAGRAM_LANGUAGES:
            assert info == language, f"rendered diagram fence must stay bare: {info!r}"
            continue
        assert _CLASSIFIED_INFO.fullmatch(info), (
            f"documentation block {info!r} is not classified executable/illustrative"
        )
        classified.append((info, body))
    return classified


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory: pytest.TempPathFactory) -> InstalledWheel:
    """Build one wheel with this interpreter and install it outside the source tree."""
    workspace = tmp_path_factory.mktemp("installed-documentation")
    dist_dir = workspace / "dist"
    dist_dir.mkdir()

    _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", dist_dir],
        cwd=ROOT,
        timeout=300,
    )
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found: {wheels}"

    site_packages = workspace / "site-packages"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            site_packages,
            wheels[0],
        ],
        cwd=workspace,
    )
    resolved_site_packages = site_packages.resolve()
    assert not resolved_site_packages.is_relative_to(ROOT)
    return InstalledWheel(
        python=Path(sys.executable),
        site_packages=resolved_site_packages,
        workspace=workspace,
    )


def test_declared_python_matrix_supports_documentation_execution():
    """The documented matrix stays 3.10/3.11/3.12 and the running interpreter fits."""
    config = _toml.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    project = config["project"]

    versions = tuple(
        tuple(int(part) for part in classifier.rsplit(" ", 1)[1].split("."))
        for classifier in project["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: \d+\.\d+", classifier)
    )
    assert versions == DECLARED_SUPPORTED_PYTHONS

    requires_python = project["requires-python"].replace(" ", "")
    assert requires_python == ">=3.10"

    # The suite validates documentation on whatever interpreter runs it; require
    # only that this interpreter satisfies the documented floor, never that an
    # unavailable matrix interpreter is present.
    assert sys.version_info[:2] >= min(DECLARED_SUPPORTED_PYTHONS)


def test_documentation_blocks_run_and_syntax_check_against_installed_wheel(
    installed_wheel: InstalledWheel,
    tmp_path: Path,
):
    """Execute documented quickstarts/executable blocks and syntax-check the rest."""
    classified = _classified_blocks(README_PATH.read_text(encoding="utf-8"))
    executable = [info for info, _ in classified if info.endswith(" executable")]
    illustrative = [info for info, _ in classified if info.endswith(" illustrative")]
    assert executable, "expected at least one executable documentation block"
    assert illustrative, "expected at least one illustrative documentation block"

    # Reconstruct the classified blocks in a document that lives outside the
    # source tree so execution never resolves the in-repo package.
    document = tmp_path / "readme-classified.md"
    document.write_text(
        "\n\n".join(f"```{info}\n{body}\n```" for info, body in classified) + "\n",
        encoding="utf-8",
    )
    assert not document.resolve().is_relative_to(ROOT)

    results = validate_documentation(
        [document],
        python_executable=installed_wheel.python,
        installed_package_root=installed_wheel.site_packages,
        source_root=ROOT,
    )

    actions = [result.action for result in results]
    assert actions.count("executed") == len(executable)
    assert actions.count("syntax-checked") == len(illustrative)

    executed = [result for result in results if result.action == "executed"]
    assert all(result.block.language == "python" for result in executed)
    # The offline quickstart records an experience and asserts documented values;
    # a clean exit proves those assertions held against the installed wheel.
    assert any(
        "SimpleHeuristicEvaluator" in result.block.source
        and "learner.record(" in result.block.source
        for result in executed
    )


def test_documented_examples_execute_from_installed_wheel_with_extras(
    installed_wheel: InstalledWheel,
):
    """Documented examples run from the installed artifact with their declared extras."""
    completed = _run(
        _installed_script_command(
            installed_wheel,
            API_GATE,
            "installed",
            "--baseline",
            BASELINE_PATH,
            "--api-reference",
            API_REFERENCE_PATH,
            "--manifest",
            MANIFEST_PATH,
            "--project-root",
            ROOT,
            "--workspace",
            installed_wheel.workspace,
        ),
        cwd=installed_wheel.workspace,
        timeout=120,
    )

    assert "PASS installed-api:" in completed.stdout
    assert "PASS installed-extras:" in completed.stdout
    for identifier in (
        "offline-quickstart",
        "llm-extra-constructors",
        "langchain-extra-constructors",
        "langgraph-extra-constructors",
    ):
        assert f"PASS installed-example[{identifier}]:" in completed.stdout


def test_generated_reference_sections_are_synced_and_cover_contracts():
    """Validate the generated lifecycle/failure/outbound/stability reference block."""
    # Raises ReferenceContractError if the API reference drifts from its sources.
    assert_api_reference_synced()

    block = render_reference_sections(
        contract=load_json(DEFAULT_CONTRACT),
        snapshot=load_json(DEFAULT_API_SNAPSHOT),
        outbound=load_json(DEFAULT_OUTBOUND_CONTRACT),
        schema_support=load_json(DEFAULT_SCHEMA_SUPPORT),
        minimum_minor_releases=MINOR_DEPRECATION_RELEASES,
    )

    # 10.4 lifecycle call order/postconditions/pending-job/idempotence.
    assert "### Lifecycle operations" in block
    for method in _REQUIRED_LIFECYCLE_METHODS:
        assert method in block
    # 10.5 typed failure trigger/state/retry.
    assert "### Typed failure contract" in block
    assert "| Trigger | Typed error | Resulting state | Retry behavior |" in block
    # 2.3 / 10.6 network, credential, and outbound behavior.
    assert "### Network and credential summary" in block
    assert "pass-through" in block
    # 10.10 current-major stability policy.
    assert "### Public API stability" in block
    assert "current major version is" in block

    # The validated block is exactly what the shipped API reference documents.
    assert block in API_REFERENCE_PATH.read_text(encoding="utf-8")
