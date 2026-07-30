"""Installed-wheel compatibility checks for the documented public API."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "api-snapshot.json"
README_PATH = ROOT / "README.md"
SNAPSHOT_GENERATOR_PATH = ROOT / "scripts" / "generate_api_snapshot.py"
QUICKSTART_START = "<!-- BEGIN EXECUTABLE QUICKSTART -->"
QUICKSTART_END = "<!-- END EXECUTABLE QUICKSTART -->"


@dataclass(frozen=True)
class InstalledArtifact:
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
    timeout: int = 120,
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


def _installed_code_command(
    artifact: InstalledArtifact,
    code: str,
) -> list[str | Path]:
    bootstrap = (
        f"import sys; sys.path.insert(0, {str(artifact.site_packages)!r}); {code}"
    )
    return [artifact.python, "-I", "-c", bootstrap]


def _installed_script_command(
    artifact: InstalledArtifact,
    script: Path,
    *arguments: Path,
) -> list[str | Path]:
    argv = [str(script), *(str(argument) for argument in arguments)]
    code = (
        "import runpy; "
        f"sys.argv = {argv!r}; "
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    return _installed_code_command(artifact, code)


@pytest.fixture(scope="module")
def installed_artifact(tmp_path_factory: pytest.TempPathFactory) -> InstalledArtifact:
    workspace = tmp_path_factory.mktemp("installed-artifact")
    dist_dir = workspace / "dist"
    dist_dir.mkdir()

    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            dist_dir,
        ],
        cwd=ROOT,
    )
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"Expected one wheel, found: {wheels}"

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
    return InstalledArtifact(
        python=Path(sys.executable),
        site_packages=site_packages.resolve(),
        workspace=workspace,
    )


def test_installed_wheel_imports_and_constructors_match_canonical_snapshot(
    installed_artifact: InstalledArtifact,
):
    generator = installed_artifact.workspace / "generate_api_snapshot.py"
    shutil.copy2(SNAPSHOT_GENERATOR_PATH, generator)
    generated_path = installed_artifact.workspace / "installed-api-snapshot.json"

    _run(
        _installed_script_command(
            installed_artifact,
            generator,
            Path("--output"),
            generated_path,
        ),
        cwd=installed_artifact.workspace,
    )
    origin = Path(
        _run(
            _installed_code_command(
                installed_artifact,
                "import experia; print(experia.__file__)",
            ),
            cwd=installed_artifact.workspace,
        ).stdout.strip()
    ).resolve()

    assert origin.is_relative_to(installed_artifact.site_packages)
    assert not origin.is_relative_to(ROOT)
    assert json.loads(generated_path.read_text(encoding="utf-8")) == json.loads(
        BASELINE_PATH.read_text(encoding="utf-8")
    )


def test_documented_quickstart_runs_outside_source_tree_against_installed_wheel(
    installed_artifact: InstalledArtifact,
):
    readme = README_PATH.read_text(encoding="utf-8")
    fenced = readme.split(QUICKSTART_START, 1)[1].split(QUICKSTART_END, 1)[0].strip()
    opening = "```python executable\n"
    assert fenced.startswith(opening)
    assert fenced.endswith("\n```")

    quickstart = installed_artifact.workspace / "documented_quickstart.py"
    quickstart.write_text(
        fenced[len(opening) : -len("\n```")] + "\n",
        encoding="utf-8",
    )
    assert not quickstart.resolve().is_relative_to(ROOT)

    _run(
        _installed_script_command(installed_artifact, quickstart),
        cwd=installed_artifact.workspace,
        timeout=30,
    )


def test_reusable_installed_gate_checks_api_extras_and_declared_examples(
    installed_artifact: InstalledArtifact,
):
    completed = _run(
        _installed_script_command(
            installed_artifact,
            ROOT / "scripts" / "api_gate.py",
            Path("installed"),
            Path("--baseline"),
            BASELINE_PATH,
            Path("--api-reference"),
            ROOT / "API_REFERENCE.md",
            Path("--manifest"),
            ROOT / "examples" / "installed-examples.json",
            Path("--project-root"),
            ROOT,
            Path("--workspace"),
            installed_artifact.workspace,
        ),
        cwd=installed_artifact.workspace,
        timeout=60,
    )

    assert "PASS installed-api:" in completed.stdout
    assert "PASS installed-extras:" in completed.stdout
    assert "PASS installed-example[offline-quickstart]:" in completed.stdout
    assert "PASS installed-example[llm-extra-constructors]:" in completed.stdout
    assert "PASS installed-example[langchain-extra-constructors]:" in completed.stdout
    assert "PASS installed-example[langgraph-extra-constructors]:" in completed.stdout


def test_classified_documentation_executes_against_installed_wheel(
    installed_artifact: InstalledArtifact,
):
    from scripts.documentation_blocks import validate_documentation

    document = installed_artifact.workspace / "classified-installed-example.md"
    document.write_text(
        "\n".join(
            [
                "```python executable",
                "from pathlib import Path",
                "import experia",
                "",
                "origin = Path(experia.__file__).resolve()",
                f"assert origin.is_relative_to(Path({str(installed_artifact.site_packages)!r}))",
                f"assert not Path.cwd().resolve().is_relative_to(Path({str(ROOT)!r}))",
                'print("installed-wheel-documentation-ok")',
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    results = validate_documentation(
        [document],
        python_executable=installed_artifact.python,
        installed_package_root=installed_artifact.site_packages,
        source_root=ROOT,
    )

    assert len(results) == 1
    assert results[0].action == "executed"
    assert results[0].stdout.strip() == "installed-wheel-documentation-ok"


def test_failing_executable_documentation_reports_its_source_location(
    installed_artifact: InstalledArtifact,
):
    from scripts.documentation_blocks import (
        DocumentationValidationError,
        validate_documentation,
    )

    document = installed_artifact.workspace / "failing-installed-example.md"
    document.write_text(
        """```python executable
import experia
raise RuntimeError("documented failure")
```
""",
        encoding="utf-8",
    )

    with pytest.raises(DocumentationValidationError) as captured:
        validate_documentation(
            [document],
            python_executable=installed_artifact.python,
            installed_package_root=installed_artifact.site_packages,
            source_root=ROOT,
        )

    assert captured.value.code == "execution_failed"
    assert captured.value.path == document
    assert captured.value.line == 1
    assert "documented failure" in captured.value.detail
