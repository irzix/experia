"""Cross-layer end-to-end verification against the installed Experia wheel.

This test builds and installs the wheel into an isolated location, then drives
the entire pipeline (durable/background recording, sanitization, atomic SQLite
persistence, evaluation/flush, retrieval, framework extraction, shutdown, and
repeated close) from a working directory outside the source tree, using the
deterministic driver in ``tests/e2e_cross_layer_driver.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = Path(__file__).resolve().parent / "e2e_cross_layer_driver.py"
SUCCESS_SENTINEL = "E2E_CROSS_LAYER_OK"


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
    timeout: int = 300,
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
    artifact: InstalledArtifact,
    script: Path,
    *arguments: str | Path,
) -> list[str | Path]:
    argv = [str(script), *(str(argument) for argument in arguments)]
    code = (
        f"import sys; sys.path.insert(0, {str(artifact.site_packages)!r}); "
        "import runpy; "
        f"sys.argv = {argv!r}; "
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    return [artifact.python, "-I", "-c", code]


@pytest.fixture(scope="module")
def installed_artifact(tmp_path_factory: pytest.TempPathFactory) -> InstalledArtifact:
    workspace = tmp_path_factory.mktemp("cross-layer-e2e")
    dist_dir = workspace / "dist"
    dist_dir.mkdir()

    _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", dist_dir],
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


def test_cross_layer_pipeline_runs_against_installed_wheel(
    installed_artifact: InstalledArtifact,
):
    driver = installed_artifact.workspace / "e2e_cross_layer_driver.py"
    shutil.copy2(DRIVER_PATH, driver)
    assert not driver.resolve().is_relative_to(ROOT)

    completed = _run(
        _installed_script_command(
            installed_artifact,
            driver,
            "--installed-root",
            installed_artifact.site_packages,
            "--source-root",
            ROOT,
        ),
        cwd=installed_artifact.workspace,
        timeout=180,
    )

    assert completed.stdout.strip().splitlines()[-1] == SUCCESS_SENTINEL
