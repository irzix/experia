"""Build and smoke-test the exact Experia distribution artifacts.

This credential-free gate performs one build operation, requires exactly one
wheel and one source distribution, records their SHA-256 hashes and sizes, and
installs the wheel into a clean virtual environment. The installed package is
then imported and exercised from a temporary working directory outside the
source repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = Path("dist/quality-gate")
MANIFEST_NAME = "artifact-manifest.json"
MANIFEST_SCHEMA_VERSION = 1
_REQUIRED_SMOKE_CHECKS = (
    "clean-virtual-environment",
    "import-outside-source-tree",
    "sqlite-memory-round-trip",
    "idempotent-close",
)


class ArtifactGateError(RuntimeError):
    """Raised when artifact construction or installed-wheel validation fails."""


@dataclass(frozen=True)
class ArtifactRecord:
    """Stable identity and size metadata for one built distribution artifact."""

    kind: str
    name: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "kind": self.kind,
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class SmokeResult:
    """Evidence returned by the clean installed-wheel smoke process."""

    status: str
    package: str
    package_version: str
    python_version: str
    import_origin: str
    working_directory: str
    checks: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "package": self.package,
            "package_version": self.package_version,
            "python_version": self.python_version,
            "import_origin": self.import_origin,
            "working_directory": self.working_directory,
            "import_origin_outside_repository": True,
            "working_directory_outside_repository": True,
            "checks": list(self.checks),
        }


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for ``path`` without loading it whole."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_artifacts(directory: Path) -> tuple[ArtifactRecord, ArtifactRecord]:
    """Require and describe exactly one wheel and one source distribution."""
    if not directory.is_dir():
        raise ArtifactGateError(f"Artifact directory does not exist: {directory}")

    wheels = sorted(path for path in directory.glob("*.whl") if path.is_file())
    sdists = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (path.name.endswith(".tar.gz") or path.name.endswith(".zip"))
    )
    if len(wheels) != 1 or len(sdists) != 1:
        wheel_names = [path.name for path in wheels]
        sdist_names = [path.name for path in sdists]
        raise ArtifactGateError(
            "Expected exactly one wheel and one sdist; "
            f"observed wheels={wheel_names}, sdists={sdist_names}."
        )

    return (
        _artifact_record("wheel", wheels[0]),
        _artifact_record("sdist", sdists[0]),
    )


def build_artifacts(
    project_root: Path, output_directory: Path
) -> tuple[ArtifactRecord, ArtifactRecord]:
    """Run one isolated build operation and inspect its two expected outputs."""
    project_root = project_root.resolve()
    output_directory = output_directory.resolve()
    if not (project_root / "pyproject.toml").is_file():
        raise ArtifactGateError(
            f"Project root does not contain pyproject.toml: {project_root}"
        )
    _prepare_empty_output_directory(output_directory)
    _run_checked(
        "artifact build",
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--outdir",
            str(output_directory),
            str(project_root),
        ],
        cwd=project_root,
        timeout=300,
    )
    return inspect_artifacts(output_directory)


def smoke_installed_wheel(wheel: Path, project_root: Path) -> SmokeResult:
    """Install ``wheel`` into a clean venv and exercise it outside the repository."""
    wheel = wheel.resolve()
    project_root = project_root.resolve()
    if not wheel.is_file():
        raise ArtifactGateError(f"Wheel does not exist: {wheel}")

    with tempfile.TemporaryDirectory(prefix="experia-wheel-smoke-") as temporary:
        workspace = Path(temporary).resolve()
        if _is_within(workspace, project_root):
            raise ArtifactGateError(
                f"Smoke workspace must be outside the repository: {workspace}"
            )

        environment_directory = workspace / "venv"
        venv.EnvBuilder(with_pip=True, clear=True, system_site_packages=False).create(
            environment_directory
        )
        python = _venv_python(environment_directory)
        environment = _clean_subprocess_environment()
        _run_checked(
            "wheel install",
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                str(wheel),
            ],
            cwd=workspace,
            environment=environment,
            timeout=300,
        )
        completed = _run_checked(
            "installed-wheel smoke",
            [
                str(python),
                "-I",
                "-c",
                _smoke_program(project_root),
            ],
            cwd=workspace,
            environment=environment,
            timeout=60,
        )
        payload = _parse_json_output(completed.stdout)
        return validate_smoke_result(payload, project_root)


def validate_smoke_result(
    payload: Mapping[str, Any], project_root: Path
) -> SmokeResult:
    """Validate smoke evidence, including both outside-repository path claims."""
    try:
        status = payload["status"]
        package = payload["package"]
        package_version = payload["package_version"]
        python_version = payload["python_version"]
        import_origin = payload["import_origin"]
        working_directory = payload["working_directory"]
        raw_checks = payload["checks"]
    except KeyError as error:
        raise ArtifactGateError(
            f"Smoke output is missing required field {error.args[0]!r}."
        ) from error

    string_fields = {
        "status": status,
        "package": package,
        "package_version": package_version,
        "python_version": python_version,
        "import_origin": import_origin,
        "working_directory": working_directory,
    }
    invalid_fields = [
        name for name, value in string_fields.items() if not isinstance(value, str)
    ]
    if invalid_fields:
        raise ArtifactGateError(
            f"Smoke output fields must be strings: {sorted(invalid_fields)}."
        )
    if status != "passed" or package != "experia":
        raise ArtifactGateError(
            f"Smoke reported unexpected identity/status: package={package!r}, "
            f"status={status!r}."
        )
    if not isinstance(raw_checks, list) or any(
        not isinstance(check, str) for check in raw_checks
    ):
        raise ArtifactGateError("Smoke output 'checks' must be a list of strings.")
    checks = tuple(raw_checks)
    if checks != _REQUIRED_SMOKE_CHECKS:
        raise ArtifactGateError(
            f"Smoke checks differ from the required checks: observed={checks}."
        )

    root = project_root.resolve()
    origin = Path(import_origin).resolve()
    workspace = Path(working_directory).resolve()
    if _is_within(origin, root):
        raise ArtifactGateError(
            f"Installed Experia resolved inside the source repository: {origin}"
        )
    if _is_within(workspace, root):
        raise ArtifactGateError(
            f"Smoke working directory resolved inside the source repository: {workspace}"
        )

    return SmokeResult(
        status=status,
        package=package,
        package_version=package_version,
        python_version=python_version,
        import_origin=str(origin),
        working_directory=str(workspace),
        checks=checks,
    )


def build_manifest(
    artifacts: Sequence[ArtifactRecord], smoke: SmokeResult
) -> dict[str, Any]:
    """Create the machine-readable build/install/smoke evidence document."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifacts": [artifact.as_dict() for artifact in artifacts],
        "smoke": smoke.as_dict(),
    }


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize gate evidence deterministically."""
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def run_gate(
    project_root: Path,
    output_directory: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Execute the complete single-build artifact gate and persist its evidence."""
    artifacts = build_artifacts(project_root, output_directory)
    wheel = next(
        output_directory.resolve() / artifact.name
        for artifact in artifacts
        if artifact.kind == "wheel"
    )
    smoke = smoke_installed_wheel(wheel, project_root)
    manifest = build_manifest(artifacts, smoke)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    return manifest


def _artifact_record(kind: str, path: Path) -> ArtifactRecord:
    return ArtifactRecord(
        kind=kind,
        name=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _prepare_empty_output_directory(directory: Path) -> None:
    if directory.exists():
        if not directory.is_dir():
            raise ArtifactGateError(f"Artifact output is not a directory: {directory}")
        entries = sorted(path.name for path in directory.iterdir())
        if entries:
            raise ArtifactGateError(
                f"Artifact output directory must be empty: {directory}; "
                f"observed={entries}."
            )
    else:
        directory.mkdir(parents=True)


def _clean_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(variable, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PIP_NO_INPUT"] = "1"
    return environment


def _run_checked(
    label: str,
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment) if environment is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ArtifactGateError(f"{label} could not complete: {error}") from error
    if completed.returncode != 0:
        raise ArtifactGateError(
            f"{label} failed with exit code {completed.returncode}.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _parse_json_output(output: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise ArtifactGateError(
            f"Installed-wheel smoke did not emit valid JSON: {output!r}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ArtifactGateError("Installed-wheel smoke output must be a JSON object.")
    return payload


def _venv_python(environment_directory: Path) -> Path:
    if os.name == "nt":
        return environment_directory / "Scripts" / "python.exe"
    return environment_directory / "bin" / "python"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _smoke_program(project_root: Path) -> str:
    return f"""
import asyncio
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import experia
from experia import Memory, MemoryType, SQLiteStore

repository = Path({str(project_root.resolve())!r})
origin = Path(experia.__file__).resolve()
working_directory = Path.cwd().resolve()
assert sys.prefix != sys.base_prefix, "smoke interpreter is not a virtual environment"
assert not origin.is_relative_to(repository), origin
assert not working_directory.is_relative_to(repository), working_directory

async def smoke():
    store = SQLiteStore(":memory:")
    await store.initialize()
    try:
        expected = Memory(content="installed wheel smoke", type=MemoryType.FACT)
        await store.save_memory(expected)
        observed = await store.get_memory(expected.id)
        assert observed is not None
        assert observed.id == expected.id
        assert observed.content == expected.content
        assert observed.type is MemoryType.FACT
    finally:
        await store.close()
        await store.close()

asyncio.run(smoke())
print(json.dumps({{
    "status": "passed",
    "package": "experia",
    "package_version": metadata.version("experia"),
    "python_version": platform.python_version(),
    "import_origin": str(origin),
    "working_directory": str(working_directory),
    "checks": {list(_REQUIRED_SMOKE_CHECKS)!r},
}}, sort_keys=True))
""".strip()


def _resolve_from_root(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help=f"project root containing pyproject.toml (default: {PROJECT_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "empty artifact destination, relative to project root by default "
            f"(default: {DEFAULT_OUTPUT_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=f"manifest destination (default: OUTPUT_DIR/{MANIFEST_NAME})",
    )
    arguments = parser.parse_args(argv)

    project_root = arguments.project_root.resolve()
    output_directory = _resolve_from_root(arguments.output_dir, project_root)
    manifest_path = (
        _resolve_from_root(arguments.manifest, project_root)
        if arguments.manifest is not None
        else output_directory / MANIFEST_NAME
    )
    try:
        manifest = run_gate(project_root, output_directory, manifest_path)
    except ArtifactGateError as error:
        print(f"artifact gate failed: {error}", file=sys.stderr)
        return 1

    print(canonical_json(manifest), end="")
    print(f"manifest: {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
