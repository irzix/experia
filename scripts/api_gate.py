"""Reusable source and installed-artifact public API quality gates."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import venv
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from scripts.api_compatibility import compare_snapshots
    from scripts.generate_api_snapshot import (
        CONSTRUCTOR_BLOCK_END,
        CONSTRUCTOR_BLOCK_START,
        build_snapshot,
        canonical_json,
        render_constructor_contracts,
        resolve_export,
    )
except ModuleNotFoundError:
    from api_compatibility import compare_snapshots
    from generate_api_snapshot import (
        CONSTRUCTOR_BLOCK_END,
        CONSTRUCTOR_BLOCK_START,
        build_snapshot,
        canonical_json,
        render_constructor_contracts,
        resolve_export,
    )

ROOT = SCRIPT_DIR.parent
DEFAULT_BASELINE = ROOT / "api-snapshot.json"
DEFAULT_API_REFERENCE = ROOT / "API_REFERENCE.md"
DEFAULT_MANIFEST = ROOT / "examples" / "installed-examples.json"
_EXTRA_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class GateFailure(RuntimeError):
    """A named, actionable quality-gate failure."""

    def __init__(self, check: str, detail: str) -> None:
        self.check = check
        self.detail = detail
        super().__init__(f"FAIL {check}: {detail}")


@dataclass(frozen=True)
class ExampleSpec:
    """One documented script and the exact extras needed to execute it."""

    identifier: str
    path: Path
    relative_path: str
    extras: tuple[str, ...]
    documentation: tuple[Path, ...]


@dataclass(frozen=True)
class ExtraImportSpec:
    """Installed dependency and public imports associated with one extra."""

    extra: str
    dependency_imports: tuple[str, ...]
    public_imports: tuple[str, ...]
    documentation: tuple[Path, ...]


@dataclass(frozen=True)
class ExampleManifest:
    """Validated installed-example and optional-import contract."""

    package: str
    examples: tuple[ExampleSpec, ...]
    extra_imports: tuple[ExtraImportSpec, ...]

    @property
    def extras(self) -> tuple[str, ...]:
        return tuple(sorted(spec.extra for spec in self.extra_imports))


def _read_json(path: Path, *, check: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateFailure(check, f"cannot read {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise GateFailure(check, f"expected a JSON object in {path}")
    return value


def _required_string(value: object, *, check: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GateFailure(check, f"{field} must be a non-empty string")
    return value


def _string_list(value: object, *, check: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise GateFailure(check, f"{field} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise GateFailure(check, f"{field} contains duplicate values")
    return tuple(value)


def _project_file(
    project_root: Path,
    relative: str,
    *,
    check: str,
    suffix: str | None = None,
) -> Path:
    candidate = (project_root / relative).resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as error:
        raise GateFailure(
            check, f"path escapes the project root: {relative!r}"
        ) from error
    if not candidate.is_file():
        raise GateFailure(check, f"declared file does not exist: {relative}")
    if suffix is not None and candidate.suffix != suffix:
        raise GateFailure(check, f"declared file must end in {suffix}: {relative}")
    return candidate


def load_example_manifest(
    path: Path = DEFAULT_MANIFEST,
    *,
    project_root: Path = ROOT,
) -> ExampleManifest:
    """Load and strictly validate the documented installed-example manifest."""
    check = "example-manifest"
    project_root = project_root.resolve()
    data = _read_json(path.resolve(), check=check)
    if data.get("schema_version") != 1:
        raise GateFailure(
            check, f"unsupported schema_version {data.get('schema_version')!r}"
        )
    package = _required_string(data.get("package"), check=check, field="package")

    raw_examples = data.get("examples")
    if not isinstance(raw_examples, list) or not raw_examples:
        raise GateFailure(check, "examples must be a non-empty list")
    examples: list[ExampleSpec] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(raw_examples):
        field = f"examples[{index}]"
        if not isinstance(raw, Mapping):
            raise GateFailure(check, f"{field} must be an object")
        identifier = _required_string(raw.get("id"), check=check, field=f"{field}.id")
        if identifier in identifiers:
            raise GateFailure(check, f"duplicate example id: {identifier}")
        identifiers.add(identifier)
        relative_path = _required_string(
            raw.get("path"), check=check, field=f"{field}.path"
        )
        example_path = _project_file(
            project_root, relative_path, check=check, suffix=".py"
        )
        extras = _string_list(raw.get("extras"), check=check, field=f"{field}.extras")
        for extra in extras:
            if _EXTRA_PATTERN.fullmatch(extra) is None:
                raise GateFailure(check, f"invalid extra name {extra!r} in {field}")
        doc_names = _string_list(
            raw.get("documentation"),
            check=check,
            field=f"{field}.documentation",
        )
        if not doc_names:
            raise GateFailure(check, f"{field}.documentation must not be empty")
        documentation = tuple(
            _project_file(project_root, name, check=check) for name in doc_names
        )
        examples.append(
            ExampleSpec(
                identifier=identifier,
                path=example_path,
                relative_path=relative_path,
                extras=extras,
                documentation=documentation,
            )
        )

    raw_imports = data.get("extra_imports")
    if not isinstance(raw_imports, list) or not raw_imports:
        raise GateFailure(check, "extra_imports must be a non-empty list")
    imports: list[ExtraImportSpec] = []
    seen_extras: set[str] = set()
    for index, raw in enumerate(raw_imports):
        field = f"extra_imports[{index}]"
        if not isinstance(raw, Mapping):
            raise GateFailure(check, f"{field} must be an object")
        extra = _required_string(raw.get("extra"), check=check, field=f"{field}.extra")
        if _EXTRA_PATTERN.fullmatch(extra) is None:
            raise GateFailure(check, f"invalid extra name {extra!r} in {field}")
        if extra in seen_extras:
            raise GateFailure(check, f"duplicate extra import declaration: {extra}")
        seen_extras.add(extra)
        dependency_imports = _string_list(
            raw.get("dependency_imports"),
            check=check,
            field=f"{field}.dependency_imports",
        )
        public_imports = _string_list(
            raw.get("public_imports"),
            check=check,
            field=f"{field}.public_imports",
        )
        if not dependency_imports or not public_imports:
            raise GateFailure(
                check,
                f"{field} must declare dependency_imports and public_imports",
            )
        doc_names = _string_list(
            raw.get("documentation"),
            check=check,
            field=f"{field}.documentation",
        )
        documentation = tuple(
            _project_file(project_root, name, check=check) for name in doc_names
        )
        imports.append(
            ExtraImportSpec(
                extra=extra,
                dependency_imports=dependency_imports,
                public_imports=public_imports,
                documentation=documentation,
            )
        )

    undeclared = sorted(
        {extra for example in examples for extra in example.extras} - seen_extras
    )
    if undeclared:
        raise GateFailure(
            check,
            f"examples reference extras without import declarations: {undeclared}",
        )
    missing_examples = sorted(
        extra
        for extra in seen_extras
        if not any(extra in example.extras for example in examples)
    )
    if missing_examples:
        raise GateFailure(
            check,
            f"extras have no documented installed example: {missing_examples}",
        )

    manifest = ExampleManifest(
        package=package,
        examples=tuple(examples),
        extra_imports=tuple(imports),
    )
    _validate_manifest_documentation(manifest)
    return manifest


def _validate_manifest_documentation(manifest: ExampleManifest) -> None:
    check = "documented-extras"
    for example in manifest.examples:
        for document in example.documentation:
            text = document.read_text(encoding="utf-8")
            if example.relative_path not in text:
                raise GateFailure(
                    check,
                    f"{document} does not link documented example {example.relative_path}",
                )
            for extra in example.extras:
                installation = f"{manifest.package}[{extra}]"
                if installation not in text:
                    raise GateFailure(
                        check,
                        f"{document} does not name required extra {installation} "
                        f"for {example.identifier}",
                    )
    for extra_spec in manifest.extra_imports:
        installation = f"{manifest.package}[{extra_spec.extra}]"
        for document in extra_spec.documentation:
            text = document.read_text(encoding="utf-8")
            missing = [
                value
                for value in (installation, *extra_spec.public_imports)
                if value not in text
            ]
            if missing:
                raise GateFailure(
                    check,
                    f"{document} is missing declarations for {extra_spec.extra}: {missing}",
                )


def load_baseline(path: Path = DEFAULT_BASELINE) -> Mapping[str, Any]:
    """Load a canonical snapshot without ever modifying the baseline file."""
    check = "api-baseline"
    resolved = path.resolve()
    baseline = _read_json(resolved, check=check)
    if resolved.read_text(encoding="utf-8") != canonical_json(dict(baseline)):
        raise GateFailure(check, f"baseline is not canonical JSON: {resolved}")
    return baseline


def _changed_export_paths(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[str, ...]:
    def by_path(snapshot: Mapping[str, Any]) -> dict[str, object]:
        exports = snapshot.get("exports")
        if not isinstance(exports, list):
            return {}
        return {
            str(item.get("path")): item
            for item in exports
            if isinstance(item, Mapping) and item.get("path") is not None
        }

    old = by_path(baseline)
    new = by_path(candidate)
    return tuple(
        sorted(
            path for path in old.keys() | new.keys() if old.get(path) != new.get(path)
        )
    )


def check_snapshot(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    installed: bool,
) -> str:
    """Enforce same-major semantic compatibility and same-version synchronization."""
    check = "installed-api" if installed else "api-baseline"
    baseline_major = baseline.get("major_version")
    candidate_major = candidate.get("major_version")
    if baseline_major != candidate_major:
        raise GateFailure(
            check,
            "candidate must be checked against its current-major baseline; "
            f"baseline major={baseline_major!r}, candidate major={candidate_major!r}",
        )

    report = compare_snapshots(baseline, candidate)
    if not report.compatible:
        details = "; ".join(
            f"[{issue.code}] {issue.path}: {issue.detail}" for issue in report.issues
        )
        raise GateFailure(check, details)

    if baseline.get("package_version") == candidate.get("package_version"):
        changed = _changed_export_paths(baseline, candidate)
        if changed:
            preview = ", ".join(changed[:8])
            if len(changed) > 8:
                preview += f", ... ({len(changed)} changed paths)"
            raise GateFailure(
                check,
                "installed/source API differs from the baseline for the same package "
                f"version; update versioned API evidence, docs, and changelog first: {preview}",
            )

    location = "installed" if installed else "source"
    exports = candidate.get("exports")
    export_count = len(exports) if isinstance(exports, list) else 0
    return (
        f"PASS {check}: {location} {candidate.get('package_version')} is compatible "
        f"with current-major baseline {baseline.get('package_version')} "
        f"({export_count} exports)"
    )


def check_api_reference(candidate: Mapping[str, Any], path: Path) -> str:
    """Ensure generated constructor docs exactly match the inspected candidate."""
    check = "api-reference"
    try:
        reference = path.resolve().read_text(encoding="utf-8")
    except OSError as error:
        raise GateFailure(check, f"cannot read {path}: {error}") from error
    if (
        reference.count(CONSTRUCTOR_BLOCK_START) != 1
        or reference.count(CONSTRUCTOR_BLOCK_END) != 1
    ):
        raise GateFailure(check, f"{path} must contain exactly one constructor block")
    actual = (
        CONSTRUCTOR_BLOCK_START
        + reference.split(CONSTRUCTOR_BLOCK_START, 1)[1].split(
            CONSTRUCTOR_BLOCK_END, 1
        )[0]
        + CONSTRUCTOR_BLOCK_END
    )
    expected = render_constructor_contracts(dict(candidate))
    if actual != expected:
        raise GateFailure(
            check,
            f"{path} does not match inspected signatures; regenerate and review API docs",
        )
    return "PASS api-reference: constructor contracts match inspected signatures"


def check_installed_origin(package: str, source_root: Path) -> str:
    """Prove the imported package comes from an installed artifact, not the source tree."""
    check = "installed-origin"
    module = importlib.import_module(package)
    origin_value = getattr(module, "__file__", None)
    if origin_value is None:
        raise GateFailure(check, f"{package} has no inspectable __file__")
    origin = Path(origin_value).resolve()
    try:
        origin.relative_to(source_root.resolve())
    except ValueError:
        return f"PASS installed-origin: {origin}"
    raise GateFailure(check, f"{package} resolved inside source tree: {origin}")


def _provided_extras(package: str) -> set[str]:
    try:
        distribution = metadata.distribution(package)
    except metadata.PackageNotFoundError as error:
        raise GateFailure(
            "installed-extras", f"distribution is not installed: {package}"
        ) from error
    return set(distribution.metadata.get_all("Provides-Extra") or ())


def check_installed_extras(
    manifest: ExampleManifest,
    specs: Iterable[ExtraImportSpec] | None = None,
) -> list[str]:
    """Verify wheel extra metadata, dependencies, and documented public imports."""
    provided = _provided_extras(manifest.package)
    expected = set(manifest.extras)
    missing_metadata = sorted(expected - provided)
    if missing_metadata:
        raise GateFailure(
            "installed-extras",
            f"wheel metadata is missing declared extras: {missing_metadata}; "
            f"observed={sorted(provided)}",
        )

    messages = [
        "PASS installed-extras: wheel metadata provides " + ", ".join(sorted(expected))
    ]
    selected = tuple(specs) if specs is not None else manifest.extra_imports
    for spec in selected:
        try:
            for module_name in spec.dependency_imports:
                importlib.import_module(module_name)
            for path in spec.public_imports:
                resolve_export(path)
        except (ImportError, AttributeError) as error:
            raise GateFailure(
                f"extra-imports[{spec.extra}]",
                f"declared extra {manifest.package}[{spec.extra}] cannot resolve "
                f"its installed imports: {type(error).__name__}: {error}",
            ) from error
        messages.append(
            f"PASS extra-imports[{spec.extra}]: "
            f"{len(spec.public_imports)} public imports resolved"
        )
    return messages


def require_external_workspace(workspace: Path, project_root: Path) -> Path:
    """Require installed checks to execute from outside the source repository."""
    resolved_workspace = workspace.resolve()
    resolved_root = project_root.resolve()
    try:
        resolved_workspace.relative_to(resolved_root)
    except ValueError:
        return resolved_workspace
    raise GateFailure(
        "execution-isolation",
        f"workspace must be outside source tree {resolved_root}; "
        f"observed={resolved_workspace}",
    )


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_installed_example(example: ExampleSpec, workspace: Path) -> str:
    """Copy and execute one example outside the source tree in this interpreter."""
    check = f"installed-example[{example.identifier}]"
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="experia-example-", dir=workspace) as raw:
        run_directory = Path(raw)
        copied = run_directory / example.path.name
        shutil.copy2(example.path, copied)
        previous_argv = sys.argv
        try:
            sys.argv = [str(copied)]
            with _working_directory(run_directory):
                runpy.run_path(str(copied), run_name="__main__")
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)) and not (
                isinstance(error, SystemExit) and error.code in (None, 0)
            ):
                raise
            if isinstance(error, SystemExit):
                pass
            else:
                raise GateFailure(
                    check,
                    f"{example.relative_path} failed with {type(error).__name__}: {error}",
                ) from error
        finally:
            sys.argv = previous_argv
    extras = ",".join(example.extras) if example.extras else "base"
    return f"PASS {check}: {example.relative_path} (extras={extras})"


def run_source_gate(
    *,
    baseline_path: Path,
    api_reference: Path,
    manifest_path: Path,
    project_root: Path,
) -> list[str]:
    manifest = load_example_manifest(manifest_path, project_root=project_root)
    baseline = load_baseline(baseline_path)
    candidate = build_snapshot()
    return [
        check_snapshot(baseline, candidate, installed=False),
        check_api_reference(candidate, api_reference),
        f"PASS example-manifest: {len(manifest.examples)} examples declare exact extras",
        "PASS documented-extras: manifest declarations are present in documentation",
    ]


def run_installed_gate(
    *,
    baseline_path: Path,
    api_reference: Path,
    manifest_path: Path,
    project_root: Path,
    workspace: Path,
    skip_api: bool = False,
    skip_examples: bool = False,
    example_ids: Sequence[str] = (),
) -> list[str]:
    manifest = load_example_manifest(manifest_path, project_root=project_root)
    workspace = require_external_workspace(workspace, project_root)
    selected = manifest.examples
    if example_ids:
        wanted = set(example_ids)
        selected = tuple(
            item for item in manifest.examples if item.identifier in wanted
        )
        missing = sorted(wanted - {item.identifier for item in selected})
        if missing:
            raise GateFailure("example-manifest", f"unknown example ids: {missing}")

    messages = [check_installed_origin(manifest.package, project_root)]
    relevant_extras = (
        manifest.extra_imports
        if not skip_api
        else tuple(
            spec
            for spec in manifest.extra_imports
            if any(spec.extra in example.extras for example in selected)
        )
    )
    messages.extend(check_installed_extras(manifest, relevant_extras))
    if not skip_api:
        baseline = load_baseline(baseline_path)
        candidate = build_snapshot()
        messages.append(check_snapshot(baseline, candidate, installed=True))
        messages.append(check_api_reference(candidate, api_reference))
    if not skip_examples:
        messages.extend(run_installed_example(item, workspace) for item in selected)
    return messages


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _run_checked(command: Sequence[str], *, cwd: Path, check: str) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        detail = "\n".join(
            part
            for part in (completed.stdout.strip(), completed.stderr.strip())
            if part
        )
        raise GateFailure(check, detail or f"exit status {completed.returncode}")
    if completed.stdout:
        print(completed.stdout, end="")


def _install_requirement(package: str, wheel: Path, extras: Sequence[str]) -> str:
    selected = f"[{','.join(extras)}]" if extras else ""
    return f"{package}{selected} @ {wheel.resolve().as_uri()}"


def run_artifact_gate(
    *,
    wheel: Path,
    baseline_path: Path,
    api_reference: Path,
    manifest_path: Path,
    project_root: Path,
    workspace: Path,
) -> list[str]:
    """Install one wheel into clean per-extra venvs and run the complete gate."""
    check = "artifact-api-gate"
    wheel = wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise GateFailure(check, f"expected one wheel path, got {wheel}")
    manifest = load_example_manifest(manifest_path, project_root=project_root)
    workspace = require_external_workspace(workspace, project_root)
    workspace.mkdir(parents=True, exist_ok=True)

    groups = {example.extras for example in manifest.examples}
    groups.add(manifest.extras)
    environments: dict[tuple[str, ...], tuple[Path, Path]] = {}
    for index, extras in enumerate(sorted(groups)):
        environment_path = workspace / f"venv-{index}"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment_path)
        python = _venv_python(environment_path)
        requirement = _install_requirement(manifest.package, wheel, extras)
        _run_checked(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-input",
                requirement,
            ],
            cwd=workspace,
            check=f"install-extras[{','.join(extras) or 'base'}]",
        )
        environments[extras] = (environment_path, python)

    common = [
        str(Path(__file__).resolve()),
        "installed",
        "--baseline",
        str(baseline_path.resolve()),
        "--api-reference",
        str(api_reference.resolve()),
        "--manifest",
        str(manifest_path.resolve()),
        "--project-root",
        str(project_root.resolve()),
        "--workspace",
        str(workspace),
    ]
    _, inspection_python = environments[manifest.extras]
    _run_checked(
        [str(inspection_python), *common, "--skip-examples"],
        cwd=workspace,
        check="installed-api",
    )

    for extras in sorted({example.extras for example in manifest.examples}):
        _, python = environments[extras]
        identifiers = [
            item.identifier for item in manifest.examples if item.extras == extras
        ]
        command = [str(python), *common, "--skip-api"]
        for identifier in identifiers:
            command.extend(["--example-id", identifier])
        _run_checked(
            command,
            cwd=workspace,
            check=f"installed-examples[{','.join(extras) or 'base'}]",
        )

    return [
        f"PASS artifact-api-gate: inspected {wheel.name} and ran "
        f"{len(manifest.examples)} examples in {len(groups)} exact-extra environments"
    ]


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--api-reference", type=Path, default=DEFAULT_API_REFERENCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--project-root", type=Path, default=ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser(
        "source", help="check source API/docs against the current-major baseline"
    )
    _add_common_arguments(source)

    installed = commands.add_parser(
        "installed",
        help="inspect an already-installed artifact and run declared examples",
    )
    _add_common_arguments(installed)
    installed.add_argument("--workspace", type=Path, default=Path.cwd())
    installed.add_argument("--skip-api", action="store_true", help=argparse.SUPPRESS)
    installed.add_argument(
        "--skip-examples", action="store_true", help=argparse.SUPPRESS
    )
    installed.add_argument(
        "--example-id", action="append", default=[], help=argparse.SUPPRESS
    )

    artifact = commands.add_parser(
        "artifact",
        help="install a wheel in clean exact-extra environments and run all gates",
    )
    _add_common_arguments(artifact)
    artifact.add_argument("--wheel", type=Path, required=True)
    artifact.add_argument("--workspace", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "source":
            messages = run_source_gate(
                baseline_path=arguments.baseline,
                api_reference=arguments.api_reference,
                manifest_path=arguments.manifest,
                project_root=arguments.project_root,
            )
        elif arguments.command == "installed":
            messages = run_installed_gate(
                baseline_path=arguments.baseline,
                api_reference=arguments.api_reference,
                manifest_path=arguments.manifest,
                project_root=arguments.project_root,
                workspace=arguments.workspace,
                skip_api=arguments.skip_api,
                skip_examples=arguments.skip_examples,
                example_ids=arguments.example_id,
            )
        else:
            messages = run_artifact_gate(
                wheel=arguments.wheel,
                baseline_path=arguments.baseline,
                api_reference=arguments.api_reference,
                manifest_path=arguments.manifest,
                project_root=arguments.project_root,
                workspace=arguments.workspace,
            )
    except GateFailure as error:
        print(error, file=sys.stderr)
        return 1
    for message in messages:
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
