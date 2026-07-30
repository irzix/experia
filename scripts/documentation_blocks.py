"""Validate explicitly classified Markdown code blocks.

Fence info strings use the strict form ``LANGUAGE CLASSIFICATION`` where
``CLASSIFICATION`` is either ``executable`` or ``illustrative``. Executable
Python blocks run independently in an isolated working directory against an
explicit installed-package root. Illustrative blocks are parsed by the checker
for their declared language and unsupported languages fail closed.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

_OPENING_FENCE = re.compile(
    r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>.*)$"
)
_LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "sh": "bash",
    "shell": "bash",
    "plaintext": "text",
}
SUPPORTED_EXECUTABLE_LANGUAGES = frozenset({"python"})
SUPPORTED_ILLUSTRATIVE_LANGUAGES = frozenset({"bash", "json", "python", "text"})


class BlockClassification(str, Enum):
    """Required execution policy for a fenced documentation block."""

    EXECUTABLE = "executable"
    ILLUSTRATIVE = "illustrative"


@dataclass(frozen=True)
class DocumentationBlock:
    """One classified fenced block with source location information."""

    path: Path
    line: int
    declared_language: str
    language: str
    classification: BlockClassification
    source: str


@dataclass(frozen=True)
class BlockValidationResult:
    """Successful validation evidence for one documentation block."""

    block: DocumentationBlock
    action: str
    stdout: str = ""


class DocumentationValidationError(ValueError):
    """Raised when documentation cannot satisfy the block-validation contract."""

    def __init__(
        self,
        *,
        path: Path,
        line: int,
        code: str,
        detail: str,
    ) -> None:
        self.path = path
        self.line = line
        self.code = code
        self.detail = detail
        super().__init__(f"{path}:{line}: [{code}] {detail}")


def parse_document(path: Path) -> tuple[DocumentationBlock, ...]:
    """Parse all fenced blocks and require explicit language/classification."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise DocumentationValidationError(
            path=path,
            line=1,
            code="unreadable_document",
            detail=str(error),
        ) from error

    blocks: list[DocumentationBlock] = []
    opening: tuple[str, int, str, BlockClassification, str, int] | None = None
    body: list[str] = []

    for line_number, raw_line in enumerate(lines, start=1):
        if opening is None:
            match = _OPENING_FENCE.fullmatch(raw_line)
            if match is None:
                continue
            fence = match.group("fence")
            language, classification = _parse_info_string(
                path,
                line_number,
                match.group("info"),
            )
            opening = (
                fence[0],
                len(fence),
                language,
                classification,
                match.group("info").split()[0],
                len(match.group("indent")),
            )
            body = []
            continue

        marker, minimum_length, language, classification, declared, indent = opening
        if _is_closing_fence(raw_line, marker, minimum_length):
            blocks.append(
                DocumentationBlock(
                    path=path,
                    line=line_number - len(body) - 1,
                    declared_language=declared,
                    language=language,
                    classification=classification,
                    source="\n".join(body),
                )
            )
            opening = None
            body = []
        else:
            body.append(_strip_container_indent(raw_line, indent))

    if opening is not None:
        opening_line = len(lines) - len(body)
        raise DocumentationValidationError(
            path=path,
            line=opening_line,
            code="unclosed_fence",
            detail="fenced code block has no matching closing fence",
        )
    return tuple(blocks)


def validate_documentation(
    paths: Sequence[Path],
    *,
    python_executable: Path | None = None,
    installed_package_root: Path | None = None,
    source_root: Path | None = None,
    package_name: str = "experia",
    timeout: float = 30.0,
) -> tuple[BlockValidationResult, ...]:
    """Validate blocks, executing Python examples only against an installed wheel.

    Installed-artifact context is mandatory when any block is executable. The
    imported package must resolve below ``installed_package_root`` and outside
    ``source_root``. Each executable block runs as a fresh script below a
    temporary directory that is also outside the source tree.
    """
    blocks = tuple(block for path in paths for block in parse_document(Path(path)))
    for block in blocks:
        if (
            block.classification is BlockClassification.EXECUTABLE
            and block.language not in SUPPORTED_EXECUTABLE_LANGUAGES
        ):
            raise _error(
                block,
                "unsupported_executable_language",
                f"declared language {block.declared_language!r} is not supported; "
                f"supported: {sorted(SUPPORTED_EXECUTABLE_LANGUAGES)}",
            )
        if (
            block.classification is BlockClassification.ILLUSTRATIVE
            and block.language not in SUPPORTED_ILLUSTRATIVE_LANGUAGES
        ):
            raise _error(
                block,
                "unsupported_illustrative_language",
                f"declared language {block.declared_language!r} has no syntax checker; "
                f"supported: {sorted(SUPPORTED_ILLUSTRATIVE_LANGUAGES)}",
            )

    executable = tuple(
        block
        for block in blocks
        if block.classification is BlockClassification.EXECUTABLE
    )

    context: _InstalledContext | None = None
    if executable:
        context = _prepare_installed_context(
            executable[0],
            python_executable=python_executable,
            installed_package_root=installed_package_root,
            source_root=source_root,
            package_name=package_name,
            timeout=timeout,
        )

    results: list[BlockValidationResult] = []
    with tempfile.TemporaryDirectory(prefix="experia-documentation-") as raw_workspace:
        workspace = Path(raw_workspace).resolve()
        if context is not None and _is_within(workspace, context.source_root):
            raise _error(
                executable[0],
                "source_tree_execution",
                f"temporary execution directory resolved inside {context.source_root}",
            )

        for index, block in enumerate(blocks):
            if block.classification is BlockClassification.EXECUTABLE:
                if block.language not in SUPPORTED_EXECUTABLE_LANGUAGES:
                    raise _error(
                        block,
                        "unsupported_executable_language",
                        f"declared language {block.declared_language!r} is not supported; "
                        f"supported: {sorted(SUPPORTED_EXECUTABLE_LANGUAGES)}",
                    )
                assert context is not None
                results.append(
                    _execute_python_block(
                        block,
                        context=context,
                        workspace=workspace / f"block-{index}",
                        timeout=timeout,
                    )
                )
            else:
                results.append(_validate_illustrative_block(block))
    return tuple(results)


def _parse_info_string(
    path: Path,
    line: int,
    info: str,
) -> tuple[str, BlockClassification]:
    tokens = info.split()
    expected = (
        "expected fence metadata 'LANGUAGE executable' or 'LANGUAGE illustrative'"
    )
    if len(tokens) != 2:
        raise DocumentationValidationError(
            path=path,
            line=line,
            code="invalid_classification",
            detail=expected,
        )
    declared_language, raw_classification = tokens
    try:
        classification = BlockClassification(raw_classification)
    except ValueError as error:
        raise DocumentationValidationError(
            path=path,
            line=line,
            code="invalid_classification",
            detail=expected,
        ) from error
    language = _LANGUAGE_ALIASES.get(declared_language, declared_language)
    return language, classification


def _is_closing_fence(line: str, marker: str, minimum_length: int) -> bool:
    stripped = line.lstrip(" ")
    indent = len(line) - len(stripped)
    candidate = stripped.rstrip(" \t")
    return (
        indent <= 3
        and len(candidate) >= minimum_length
        and candidate
        and set(candidate) == {marker}
    )


def _strip_container_indent(line: str, indent: int) -> str:
    removable = min(indent, len(line) - len(line.lstrip(" ")))
    return line[removable:]


@dataclass(frozen=True)
class _InstalledContext:
    python: Path
    package_root: Path
    source_root: Path
    package_name: str


def _prepare_installed_context(
    block: DocumentationBlock,
    *,
    python_executable: Path | None,
    installed_package_root: Path | None,
    source_root: Path | None,
    package_name: str,
    timeout: float,
) -> _InstalledContext:
    if (
        python_executable is None
        or installed_package_root is None
        or source_root is None
    ):
        raise _error(
            block,
            "missing_installed_context",
            "executable blocks require python_executable, installed_package_root, "
            "and source_root",
        )
    # Keep virtual-environment launcher symlinks intact: resolving the symlink can
    # select the base interpreter and discard the environment's site-packages.
    python = Path(python_executable).expanduser().absolute()
    package_root = Path(installed_package_root).resolve()
    source = Path(source_root).resolve()
    if not python.is_file():
        raise _error(block, "invalid_python", f"Python executable not found: {python}")
    if not package_root.is_dir():
        raise _error(
            block,
            "invalid_installed_root",
            f"installed package root not found: {package_root}",
        )
    if _is_within(package_root, source):
        raise _error(
            block,
            "source_tree_install",
            f"installed package root must be outside source tree {source}",
        )
    if not package_name or not all(
        part.isidentifier() for part in package_name.split(".")
    ):
        raise _error(
            block,
            "invalid_package_name",
            f"invalid package name {package_name!r}",
        )

    context = _InstalledContext(
        python=python,
        package_root=package_root,
        source_root=source,
        package_name=package_name,
    )
    _verify_installed_package(block, context=context, timeout=timeout)
    return context


def _verify_installed_package(
    block: DocumentationBlock,
    *,
    context: _InstalledContext,
    timeout: float,
) -> None:
    code = (
        "import importlib, pathlib, sys; "
        f"sys.path.insert(0, {str(context.package_root)!r}); "
        f"module = importlib.import_module({context.package_name!r}); "
        "print(pathlib.Path(module.__file__).resolve())"
    )
    completed = _run_python(
        context,
        code,
        cwd=context.package_root.parent,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise _error(
            block,
            "installed_import_failed",
            _process_detail(completed),
        )
    output_lines = completed.stdout.strip().splitlines()
    if len(output_lines) != 1:
        raise _error(
            block,
            "installed_import_origin",
            f"expected one package origin line, observed {completed.stdout!r}",
        )
    origin = Path(output_lines[0]).resolve()
    if not _is_within(origin, context.package_root) or _is_within(
        origin, context.source_root
    ):
        raise _error(
            block,
            "installed_import_origin",
            f"{context.package_name} resolved to {origin}, expected below "
            f"{context.package_root} and outside {context.source_root}",
        )


def _execute_python_block(
    block: DocumentationBlock,
    *,
    context: _InstalledContext,
    workspace: Path,
    timeout: float,
) -> BlockValidationResult:
    workspace.mkdir()
    script = workspace / "example.py"
    script.write_text(block.source + "\n", encoding="utf-8")
    code = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(context.package_root)!r}); "
        f"sys.argv = [{str(script)!r}]; "
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    completed = _run_python(context, code, cwd=workspace, timeout=timeout)
    if completed.returncode != 0:
        raise _error(block, "execution_failed", _process_detail(completed))
    return BlockValidationResult(
        block=block, action="executed", stdout=completed.stdout
    )


def _run_python(
    context: _InstalledContext,
    code: str,
    *,
    cwd: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    try:
        return subprocess.run(
            [str(context.python), "-I", "-c", code],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DocumentationValidationError(
            path=Path("<installed-python>"),
            line=1,
            code="execution_environment_failed",
            detail=str(error),
        ) from error


def _validate_illustrative_block(
    block: DocumentationBlock,
) -> BlockValidationResult:
    if block.language not in SUPPORTED_ILLUSTRATIVE_LANGUAGES:
        raise _error(
            block,
            "unsupported_illustrative_language",
            f"declared language {block.declared_language!r} has no syntax checker; "
            f"supported: {sorted(SUPPORTED_ILLUSTRATIVE_LANGUAGES)}",
        )
    try:
        if block.language == "python":
            ast.parse(block.source, filename=f"{block.path}:{block.line}")
        elif block.language == "json":
            json.loads(block.source)
        elif block.language == "bash":
            _check_bash(block.source)
        elif "\x00" in block.source:
            raise ValueError("text blocks cannot contain NUL characters")
    except (SyntaxError, json.JSONDecodeError, ValueError) as error:
        raise _error(block, "invalid_syntax", str(error)) from error
    return BlockValidationResult(block=block, action="syntax-checked")


def _check_bash(source: str) -> None:
    bash = shutil.which("bash")
    if bash is None:
        raise ValueError("bash syntax checker is unavailable")
    completed = subprocess.run(
        [bash, "-n"],
        input=source,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "bash rejected the block")


def _process_detail(completed: subprocess.CompletedProcess[str]) -> str:
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    return f"exit code {completed.returncode}; stdout={stdout!r}; stderr={stderr!r}"


def _error(
    block: DocumentationBlock,
    code: str,
    detail: str,
) -> DocumentationValidationError:
    return DocumentationValidationError(
        path=block.path,
        line=block.line,
        code=code,
        detail=detail,
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def format_result(result: BlockValidationResult) -> str:
    """Render stable success evidence for command-line quality gates."""
    block = result.block
    return (
        f"PASS {block.path}:{block.line}: {result.action} "
        f"{block.classification.value} {block.declared_language} block"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--python", type=Path, dest="python_executable")
    parser.add_argument("--installed-package-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--package", default="experia", dest="package_name")
    parser.add_argument("--timeout", type=float, default=30.0)
    arguments = parser.parse_args(argv)

    try:
        results = validate_documentation(
            arguments.documents,
            python_executable=arguments.python_executable,
            installed_package_root=arguments.installed_package_root,
            source_root=arguments.source_root,
            package_name=arguments.package_name,
            timeout=arguments.timeout,
        )
    except DocumentationValidationError as error:
        print(f"FAIL documentation-blocks: {error}", file=sys.stderr)
        return 1

    for result in results:
        print(format_result(result))
    print(f"PASS documentation-blocks: validated {len(results)} block(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
