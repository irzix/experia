"""Enforce granular line coverage while reporting branches informationally."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

LINE_THRESHOLD = 85.0
DEFAULT_REPORT = Path(".coverage.granular.json")


@dataclass(frozen=True)
class CoverageTarget:
    """One source directory with an independently enforced line threshold."""

    name: str
    path: str


@dataclass(frozen=True)
class CoverageResult:
    """Aggregated line and branch measurements for one coverage target."""

    target: CoverageTarget
    file_count: int
    covered_lines: int
    statement_count: int
    covered_branches: int
    branch_count: int
    line_percent: float
    branch_percent: float | None
    passed: bool


TARGETS = (
    CoverageTarget("core", "experia/core"),
    CoverageTarget("memory", "experia/memory"),
    CoverageTarget("integration:langchain", "experia/integrations/langchain"),
    CoverageTarget("integration:langgraph", "experia/integrations/langgraph"),
)


class CoverageReportError(ValueError):
    """Raised when coverage JSON cannot support granular measurement."""


def evaluate_report(
    report: Mapping[str, Any],
    *,
    targets: Sequence[CoverageTarget] = TARGETS,
    line_threshold: float = LINE_THRESHOLD,
) -> tuple[CoverageResult, ...]:
    """Aggregate and independently evaluate line coverage for every target."""
    if not 0.0 <= line_threshold <= 100.0:
        raise CoverageReportError("line_threshold must be between 0 and 100")

    files = report.get("files")
    if not isinstance(files, Mapping):
        raise CoverageReportError("coverage report must contain a 'files' mapping")

    results = []
    for target in targets:
        prefix = target.path.rstrip("/") + "/"
        matching_summaries = [
            payload.get("summary")
            for raw_path, payload in files.items()
            if _normalized_path(raw_path).startswith(prefix)
            and isinstance(payload, Mapping)
        ]
        if any(not isinstance(summary, Mapping) for summary in matching_summaries):
            raise CoverageReportError(
                f"coverage files for {target.name!r} must contain summary mappings"
            )

        covered_lines = sum(
            _count(summary, "covered_lines", target) for summary in matching_summaries
        )
        statement_count = sum(
            _count(summary, "num_statements", target) for summary in matching_summaries
        )
        covered_branches = sum(
            _count(summary, "covered_branches", target)
            for summary in matching_summaries
        )
        branch_count = sum(
            _count(summary, "num_branches", target) for summary in matching_summaries
        )
        if covered_lines > statement_count:
            raise CoverageReportError(
                f"covered lines exceed statements for target {target.name!r}"
            )
        if covered_branches > branch_count:
            raise CoverageReportError(
                f"covered branches exceed branches for target {target.name!r}"
            )

        line_percent = (
            100.0 * covered_lines / statement_count if statement_count else 0.0
        )
        branch_percent = (
            100.0 * covered_branches / branch_count if branch_count else None
        )
        results.append(
            CoverageResult(
                target=target,
                file_count=len(matching_summaries),
                covered_lines=covered_lines,
                statement_count=statement_count,
                covered_branches=covered_branches,
                branch_count=branch_count,
                line_percent=line_percent,
                branch_percent=branch_percent,
                passed=statement_count > 0 and line_percent >= line_threshold,
            )
        )
    return tuple(results)


def load_report(path: Path) -> Mapping[str, Any]:
    """Load one coverage.py JSON report with a contextual parse error."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CoverageReportError(
            f"unable to read coverage report {path}: {error}"
        ) from error
    if not isinstance(report, Mapping):
        raise CoverageReportError("coverage report root must be a JSON object")
    return report


def format_result(result: CoverageResult) -> str:
    """Render one stable, named coverage diagnostic."""
    status = "PASS" if result.passed else "FAIL"
    branch = (
        f"{result.branch_percent:.2f}% "
        f"({result.covered_branches}/{result.branch_count})"
        if result.branch_percent is not None
        else "n/a (0/0)"
    )
    return (
        f"{status} {result.target.name}: lines {result.line_percent:.2f}% "
        f"({result.covered_lines}/{result.statement_count}, minimum "
        f"{LINE_THRESHOLD:.2f}%); branches {branch} informational"
    )


def _normalized_path(path: object) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def _count(summary: Mapping[str, Any], key: str, target: CoverageTarget) -> int:
    value = summary.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CoverageReportError(
            f"{key!r} for target {target.name!r} must be a non-negative integer"
        )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"coverage.py JSON report (default: {DEFAULT_REPORT})",
    )
    arguments = parser.parse_args(argv)

    try:
        results = evaluate_report(load_report(arguments.report))
    except CoverageReportError as error:
        parser.error(str(error))

    for result in results:
        print(format_result(result))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
