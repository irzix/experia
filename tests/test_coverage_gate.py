"""Tests for granular line-only coverage enforcement."""

from __future__ import annotations

from scripts.coverage_gate import (
    LINE_THRESHOLD,
    TARGETS,
    evaluate_report,
    format_result,
)


def _report(*, covered_lines: int = 85, covered_branches: int = 0):
    return {
        "files": {
            f"{target.path}/module.py": {
                "summary": {
                    "covered_lines": covered_lines,
                    "num_statements": 100,
                    "covered_branches": covered_branches,
                    "num_branches": 100,
                }
            }
            for target in TARGETS
        }
    }


def test_each_target_enforces_85_percent_line_coverage_independently():
    report = _report()
    core_path = f"{TARGETS[0].path}/module.py"
    report["files"][core_path]["summary"]["covered_lines"] = 84

    results = evaluate_report(report)

    assert LINE_THRESHOLD == 85.0
    assert results[0].target.name == "core"
    assert results[0].passed is False
    assert all(result.passed for result in results[1:])
    assert "minimum 85.00%" in format_result(results[0])


def test_branch_measurement_is_reported_but_does_not_affect_line_gate():
    results = evaluate_report(_report(covered_branches=0))

    assert all(result.line_percent == 85.0 for result in results)
    assert all(result.branch_percent == 0.0 for result in results)
    assert all(result.passed for result in results)
    assert all(
        "branches 0.00% (0/100) informational" in format_result(result)
        for result in results
    )
