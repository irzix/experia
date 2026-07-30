"""Focused source metadata checks for the shipped Experia package."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> dict:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)["project"]


def test_project_identity_urls_license_and_python_classifiers_are_truthful() -> None:
    project = _project_metadata()

    assert project["urls"] == {
        "Homepage": "https://github.com/irzix/experia",
        "Repository": "https://github.com/irzix/experia",
        "Issues": "https://github.com/irzix/experia/issues",
    }
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert project["requires-python"] == ">=3.10"
    assert set(project["classifiers"]) == {
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Typing :: Typed",
    }
    assert project["authors"] == [{"name": "Experia AI Contributors"}]


def test_license_and_typing_marker_exist_in_declared_source_locations() -> None:
    license_path = REPOSITORY_ROOT / "LICENSE"
    typing_marker = REPOSITORY_ROOT / "experia" / "py.typed"

    assert license_path.read_text(encoding="utf-8").startswith("MIT License\n")
    assert typing_marker.is_file()
    assert typing_marker.read_bytes() == b""


def test_optional_extras_match_implemented_feature_guards() -> None:
    extras = _project_metadata()["optional-dependencies"]

    assert set(extras) == {"dev", "llm", "langchain", "langgraph"}
    assert extras["llm"] == ["litellm>=1.0.0"]
    assert extras["langchain"] == ["langchain-core>=0.3"]
    assert extras["langgraph"] == ["langgraph"]
    assert set(extras["llm"] + extras["langchain"] + extras["langgraph"]).issubset(
        set(extras["dev"])
    )
    assert "tomli==2.2.1; python_version < '3.11'" in extras["dev"]


def test_ci_runs_optional_dependency_changelog_gate_against_base_revision() -> None:
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "fetch-depth: 0" in workflow
    assert "Validate optional dependency range changelog" in workflow
    assert 'git show "${BASE_REVISION}:pyproject.toml"' in workflow
    assert "python scripts/changelog_gate.py" in workflow
    assert '--previous-pyproject "${base_pyproject}"' in workflow
    assert "--current-pyproject pyproject.toml" in workflow
    assert "--changelog CHANGELOG.md" in workflow
