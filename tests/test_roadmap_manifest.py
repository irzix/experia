"""Focused tests for the roadmap ownership/readiness manifest and validator."""

from copy import deepcopy

import pytest

from scripts.roadmap_manifest import (
    DEFAULT_MANIFEST,
    RoadmapManifestError,
    assert_planned_entrypoints_unavailable,
    assert_readme_synced,
    canonical_yaml,
    load_manifest,
    render_roadmap_status,
    validate_entry,
    validate_manifest,
)


def _item(manifest, item_id):
    return next(item for item in manifest["items"] if item["id"] == item_id)


def test_committed_manifest_is_canonical_valid_and_documentation_synced():
    manifest = load_manifest()

    validate_manifest(manifest)

    # The committed file is canonical block-style YAML.
    assert DEFAULT_MANIFEST.read_text(encoding="utf-8") == canonical_yaml(manifest)
    # Ownership is never invented: every current item is explicitly unassigned.
    assert all(item.get("unassigned") is True for item in manifest["items"])
    assert all(item["readiness"] == "planned" for item in manifest["items"])
    # Every importable placeholder is connected to a live UnavailableFeatureError.
    assert assert_planned_entrypoints_unavailable(manifest) == 4
    # The README Project Status planned section is generated from the manifest.
    assert_readme_synced(manifest)


def test_rendered_status_lists_every_item_as_unavailable_planned():
    manifest = load_manifest()

    rendered = render_roadmap_status(manifest)

    for item in manifest["items"]:
        assert item["title"] in rendered
    assert rendered.count("readiness `planned`") == len(manifest["items"])
    assert "`unassigned`" in rendered
    assert "raises `UnavailableFeatureError`" in rendered


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda m: _item(m, "autogen").pop("unassigned"), "exactly one ownership"),
        (
            lambda m: _item(m, "autogen").update(owner="a-maintainer"),
            "exactly one ownership",
        ),
        (
            lambda m: _item(m, "autogen").update(unassigned=False),
            "must be the literal true",
        ),
        (
            lambda m: (
                _item(m, "autogen").pop("unassigned"),
                _item(m, "autogen").update(owner="TBD"),
            ),
            "placeholder value is not allowed",
        ),
        (
            lambda m: _item(m, "autogen").update(readiness=["planned", "blocked"]),
            "must be exactly one readiness status",
        ),
        (
            lambda m: _item(m, "autogen").update(readiness="shipped"),
            "must be one of",
        ),
        (
            lambda m: _item(m, "autogen").update(kind="backend"),
            "must be one of",
        ),
        (
            lambda m: _item(m, "autogen").update(surprise="value"),
            "unexpected keys: surprise",
        ),
        (lambda m: _item(m, "autogen").pop("id"), "missing required keys: id"),
        (
            lambda m: _item(m, "crewai")["entrypoint"].pop("feature"),
            "missing required keys: feature",
        ),
        (
            lambda m: _item(m, "autogen").update(id="crewai"),
            "duplicate item id",
        ),
        (
            lambda m: m["items"].reverse(),
            "must be sorted by id",
        ),
        (
            lambda m: m.update(schema_version=99),
            "expected 1",
        ),
    ],
)
def test_validator_rejects_invalid_manifests(mutate, expected):
    manifest = deepcopy(load_manifest())
    mutate(manifest)

    with pytest.raises(RoadmapManifestError, match=expected):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "entry",
    [
        {
            "id": "x",
            "title": "X",
            "kind": "adapter",
            "readiness": "planned",
            "unassigned": True,
        },
        {
            "id": "x",
            "title": "X",
            "kind": "integration",
            "readiness": "blocked",
            "owner": "a-maintainer",
        },
        {
            "id": "x",
            "title": "X",
            "kind": "runtime",
            "readiness": "in_progress",
            "team": "core-team",
        },
    ],
)
def test_validate_entry_accepts_exactly_one_ownership_and_one_readiness(entry):
    assert validate_entry(entry) == "x"


@pytest.mark.parametrize(
    "entry",
    [
        # zero ownership selectors
        {"id": "x", "title": "X", "kind": "adapter", "readiness": "planned"},
        # two ownership selectors
        {
            "id": "x",
            "title": "X",
            "kind": "adapter",
            "readiness": "planned",
            "owner": "a",
            "team": "b",
        },
        # missing readiness
        {"id": "x", "title": "X", "kind": "adapter", "unassigned": True},
    ],
)
def test_validate_entry_rejects_wrong_ownership_or_readiness_cardinality(entry):
    with pytest.raises(RoadmapManifestError):
        validate_entry(entry)


def test_entrypoint_must_actually_raise_unavailable_feature_error():
    manifest = deepcopy(load_manifest())
    # Point a planned entry at a class that constructs without complaint.
    _item(manifest, "crewai")["entrypoint"] = {
        "module": "experia.core.exceptions",
        "attribute": "ExperiaError",
        "feature": "crewai",
    }

    with pytest.raises(RoadmapManifestError, match="appears operational"):
        assert_planned_entrypoints_unavailable(manifest)


def test_entrypoint_status_must_match_declared_readiness():
    manifest = deepcopy(load_manifest())
    # The crewai placeholder raises status="planned"; declare a mismatch.
    _item(manifest, "crewai")["readiness"] = "blocked"

    with pytest.raises(RoadmapManifestError, match="does not match|declares readiness"):
        assert_planned_entrypoints_unavailable(manifest)


def test_stale_readme_block_is_detected(tmp_path):
    manifest = load_manifest()
    stale = tmp_path / "README.md"
    stale.write_text(
        "<!-- BEGIN GENERATED ROADMAP STATUS -->\nstale\n"
        "<!-- END GENERATED ROADMAP STATUS -->\n",
        encoding="utf-8",
    )

    with pytest.raises(RoadmapManifestError, match="stale"):
        assert_readme_synced(manifest, stale)
