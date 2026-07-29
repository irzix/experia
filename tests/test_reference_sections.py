from copy import deepcopy
from pathlib import Path

import pytest

from scripts.api_compatibility import MINOR_DEPRECATION_RELEASES
from scripts.reference_sections import (
    DEFAULT_API_REFERENCE,
    DEFAULT_API_SNAPSHOT,
    DEFAULT_CONTRACT,
    DEFAULT_OUTBOUND_CONTRACT,
    DEFAULT_SCHEMA_SUPPORT,
    ReferenceContractError,
    assert_api_reference_synced,
    canonical_json,
    load_json,
    render_reference_sections,
    validate_contract,
    validate_schema_support,
)

ROOT = Path(__file__).resolve().parents[1]


def _rendered() -> str:
    return render_reference_sections(
        contract=load_json(DEFAULT_CONTRACT),
        snapshot=load_json(DEFAULT_API_SNAPSHOT),
        outbound=load_json(DEFAULT_OUTBOUND_CONTRACT),
        schema_support=load_json(DEFAULT_SCHEMA_SUPPORT),
        minimum_minor_releases=MINOR_DEPRECATION_RELEASES,
    )


def test_contract_source_is_canonical_and_valid():
    contract = load_json(DEFAULT_CONTRACT)

    validate_contract(contract, load_json(DEFAULT_API_SNAPSHOT))
    assert DEFAULT_CONTRACT.read_text(encoding="utf-8") == canonical_json(contract)


def test_api_reference_contains_the_generated_reference_block():
    assert_api_reference_synced()


def test_generated_sections_cover_every_requirement_area():
    block = _rendered()

    # 10.4 lifecycle contract for the four required operations.
    assert "### Lifecycle operations" in block
    for method in (
        "experia.memory.store.SQLiteStore.initialize",
        "experia.core.learner.Learner.flush",
        "experia.core.learner.Learner.shutdown",
        "experia.memory.store.SQLiteStore.close",
    ):
        assert method in block
    # 10.5 typed failure trigger/state/retry.
    assert "### Typed failure contract" in block
    assert "| Trigger | Typed error | Resulting state | Retry behavior |" in block
    # 2.3 / 10.6 network, credential, and outbound behavior.
    assert "### Network and credential summary" in block
    assert "pass-through" in block
    # 10.10 current-major stability.
    assert "### Public API stability" in block
    assert "current major version is **0**" in block
    # 3.6 schema support window.
    assert "### SQLite schema support window" in block
    assert "| 3 | Current | No migration required |" in block
    # 10.11 / 10.12 deprecation window.
    assert "### Deprecation window" in block
    assert f"at least {MINOR_DEPRECATION_RELEASES} consecutive minor releases" in block


def test_lifecycle_methods_and_typed_errors_come_from_the_api_snapshot():
    snapshot = load_json(DEFAULT_API_SNAPSHOT)
    contract = load_json(DEFAULT_CONTRACT)

    exports = snapshot["exports"]
    method_paths = {
        f"{export['target']}.{member['name']}"
        for export in exports
        for member in export.get("members", [])
    }
    exception_paths = {
        export["path"] for export in exports if export["kind"] == "exception"
    }

    assert all(
        operation["method"] in method_paths for operation in contract["operations"]
    )
    assert all(
        failure["typed_error"] in exception_paths for failure in contract["failures"]
    )


def test_schema_support_matches_the_migration_registry():
    from experia.memory.migrations import (
        CURRENT_SCHEMA_VERSION,
        SUPPORTED_SCHEMA_VERSIONS,
    )

    window = validate_schema_support(load_json(DEFAULT_SCHEMA_SUPPORT))

    assert window["current_schema_version"] == CURRENT_SCHEMA_VERSION
    assert list(window["supported_schema_versions"]) == list(SUPPORTED_SCHEMA_VERSIONS)


def test_deprecation_window_length_is_sourced_from_the_compatibility_policy():
    # The rendered window text must not hard-code a value that drifts from the
    # single compatibility-policy constant.
    assert (
        f"at least {MINOR_DEPRECATION_RELEASES} consecutive minor releases"
        in _rendered()
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda contract: contract["operations"][0].pop("postconditions"),
            "missing required keys: postconditions",
        ),
        (
            lambda contract: contract["operations"][0].update(idempotence="TBD"),
            "placeholder value is not allowed",
        ),
        (
            lambda contract: contract["operations"][0].update(
                method="experia.core.learner.Learner.does_not_exist"
            ),
            "is not a public method in the API snapshot",
        ),
        (
            lambda contract: contract["failures"][0].update(
                typed_error="experia.NotAnError"
            ),
            "is not an exported Experia exception",
        ),
        (
            lambda contract: [
                contract["operations"].remove(operation)
                for operation in list(contract["operations"])
                if operation["method"] == "experia.memory.store.SQLiteStore.close"
            ],
            "missing required lifecycle operations",
        ),
    ],
)
def test_validator_rejects_inconsistent_or_placeholder_contracts(mutate, expected):
    contract = deepcopy(load_json(DEFAULT_CONTRACT))
    snapshot = load_json(DEFAULT_API_SNAPSHOT)
    mutate(contract)

    with pytest.raises(ReferenceContractError, match=expected):
        validate_contract(contract, snapshot)


def test_schema_support_validator_rejects_drift_from_migrations():
    support = deepcopy(load_json(DEFAULT_SCHEMA_SUPPORT))
    support["support_window"]["supported_schema_versions"] = [0, 1]

    with pytest.raises(ReferenceContractError, match="does not match migrations"):
        validate_schema_support(support)


def test_api_reference_block_matches_freshly_rendered_sections():
    reference = DEFAULT_API_REFERENCE.read_text(encoding="utf-8")
    start = "<!-- BEGIN GENERATED LIFECYCLE AND CONTRACT REFERENCE -->"
    end = "<!-- END GENERATED LIFECYCLE AND CONTRACT REFERENCE -->"

    assert reference.count(start) == 1
    assert reference.count(end) == 1
    documented = start + reference.split(start, 1)[1].split(end, 1)[0] + end
    assert documented == _rendered()
