from copy import deepcopy

import pytest

from scripts.outbound_data_contract import (
    DEFAULT_CONTRACT,
    OutboundContractError,
    canonical_json,
    discover_protected_sinks,
    load_contract,
    validate_contract,
)


def _feature(contract, feature_id):
    return next(
        feature for feature in contract["features"] if feature["id"] == feature_id
    )


def test_canonical_contract_is_stable_and_covers_every_current_external_sink():
    contract = load_contract()

    validate_contract(contract)

    assert DEFAULT_CONTRACT.read_text(encoding="utf-8") == canonical_json(contract)
    assert {feature["sink"] for feature in contract["features"]} == {
        sink.path for sink in discover_protected_sinks()
    }
    assert contract["default_without_sanitizer"] == "pass-through"
    assert all(
        field["without_sanitizer"] == "pass-through"
        for feature in contract["features"]
        for field in (*feature["request_fields"], *feature["metadata_fields"])
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda contract: _feature(contract, "llm_evaluator").pop(
                "credential_category"
            ),
            "missing required keys: credential_category",
        ),
        (
            lambda contract: _feature(contract, "llm_evaluator").update(
                network_details="TBD"
            ),
            "placeholder value is not allowed",
        ),
        (
            lambda contract: contract["features"].pop(),
            "missing current protected sink entries",
        ),
        (
            lambda contract: _feature(contract, "litellm_embedder")[
                "metadata_fields"
            ].pop(),
            "metadata fields do not match source",
        ),
        (
            lambda contract: _feature(contract, "external_embedder")["request_fields"][
                0
            ].update(without_sanitizer="sanitized"),
            "must preserve the contract default 'pass-through'",
        ),
    ],
)
def test_validator_rejects_incomplete_placeholder_or_false_contracts(mutate, expected):
    contract = deepcopy(load_contract())
    mutate(contract)

    with pytest.raises(OutboundContractError, match=expected):
        validate_contract(contract)
