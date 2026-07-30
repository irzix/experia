"""
Focused tests for the deterministic, offline learning benchmark runner.

These cover the three guarantees task 16.1 adds to ``benchmark.learning_benchmark``:

  * clean-state reset — every variant starts from empty persisted state;
  * controlled-input equality — baseline and Experia share identical task order,
    action set, evaluator, embedder, and outcome rules, and differ only in
    learning activation;
  * deterministic ordered output — an explicit seed produces byte-identical,
    stable-ordered serialization and identical ordered outcomes/counts.

Everything here is fully offline (no LLM, no network, no credentials).

Validates: Requirements 11.5, 11.7
"""

from __future__ import annotations

import json

import pytest

from benchmark.learning_benchmark import (
    BASELINE,
    EXPERIA,
    REPORT_SCHEMA,
    VARIANTS,
    LearningScenario,
    open_clean_store,
    persisted_state_is_clean,
    run_benchmark,
    run_variant,
    serialize_report,
)


# --------------------------------------------------------------------------- #
# Clean-state reset
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_open_clean_store_starts_from_empty_persisted_state() -> None:
    store = await open_clean_store()
    try:
        assert await persisted_state_is_clean(store) is True
        assert await store.get_recent_experiences(limit=5) == []
        assert await store.search_memories() == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_every_variant_reports_a_clean_start() -> None:
    scenario = LearningScenario(seed=7, rounds=2)
    for variant in VARIANTS:
        result = await run_variant(scenario, variant)
        assert result["clean_start"] is True


@pytest.mark.asyncio
async def test_learning_variant_persists_state_that_a_fresh_store_does_not_share() -> (
    None
):
    # The Experia arm records experiences, yet the next variant must not inherit
    # them: a freshly opened store is clean again.
    scenario = LearningScenario(seed=3, rounds=2)
    experia = await run_variant(scenario, EXPERIA)
    assert experia["totals"]["successes"] > 0  # it actually wrote experiences

    store = await open_clean_store()
    try:
        assert await persisted_state_is_clean(store) is True
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# Controlled-input equality (vary only learning activation)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_variants_differ_only_in_learning_activation() -> None:
    scenario = LearningScenario(seed=99, rounds=3)
    report = await run_benchmark(scenario)

    baseline = report["variants"]["baseline"]
    experia = report["variants"]["experia"]

    # The activation flag is the intended difference.
    assert baseline["learning_enabled"] is False
    assert experia["learning_enabled"] is True

    # Same task order, positions, and rounds — the controlled input space.
    baseline_sequence = [
        (episode["round"], episode["position"], episode["task"])
        for episode in baseline["episodes"]
    ]
    experia_sequence = [
        (episode["round"], episode["position"], episode["task"])
        for episode in experia["episodes"]
    ]
    assert baseline_sequence == experia_sequence
    assert len(baseline["episodes"]) == len(experia["episodes"])
    assert baseline["totals"]["tasks_per_round"] == experia["totals"]["tasks_per_round"]


def test_controlled_inputs_are_shared_and_seed_scoped() -> None:
    scenario = LearningScenario(seed=99, rounds=3)
    controlled = scenario.controlled_inputs()

    assert controlled["evaluator"] == "SimpleHeuristicEvaluator"
    assert controlled["embedder"] == "none"
    assert controlled["rounds"] == 3
    assert controlled["seed"] == 99
    # Task order is a stable permutation of the full task set.
    assert sorted(controlled["task_order"]) == sorted(controlled["action_set"])
    assert set(controlled["outcome_rules"]) == set(controlled["task_order"])
    # Every task exposes exactly its two candidate actions.
    assert all(len(actions) == 2 for actions in controlled["action_set"].values())


def test_variant_definitions_only_toggle_learning() -> None:
    assert BASELINE.learning_enabled is False
    assert EXPERIA.learning_enabled is True
    assert {variant.name for variant in VARIANTS} == {"baseline", "experia"}


# --------------------------------------------------------------------------- #
# Deterministic ordered output
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_identical_seed_produces_identical_ordered_report() -> None:
    scenario = LearningScenario(seed=2024, rounds=4)

    first = await run_benchmark(scenario)
    second = await run_benchmark(LearningScenario(seed=2024, rounds=4))

    assert first["schema"] == REPORT_SCHEMA
    # Identical ordered outcomes and outcome counts (Requirement 11.5).
    assert first == second
    assert first["outcomes_id"] == second["outcomes_id"]
    assert serialize_report(first) == serialize_report(second)


@pytest.mark.asyncio
async def test_serialization_is_stable_and_key_sorted() -> None:
    report = await run_benchmark(LearningScenario(seed=11, rounds=2))
    serialized = serialize_report(report)

    # Round-trips losslessly and is deterministically key-sorted.
    assert json.loads(serialized) == report
    assert serialized == serialize_report(json.loads(serialized))
    assert serialized == json.dumps(report, indent=2, sort_keys=True) + "\n"


@pytest.mark.asyncio
async def test_seed_changes_order_but_preserves_fair_outcome_counts() -> None:
    # A different seed reorders tasks (fresh identity) yet the fair comparison is
    # unchanged: baseline never learns, Experia stops repeating mistakes.
    one = await run_benchmark(LearningScenario(seed=1, rounds=4))
    two = await run_benchmark(LearningScenario(seed=2, rounds=4))

    assert (
        one["controlled_inputs"]["task_order"] != two["controlled_inputs"]["task_order"]
    )
    for report in (one, two):
        baseline = report["variants"]["baseline"]["totals"]
        experia = report["variants"]["experia"]["totals"]
        assert baseline["successes"] == 0
        assert baseline["repeated_mistakes"] > 0
        assert experia["repeated_mistakes"] == 0
        assert experia["successes"] > baseline["successes"]


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"seed": True}, "seed must be an integer"),
        ({"seed": 1.5}, "seed must be an integer"),
        ({"rounds": 0}, "rounds must be a positive integer"),
        ({"rounds": True}, "rounds must be an integer"),
    ],
)
def test_scenario_rejects_invalid_configuration(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        LearningScenario(**kwargs)
