"""Property test for benchmark reproducibility and comparison fairness.

This is Property 26 of the open-source-project-improvements design. It exercises
the deterministic, fully offline learning benchmark
(:mod:`benchmark.learning_benchmark`) and proves two guarantees for every valid
offline scenario:

  * **Reproducibility (Requirement 11.5)** — two runs driven by an identical
    seed, environment, initial (clean) persisted state, and inputs produce
    identical ordered outcomes and outcome counts. We assert this at the
    strongest possible level: the two stable-ordered serialized reports are
    byte-for-byte identical.

  * **Comparison fairness (Requirement 11.7)** — the baseline and Experia
    variants start from clean persisted state and share identical controlled
    inputs (task order, action set, evaluator, embedder, and outcome rules).
    They differ only in whether learning is activated.

Both runs execute under :func:`benchmark.offline.deny_network`, so the property
also demonstrates the benchmark stays offline and credential-free: any attempted
external access would fail loudly instead of quietly reaching a service.
"""

from __future__ import annotations

import asyncio
from typing import Any

from hypothesis import example, given, settings
from hypothesis import strategies as st

from benchmark.learning_benchmark import (
    DEFAULT_ROUNDS,
    DEFAULT_SEED,
    EMBEDDER_IDENTITY,
    EVALUATOR_IDENTITY,
    LearningScenario,
    run_offline_benchmark,
    serialize_report,
)

# Valid offline scenario inputs. ``LearningScenario`` accepts any non-bool
# integer seed and any positive non-bool integer round count; rounds stay small
# so a single generated example still runs both variants of two full benchmarks
# quickly while covering the reproducibility/fairness space.
_SEEDS = st.integers(min_value=-(2**31), max_value=2**31 - 1)
_ROUNDS = st.integers(min_value=1, max_value=3)


def _ordered_task_sequence(variant: dict[str, Any]) -> list[tuple[int, int, str]]:
    """The (round, position, task) sequence a variant actually executed."""
    return [
        (episode["round"], episode["position"], episode["task"])
        for episode in variant["episodes"]
    ]


def _assert_variant_obeys_controlled_inputs(
    variant: dict[str, Any], controlled: dict[str, Any]
) -> None:
    """Every episode of a variant draws from the shared, controlled world."""
    action_set = controlled["action_set"]
    outcome_rules = controlled["outcome_rules"]
    for episode in variant["episodes"]:
        task = episode["task"]
        action = episode["action"]
        # The action comes from the shared action set for this task.
        assert action in action_set[task]

        # The (result, success) pair obeys the shared, identical outcome rules.
        rule = outcome_rules[task]
        success_action, success_result = rule["success"]
        _failure_action, failure_result = rule["failure"]
        if action == success_action:
            assert episode["result"] == success_result
            assert episode["success"] is True
        else:
            assert episode["result"] == failure_result
            assert episode["success"] is False


async def _exercise_reproducibility_and_fairness(seed: int, rounds: int) -> None:
    # Two independent scenarios with identical seed, inputs, and (clean) initial
    # state. Both run offline with external network access denied.
    first = await run_offline_benchmark(LearningScenario(seed=seed, rounds=rounds))
    second = await run_offline_benchmark(LearningScenario(seed=seed, rounds=rounds))

    # --- Requirement 11.5: reproducible, byte-identical ordered outcomes ----- #
    assert serialize_report(first) == serialize_report(second)
    assert first == second
    assert first["outcomes_id"] == second["outcomes_id"]
    assert first["controlled_inputs_id"] == second["controlled_inputs_id"]

    baseline = first["variants"]["baseline"]
    experia = first["variants"]["experia"]

    # --- Requirement 11.7: fair comparison (only learning differs) ----------- #
    # Both variants start from clean persisted state.
    assert baseline["clean_start"] is True
    assert experia["clean_start"] is True

    # The single controlled difference is learning activation.
    assert baseline["learning_enabled"] is False
    assert experia["learning_enabled"] is True

    # Identical controlled inputs: same evaluator, embedder, seed, and rounds.
    controlled = first["controlled_inputs"]
    assert controlled["evaluator"] == EVALUATOR_IDENTITY
    assert controlled["embedder"] == EMBEDDER_IDENTITY
    assert controlled["seed"] == seed
    assert controlled["rounds"] == rounds

    # Identical task order and identical number of episodes across variants.
    assert _ordered_task_sequence(baseline) == _ordered_task_sequence(experia)
    assert len(baseline["episodes"]) == len(experia["episodes"])
    assert baseline["totals"]["tasks_per_round"] == experia["totals"]["tasks_per_round"]
    assert baseline["totals"]["episodes"] == experia["totals"]["episodes"]

    # Both variants draw from the identical action set and identical outcome
    # rules; only learning activation changes which actions get chosen.
    _assert_variant_obeys_controlled_inputs(baseline, controlled)
    _assert_variant_obeys_controlled_inputs(experia, controlled)


# Feature: open-source-project-improvements, Property 26: Benchmarks are reproducible and comparisons are fair
# **Validates: Requirements 11.5, 11.7**
@settings(max_examples=100, deadline=None)
@given(seed=_SEEDS, rounds=_ROUNDS)
@example(seed=DEFAULT_SEED, rounds=DEFAULT_ROUNDS)
@example(seed=0, rounds=1)
def test_benchmarks_are_reproducible_and_comparisons_are_fair(
    seed: int, rounds: int
) -> None:
    asyncio.run(_exercise_reproducibility_and_fairness(seed, rounds))
