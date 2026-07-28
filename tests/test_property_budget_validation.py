"""Property tests for exact prompt-budget configuration validation."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from experia.context import BudgetUnit, ContextBuilder, PromptBudget
from experia.core.exceptions import ConfigurationError

_AMOUNT_VALUES = st.one_of(
    st.integers(min_value=-10_000, max_value=10_000),
    st.booleans(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=16),
    st.none(),
)
_UNIT_VALUES = st.one_of(
    st.sampled_from(tuple(BudgetUnit)),
    st.sampled_from(tuple(unit.value for unit in BudgetUnit)),
    st.text(max_size=16),
    st.integers(min_value=-2, max_value=2),
    st.booleans(),
    st.none(),
)
_DOCUMENTED_UNITS = {
    BudgetUnit.CHARACTERS,
    BudgetUnit.TOKENS,
    BudgetUnit.CHARACTERS.value,
    BudgetUnit.TOKENS.value,
}


class _UnusedTokenCounter:
    """A present counter that proves validation does not construct a prompt."""

    def count(self, text: str) -> int:
        raise AssertionError("budget validation must not measure prompt text")


def _documented_budget_predicate(
    amount: Any,
    unit: Any,
    *,
    token_counter_present: bool,
) -> bool:
    amount_is_valid = (
        isinstance(amount, int) and not isinstance(amount, bool) and amount >= 0
    )
    unit_is_valid = unit in _DOCUMENTED_UNITS
    token_counter_is_valid = (
        unit not in {BudgetUnit.TOKENS, BudgetUnit.TOKENS.value}
        or token_counter_present
    )
    return amount_is_valid and unit_is_valid and token_counter_is_valid


def _expected_invalid_parameter(
    amount: Any,
    unit: Any,
    *,
    token_counter_present: bool,
) -> str:
    if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
        return "amount"
    if unit not in _DOCUMENTED_UNITS:
        return "unit"
    assert unit in {BudgetUnit.TOKENS, BudgetUnit.TOKENS.value}
    assert not token_counter_present
    return "token_counter"


# Feature: open-source-project-improvements, Property 9: Budget validity is exact
@settings(max_examples=100, deadline=None)
@given(
    amount=_AMOUNT_VALUES,
    unit=_UNIT_VALUES,
    token_counter_present=st.booleans(),
)
def test_budget_acceptance_matches_documented_predicate_before_prompt_construction(
    amount: Any,
    unit: Any,
    token_counter_present: bool,
) -> None:
    """**Validates: Requirements 2.6**"""
    expected_valid = _documented_budget_predicate(
        amount,
        unit,
        token_counter_present=token_counter_present,
    )
    token_counter = _UnusedTokenCounter() if token_counter_present else None

    if expected_valid:
        budget = PromptBudget(amount, unit)
        builder = ContextBuilder(budget, token_counter)

        assert budget.amount == amount
        assert budget.unit is BudgetUnit(unit)
        assert builder.budget is budget
        assert builder.token_counter is token_counter
        return

    with pytest.raises(ConfigurationError) as raised:
        budget = PromptBudget(amount, unit)
        ContextBuilder(budget, token_counter)

    assert raised.value.feature == "prompt_context"
    assert raised.value.parameter == _expected_invalid_parameter(
        amount,
        unit,
        token_counter_present=token_counter_present,
    )
