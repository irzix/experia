"""Property tests for persisted-experience retention after downstream failure."""

from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from experia.core.exceptions import EvaluationFailure
from experia.core.work import AsyncWorkManager, OperationType
from experia.experience.models import ExperienceRecord
from experia.memory.store import SQLiteStore

_SAFE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=32,
)
_DOWNSTREAM_OPERATIONS = st.sampled_from(
    (
        OperationType.EVALUATION,
        OperationType.EMBEDDING,
        OperationType.RULE_GENERATION,
        OperationType.REFLECTION,
    )
)
_FAILURE_TYPES = st.sampled_from((ValueError, RuntimeError, LookupError))
_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False),
    _SAFE_TEXT,
)


@st.composite
def _persistable_experiences(draw: st.DrawFn) -> ExperienceRecord:
    return ExperienceRecord(
        id=draw(st.uuids()),
        task=draw(_SAFE_TEXT),
        action=draw(_SAFE_TEXT),
        result=draw(_SAFE_TEXT),
        agent_role=draw(_SAFE_TEXT),
        context={
            "scalar": draw(_JSON_SCALARS),
            "nested": {"values": draw(st.lists(_JSON_SCALARS, max_size=4))},
        },
    )


# Feature: open-source-project-improvements, Property 4: Downstream failure preserves the persisted experience
@pytest.mark.asyncio
@settings(max_examples=100, deadline=None)
@given(
    experience=_persistable_experiences(),
    operation=_DOWNSTREAM_OPERATIONS,
    failure_type=_FAILURE_TYPES,
    failure_message=_SAFE_TEXT,
)
async def test_downstream_failure_preserves_the_persisted_experience(
    experience: ExperienceRecord,
    operation: OperationType,
    failure_type: Callable[[str], Exception],
    failure_message: str,
) -> None:
    """**Validates: Requirements 1.5, 1.6**"""
    store = SQLiteStore(":memory:")
    await store.initialize()
    manager = AsyncWorkManager()

    try:
        await store.save_experience(experience)
        persisted_before_failure = await store.get_experience(experience.id)
        assert persisted_before_failure == experience

        async def fail_downstream() -> None:
            raise failure_type(failure_message)

        handle = manager.submit(
            operation,
            fail_downstream,
            experience_id=experience.id,
        )

        with pytest.raises(EvaluationFailure) as raised:
            await manager.flush()

        failure = raised.value
        assert type(failure) is EvaluationFailure
        assert failure.job_id == handle.job_id
        assert failure.operation == operation.value
        assert failure.experience_id == experience.id
        assert len(failure.failures) == 1
        assert failure.failures[0].job_id == handle.job_id
        assert failure.failures[0].operation == operation.value
        assert failure.failures[0].experience_id == experience.id
        assert failure.failures[0].error_type == failure_type.__name__

        persisted_after_failure = await store.get_experience(experience.id)
        assert persisted_after_failure == persisted_before_failure
        assert persisted_after_failure == experience
    finally:
        await store.close()
