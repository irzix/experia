"""Property tests for concurrent SQLite memory feedback."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import example, given, settings
from hypothesis import strategies as st

from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore
from tests.quality_profiles import (
    CONCURRENCY_MIN_EXAMPLES,
    CONCURRENCY_MIN_OPERATIONS,
    concurrency_max_operations,
)

_MAX_OPERATIONS = concurrency_max_operations(settings.get_current_profile_name())


@dataclass(frozen=True)
class FeedbackCase:
    """A valid initial memory state and its concurrent feedback batch."""

    reinforcement_count: int
    success_count: int
    confidence: float
    outcomes: tuple[bool, ...]


@st.composite
def feedback_cases(draw) -> FeedbackCase:
    reinforcement_count = draw(st.integers(min_value=0, max_value=1_000))
    success_count = draw(st.integers(min_value=0, max_value=reinforcement_count))
    return FeedbackCase(
        reinforcement_count=reinforcement_count,
        success_count=success_count,
        confidence=draw(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        outcomes=tuple(
            draw(
                st.lists(
                    st.booleans(),
                    min_size=CONCURRENCY_MIN_OPERATIONS,
                    max_size=_MAX_OPERATIONS,
                )
            )
        ),
    )


async def _exercise_concurrent_feedback(case: FeedbackCase) -> None:
    with TemporaryDirectory(prefix="experia-concurrent-feedback-") as directory:
        store = SQLiteStore(str(Path(directory) / "feedback.db"))
        await store.initialize()
        try:
            memory = Memory(
                content="Generated concurrent feedback memory",
                type=MemoryType.LESSON,
                confidence=case.confidence,
                reinforcement_count=case.reinforcement_count,
                success_count=case.success_count,
            )
            await store.save_memory(memory)

            all_ready = asyncio.Event()
            release = asyncio.Event()
            ready_count = 0

            async def apply_feedback(success: bool):
                nonlocal ready_count
                ready_count += 1
                if ready_count == len(case.outcomes):
                    all_ready.set()
                await release.wait()
                return await store.update_memory_feedback(memory.id, success=success)

            operations = [
                asyncio.create_task(apply_feedback(success))
                for success in case.outcomes
            ]
            await all_ready.wait()
            release.set()
            results = await asyncio.gather(*operations)

            persisted = await store.get_memory(memory.id)
            assert persisted is not None
            assert all(result is not None for result in results)
            assert persisted.reinforcement_count == (
                case.reinforcement_count + len(case.outcomes)
            )
            assert persisted.success_count == (case.success_count + sum(case.outcomes))
            assert 0.0 <= persisted.confidence <= 1.0
        finally:
            await store.close()


# Feature: open-source-project-improvements, Property 11: Concurrent feedback preserves counts and confidence
# **Validates: Requirements 3.1, 3.2, 3.3, 7.5**
@settings(max_examples=CONCURRENCY_MIN_EXAMPLES, deadline=None)
@given(case=feedback_cases())
@example(
    case=FeedbackCase(
        reinforcement_count=0,
        success_count=0,
        confidence=0.0,
        outcomes=(False, True),
    )
)
@example(
    case=FeedbackCase(
        reinforcement_count=1_000,
        success_count=1_000,
        confidence=1.0,
        outcomes=tuple(index % 2 == 0 for index in range(_MAX_OPERATIONS)),
    )
)
def test_concurrent_feedback_preserves_counts_and_confidence(
    case: FeedbackCase,
) -> None:
    asyncio.run(_exercise_concurrent_feedback(case))
