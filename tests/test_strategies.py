"""Tests for the strategies module."""

from experia.improvement.strategies import _topic_overlap, merge_strategies
from experia.memory.models import Memory, MemoryType


def test_topic_overlap_same():
    assert _topic_overlap(
        "Always check logs before restarting services",
        "Check logs before restarting any service",
    )


def test_topic_overlap_different():
    assert not _topic_overlap(
        "Always check logs before restarting services",
        "Python performance is better with type hints",
    )


def test_topic_overlap_empty():
    assert not _topic_overlap("", "something")
    assert not _topic_overlap("something", "")


def test_merge_strategies_replaces_lower_confidence():
    existing = [
        Memory(
            content="Always check logs for errors before any restart",
            type=MemoryType.STRATEGY,
            confidence=0.5,
            importance=0.8,
        ),
    ]
    new_strategy = Memory(
        content="Always check logs before restarting services",
            type=MemoryType.STRATEGY,
            confidence=0.9,
            importance=1.0,
    )
    result = merge_strategies(existing, new_strategy)
    assert len(result) == 1
    assert result[0].content == "Always check logs before restarting services"


def test_merge_strategies_appends_when_no_overlap():
    existing = [
        Memory(
            content="Python performance tip",
            type=MemoryType.STRATEGY,
            confidence=0.8,
            importance=0.7,
        ),
    ]
    new_strategy = Memory(
        content="Always check logs before restarting services",
        type=MemoryType.STRATEGY,
        confidence=0.9,
        importance=1.0,
    )
    result = merge_strategies(existing, new_strategy)
    assert len(result) == 2
