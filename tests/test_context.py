"""Tests for the ContextBuilder."""

from experia.context.builder import ContextBuilder
from experia.memory.models import Memory, MemoryType


def test_format_for_prompt_empty():
    builder = ContextBuilder()
    result = builder.format_for_prompt([])
    assert result == ""


def test_format_for_prompt_single():
    builder = ContextBuilder()
    memory = Memory(content="User likes Python", type=MemoryType.PREFERENCE, confidence=0.9)
    result = builder.format_for_prompt([memory])
    assert "[PREFERENCE]" in result
    assert "User likes Python" in result


def test_format_for_prompt_multiple_types():
    builder = ContextBuilder()
    memories = [
        Memory(content="Use type hints", type=MemoryType.RULE, confidence=1.0, importance=0.9),
        Memory(content="Python 3.10+", type=MemoryType.FACT, confidence=0.8, importance=0.5),
    ]
    result = builder.format_for_prompt(memories)
    assert "[RULE]" in result
    assert "[FACT]" in result
    assert "Use type hints" in result
    assert "Python 3.10+" in result
    assert "--- User Context & Learned Experience ---" in result
