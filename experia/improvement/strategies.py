"""Strategy generation and management for Experia.

Strategies are high-level behavioral patterns derived from reflection
over multiple experiences. They represent the agent's most durable
learned knowledge.
"""

from typing import List, Optional

from experia.core.logging import logger
from experia.memory.models import Memory, MemoryType


def merge_strategies(existing: List[Memory], new_strategy: Memory) -> List[Memory]:
    """Merge a new strategy into an existing list, replacing lower-confidence
    strategies on the same topic if the new one has higher confidence.

    Args:
        existing: List of existing STRATEGY memories.
        new_strategy: The new strategy to merge.

    Returns:
        Updated list of strategies.
    """
    merged = []
    replaced = False
    for mem in existing:
        if (
            mem.type == MemoryType.STRATEGY
            and _topic_overlap(mem.content, new_strategy.content)
            and mem.confidence <= new_strategy.confidence
        ):
            merged.append(new_strategy)
            replaced = True
            logger.info(
                f"Replaced strategy '{mem.content[:50]}...' "
                f"with '{new_strategy.content[:50]}...'"
            )
        else:
            merged.append(mem)

    if not replaced:
        merged.append(new_strategy)

    return merged


def _topic_overlap(a: str, b: str) -> bool:
    """Quick heuristic check if two strategy texts are on the same topic."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of",
        "in", "for", "on", "and", "or", "with", "before", "after",
    }
    significant_a = a_words - stop_words
    significant_b = b_words - stop_words
    if not significant_a or not significant_b:
        return False
    overlap = significant_a & significant_b
    return len(overlap) / min(len(significant_a), len(significant_b)) >= 0.3
