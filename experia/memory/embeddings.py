"""
Pluggable embedding backends for semantic memory retrieval.

Experia stays dependency-light in local mode: embeddings are optional. If no
Embedder is configured, the store transparently falls back to keyword (LIKE)
search. When an Embedder is provided, memories are embedded on write and
retrieval ranks candidates by cosine similarity blended with importance.
"""

import math
from typing import List, Optional, Protocol, runtime_checkable

from experia.core.exceptions import EvaluationError
from experia.core.logging import logger

try:
    import litellm
except ImportError:
    litellm = None


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding backends."""

    async def embed(self, texts: List[str]) -> List[List[float]]: ...

    async def embed_one(self, text: str) -> List[float]: ...


class LiteLLMEmbedder(Embedder):
    """
    Default embedder backed by litellm, so any provider it supports works
    (OpenAI, Azure, Cohere, local, etc.). Requires the ``llm`` extra and an
    API key for the chosen model.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        if not litellm:
            raise EvaluationError(
                "litellm is not installed. Install experia[llm] to use LiteLLMEmbedder."
            )
        self.model = model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            response = await litellm.aembedding(model=self.model, input=texts)
            # litellm normalises to an OpenAI-style response object/dict.
            data = response["data"] if isinstance(response, dict) else response.data
            return [item["embedding"] for item in data]
        except Exception as e:
            logger.error(f"LiteLLMEmbedder failed to embed {len(texts)} text(s): {e}")
            raise EvaluationError(f"Embedding failed: {e}")

    async def embed_one(self, text: str) -> List[float]:
        vectors = await self.embed([text])
        return vectors[0] if vectors else []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity (keeps local mode free of native deps)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def most_similar(
    query: List[float],
    candidates: List[List[float]],
    threshold: float = 0.0,
) -> Optional[int]:
    """Return the index of the most similar candidate above ``threshold``."""
    best_idx: Optional[int] = None
    best_score = threshold
    for i, vec in enumerate(candidates):
        score = cosine_similarity(query, vec)
        if score >= best_score:
            best_score = score
            best_idx = i
    return best_idx
