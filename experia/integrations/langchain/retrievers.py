from __future__ import annotations

from typing import Any, List

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    AsyncCallbackManagerForRetrieverRun = Any
    CallbackManagerForRetrieverRun = Any
    Document = Any

    class BaseRetriever:
        pass
else:
    _LANGCHAIN_AVAILABLE = True


from experia.core.dependencies import require_optional_dependency
from experia.core.learner import Learner


class ExperiaLearningRetriever(BaseRetriever):
    """
    A LangChain Retriever that fetches past lessons, rules, and strategies
    from the Experia cognitive memory layer.

    It intentionally focuses on 'learning' (lessons/rules) rather than raw experiences,
    ensuring the agent retrieves learned cognitive behaviors.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "ExperiaLearningRetriever":
        require_optional_dependency(
            _LANGCHAIN_AVAILABLE,
            feature="ExperiaLearningRetriever",
            extra="experia[langchain]",
        )
        return super().__new__(cls)

    agent: Learner
    limit: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Synchronous retrieval is not natively supported since Experia is async-first."""
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "ExperiaLearningRetriever must be called asynchronously (use ainvoke or astream)."
            )
        return loop.run_until_complete(
            self._aget_relevant_documents(query, run_manager=run_manager)
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously fetches context from Experia and wraps it in LangChain Documents."""

        # Note: In a more advanced implementation, we might filter by MemoryType (LESSON, RULE, STRATEGY)
        # For now, we use the agent's built-in retrieve_context which already formats the best memories.

        # Since retrieve_context returns a formatted string, we can return it as a single Document,
        # or we could access agent.store directly to return multiple Documents.

        context_str = await self.agent.retrieve_context(query=query, limit=self.limit)

        if not context_str.strip():
            return []

        # Return as a single consolidated context document
        return [
            Document(
                page_content=context_str,
                metadata={"source": "experia_cognitive_memory"},
            )
        ]
