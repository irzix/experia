import asyncio
from typing import Any, Callable, Dict, Optional

from experia.core.learner import Learner
from experia.core.logging import logger
from experia.integrations.langgraph.utils import default_messages_state_extractor

try:
    from langchain_core.messages import SystemMessage
except ImportError:

    class SystemMessage:
        def __init__(self, content: str):
            self.content = content


class ExperiaContextNode:
    """
    A LangGraph Node that reads the latest user query from the State,
    fetches learned Lessons and Rules from the Experia Cognitive Layer,
    and injects them back into the State as a SystemMessage.

    Position in Graph: Entry Point (Before the Agent)
    """

    def __init__(self, agent: Learner, limit: int = 5):
        self.agent = agent
        self.limit = limit

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {}

        # Find the last human message
        query = ""
        for msg in reversed(messages):
            msg_type = getattr(msg, "type", type(msg).__name__.lower())
            if msg_type == "human":
                query = str(getattr(msg, "content", ""))
                break

        if not query:
            return {}

        logger.debug(f"ExperiaContextNode fetching context for query: {query[:50]}")
        context = await self.agent.retrieve_context(query=query, limit=self.limit)

        if not context.strip():
            return {}

        # Return a state update containing the learned context
        return {"messages": [SystemMessage(content=context)]}


class ExperiaLearningNode:
    """
    A LangGraph Node that reads the entire Execution State,
    extracts the Action taken and the Result, and records it to the Experia Cognitive Layer.

    Position in Graph: End of Tool Execution loop
    """

    def __init__(
        self,
        agent: Learner,
        extractor: Optional[
            Callable[[Dict[str, Any]], Optional[Dict[str, str]]]
        ] = None,
    ):
        self.agent = agent
        self.extractor = extractor or default_messages_state_extractor

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        extracted = self.extractor(state)

        if not extracted:
            logger.debug(
                "ExperiaLearningNode: No actionable experience found in state."
            )
            return {}

        task = extracted.get("task", "")
        action = extracted.get("action", "")
        result = extracted.get("result", "")

        if task and action and result:
            logger.info(
                "ExperiaLearningNode: Found experience, recording asynchronously."
            )
            # Fire and forget
            asyncio.create_task(
                self.agent.record(
                    task=task,
                    action=action,
                    result=result,
                    context={"source": "langgraph"},
                )
            )

        # This is a side-effect node, it doesn't change the graph state
        return {}
