from typing import Any, Callable, Dict, Optional

from experia.core.dependencies import require_optional_dependency
from experia.core.learner import Learner
from experia.core.logging import IntegrationName, logger
from experia.integrations._candidates import finalize_experience_candidate
from experia.integrations._dispatch import (
    CallbackMode,
    dispatch_callback_record,
    validate_callback_mode,
)
from experia.integrations.langgraph.utils import default_messages_state_extractor

try:
    import langgraph as _langgraph
    from langchain_core.messages import SystemMessage
except ImportError:
    _LANGGRAPH_AVAILABLE = False

    class SystemMessage:
        def __init__(self, content: str):
            self.content = content
else:
    _LANGGRAPH_AVAILABLE = _langgraph is not None


class ExperiaContextNode:
    """
    A LangGraph Node that reads the latest user query from the State,
    fetches learned Lessons and Rules from the Experia Cognitive Layer,
    and injects them back into the State as a SystemMessage.

    Position in Graph: Entry Point (Before the Agent)
    """

    def __init__(self, agent: Learner, limit: int = 5):
        require_optional_dependency(
            _LANGGRAPH_AVAILABLE,
            feature="ExperiaContextNode",
            extra="experia[langgraph]",
        )
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
        *,
        callback_mode: CallbackMode = "background",
    ):
        require_optional_dependency(
            _LANGGRAPH_AVAILABLE,
            feature="ExperiaLearningNode",
            extra="experia[langgraph]",
        )
        validated_mode = validate_callback_mode(
            callback_mode,
            feature="ExperiaLearningNode",
        )
        self.agent = agent
        self.extractor = extractor or default_messages_state_extractor
        self.callback_mode = validated_mode

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        extracted = self.extractor(state)
        candidate = finalize_experience_candidate(
            extracted,
            integration=IntegrationName.LANGGRAPH,
            default_context={"source": "langgraph"},
        )

        if candidate is None:
            logger.debug(
                "ExperiaLearningNode: No actionable experience found in state."
            )
            return {}

        logger.info("ExperiaLearningNode: Found experience, recording callback.")
        await dispatch_callback_record(
            self.agent,
            self.callback_mode,
            task=candidate.task,
            action=candidate.action,
            result=candidate.result,
            context=candidate.context,
        )

        # This is a side-effect node, it doesn't change the graph state
        return {}
