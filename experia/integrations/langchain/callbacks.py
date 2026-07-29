from typing import Any, Dict, Optional
from uuid import UUID

try:
    from langchain_core.callbacks import AsyncCallbackHandler
except ImportError:
    _LANGCHAIN_AVAILABLE = False

    class AsyncCallbackHandler:
        pass
else:
    _LANGCHAIN_AVAILABLE = True


from experia.core.dependencies import require_optional_dependency
from experia.core.learner import Learner
from experia.integrations._dispatch import CallbackMode
from experia.integrations.langchain.builder import ExperienceBuilder


class ExperiaCallbackHandler(AsyncCallbackHandler):
    """
    Native LangChain Callback Handler for Experia.
    Listens to LangChain execution events and routes them to the ExperienceBuilder,
    which compiles cohesive experiences and records them to the Learner.
    """

    def __init__(
        self,
        agent: Learner,
        *,
        callback_mode: CallbackMode = "background",
    ):
        require_optional_dependency(
            _LANGCHAIN_AVAILABLE,
            feature="ExperiaCallbackHandler",
            extra="experia[langchain]",
        )
        builder = ExperienceBuilder(agent=agent, callback_mode=callback_mode)
        super().__init__()
        self.builder = builder

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Only capture top-level chains to avoid nested spam
        if not parent_run_id:
            await self.builder.on_chain_start(run_id, serialized, inputs, **kwargs)

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Pass inputs if available, fallback to input_str
        actual_inputs = inputs if inputs is not None else {"input": input_str}
        await self.builder.on_tool_start(
            run_id, parent_run_id, serialized, actual_inputs, **kwargs
        )

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        await self.builder.on_tool_end(run_id, parent_run_id, str(output), **kwargs)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        await self.builder.on_tool_error(run_id, parent_run_id, error, **kwargs)

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not parent_run_id:
            await self.builder.on_chain_end(run_id, outputs, **kwargs)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not parent_run_id:
            await self.builder.on_chain_error(run_id, error, **kwargs)
