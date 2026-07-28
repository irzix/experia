"""Shared managed dispatch for framework callback integrations."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, cast

from experia.core.exceptions import ConfigurationError
from experia.core.learner import Learner

CallbackMode: TypeAlias = Literal["background", "durable"]


def validate_callback_mode(value: str, *, feature: str) -> CallbackMode:
    """Validate callback persistence mode before integration state is changed."""

    if value not in {"background", "durable"}:
        raise ConfigurationError(
            "callback_mode must be 'background' or 'durable'.",
            feature=feature,
            parameter="callback_mode",
        )
    return cast(CallbackMode, value)


async def dispatch_callback_record(
    learner: Learner,
    mode: CallbackMode,
    *,
    task: str,
    action: str,
    result: str,
    context: dict[str, Any],
    on_success: Any = None,
    on_failure: Any = None,
) -> None:
    """Await durable records or register default background records with Learner."""

    if mode == "durable":
        try:
            await learner.record(
                task=task,
                action=action,
                result=result,
                context=context,
            )
        except Exception:
            if on_failure is not None:
                on_failure()
            raise
        if on_success is not None:
            on_success()
        return

    try:
        learner._submit_callback_record(
            task=task,
            action=action,
            result=result,
            context=context,
            on_success=on_success,
            on_failure=on_failure,
        )
    except Exception:
        if on_failure is not None:
            on_failure()
        raise
