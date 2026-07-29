"""Internal helpers for callable, metadata-rich API deprecations."""

from __future__ import annotations

import inspect
import re
import warnings
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")
_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def warn_deprecated(
    api: str,
    *,
    since: str,
    replacement: str,
    message: str | None = None,
) -> None:
    """Warn from the calling API while pointing at its caller."""
    warning_message = _deprecation_message(
        api,
        since=since,
        replacement=replacement,
        message=message,
    )
    warnings.warn(warning_message, DeprecationWarning, stacklevel=2)


def deprecated(
    *,
    since: str,
    replacement: str,
    message: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Keep a deprecated callable usable while warning with snapshot metadata.

    Functions, async functions, methods, and classes preserve their callable
    identity/signature. The attached metadata is consumed by the canonical API
    snapshot generator.
    """
    _validate_policy(since=since, replacement=replacement)

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        api = f"{target.__module__}.{target.__qualname__}"
        warning_message = _deprecation_message(
            api,
            since=since,
            replacement=replacement,
            message=message,
        )

        if inspect.isclass(target):
            original_init = target.__init__

            @wraps(original_init)
            def deprecated_init(instance: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(
                    warning_message,
                    DeprecationWarning,
                    stacklevel=2,
                )
                original_init(instance, *args, **kwargs)

            target.__init__ = deprecated_init
            wrapped = target
        elif inspect.iscoroutinefunction(target):

            @wraps(target)
            async def deprecated_async(*args: P.args, **kwargs: P.kwargs) -> Any:
                warnings.warn(
                    warning_message,
                    DeprecationWarning,
                    stacklevel=2,
                )
                return await target(*args, **kwargs)

            wrapped = deprecated_async
        else:

            @wraps(target)
            def deprecated_sync(*args: P.args, **kwargs: P.kwargs) -> Any:
                warnings.warn(
                    warning_message,
                    DeprecationWarning,
                    stacklevel=2,
                )
                return target(*args, **kwargs)

            wrapped = deprecated_sync

        setattr(wrapped, "__deprecated__", warning_message)
        setattr(wrapped, "__deprecated_since__", since)
        setattr(wrapped, "__deprecated_replacement__", replacement)
        return cast(Callable[P, R], wrapped)

    return decorate


def _validate_policy(*, since: str, replacement: str) -> None:
    if not isinstance(since, str) or _VERSION_PATTERN.fullmatch(since) is None:
        raise ValueError("since must use MAJOR.MINOR.PATCH format.")
    if not isinstance(replacement, str) or not replacement.strip():
        raise ValueError("replacement must name a supported API.")


def _deprecation_message(
    api: str,
    *,
    since: str,
    replacement: str,
    message: str | None,
) -> str:
    _validate_policy(since=since, replacement=replacement)
    if message is None:
        return f"{api} is deprecated since {since}; use {replacement} instead."
    rendered = message.strip()
    if replacement not in rendered:
        rendered = f"{rendered} Use {replacement} instead."
    return rendered


__all__ = ["deprecated", "warn_deprecated"]
