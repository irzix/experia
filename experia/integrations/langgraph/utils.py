"""Deterministic extraction helpers for LangGraph message state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

_UNUSABLE_ID = object()


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _message_type(message: object) -> str:
    return str(_field(message, "type", type(message).__name__.lower())).lower()


def _association_id(value: object, field: str) -> object:
    identifier = _field(value, field)
    if identifier is None or (isinstance(identifier, str) and not identifier.strip()):
        return None
    try:
        hash(identifier)
    except TypeError:
        return _UNUSABLE_ID
    return identifier


def _latest_contiguous_tool_group(
    messages: Sequence[object],
) -> tuple[int, list[object], list[object]] | None:
    """Return the AI index, declarations, and newest contiguous result block."""

    tool_end = -1
    for index in range(len(messages) - 1, -1, -1):
        if _message_type(messages[index]) == "tool":
            tool_end = index
            break

    if tool_end < 0:
        return None

    tool_start = tool_end
    while tool_start > 0 and _message_type(messages[tool_start - 1]) == "tool":
        tool_start -= 1

    ai_index = tool_start - 1
    if ai_index < 0 or _message_type(messages[ai_index]) != "ai":
        return None

    declared = _field(messages[ai_index], "tool_calls", [])
    if (
        not isinstance(declared, Sequence)
        or isinstance(declared, (str, bytes, bytearray))
        or not declared
    ):
        return None

    return ai_index, list(declared), list(messages[tool_start : tool_end + 1])


def _associate_tool_results(
    tool_calls: Sequence[object], tool_results: Sequence[object]
) -> list[object] | None:
    """Validate one result block and return results in framework event order.

    Identifier-based association accepts multiple results for a declaration, but
    every declaration must be represented and every result must match the current
    AI message. Without identifiers, association uses only the deterministic
    positional fallback: scan both lists right-to-left, preserving declaration
    positions. That fallback is valid only for equal counts.
    """

    call_ids = [_association_id(call, "id") for call in tool_calls]
    result_ids = [_association_id(result, "tool_call_id") for result in tool_results]

    if _UNUSABLE_ID in call_ids or _UNUSABLE_ID in result_ids:
        return None

    call_ids_present = [identifier is not None for identifier in call_ids]
    result_ids_present = [identifier is not None for identifier in result_ids]

    if all(call_ids_present) and all(result_ids_present):
        if len(set(call_ids)) != len(call_ids):
            return None

        matched_ids: set[object] = set()
        declared_ids = set(call_ids)
        for result_id in result_ids:
            if result_id not in declared_ids:
                return None
            matched_ids.add(result_id)

        if matched_ids != declared_ids:
            return None
        return list(tool_results)

    if any(call_ids_present) or any(result_ids_present):
        return None

    if len(tool_calls) != len(tool_results):
        return None

    # Associate from the newest pair to the oldest, then restore event order.
    right_to_left = [
        result for _call, result in zip(reversed(tool_calls), reversed(tool_results))
    ]
    right_to_left.reverse()
    return right_to_left


def default_messages_state_extractor(state: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract the newest complete LangGraph task/action/result tool group.

    The newest contiguous ``ToolMessage`` block must immediately follow the
    declaring ``AIMessage``. IDs, when available, are mandatory on both sides and
    must match only declarations from that AI message. The ID-less fallback is
    limited to equal declaration/result counts, preventing cross-run guesses.
    """

    messages = state.get("messages", [])
    if (
        not isinstance(messages, Sequence)
        or isinstance(messages, (str, bytes, bytearray))
        or not messages
    ):
        return None

    group = _latest_contiguous_tool_group(messages)
    if group is None:
        return None

    ai_index, tool_calls, tool_results = group
    associated_results = _associate_tool_results(tool_calls, tool_results)
    if associated_results is None:
        return None

    task = ""
    for index in range(ai_index - 1, -1, -1):
        message = messages[index]
        if _message_type(message) == "human":
            task = str(_field(message, "content", ""))
            break

    action = f"Tool Calls: {tool_calls}"
    result = "\n".join(
        str(_field(message, "content", "")) for message in associated_results
    )

    if not task or not action or not result:
        return None

    return {
        "task": task,
        "action": action,
        "result": result,
    }
