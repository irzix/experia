from typing import Any, Dict, Optional


def default_messages_state_extractor(state: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Analyzes a standard LangGraph `MessagesState` (which contains a 'messages' list)
    and extracts the most recent Task, Action, and Result for learning.

    Returns:
        Dict with keys: 'task', 'action', 'result', or None if no tool execution occurred.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    task_str = ""
    action_str = ""
    result_str = ""

    # We need to find the latest ToolMessage
    # Then the AIMessage before it (the Action)
    # Then the HumanMessage before that (the Task)

    # Scan backwards
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        msg_type = getattr(msg, "type", type(msg).__name__.lower())

        if msg_type == "tool" and not result_str:
            # We found the latest tool result
            # Can be multiple tool messages, we just grab the content of the last one for simplicity
            # In a robust system, you'd aggregate all contiguous ToolMessages
            result_str = str(getattr(msg, "content", ""))

        elif msg_type == "ai" and result_str and not action_str:
            # This AI message should be the one that invoked the tool
            tool_calls = getattr(msg, "tool_calls", [])
            if tool_calls:
                action_str = f"Tool Calls: {tool_calls}"
            else:
                action_str = str(getattr(msg, "content", "No tool calls found"))

        elif msg_type == "human" and action_str and not task_str:
            # The prompt that caused this action
            task_str = str(getattr(msg, "content", ""))
            break  # We found all three components

    if task_str and action_str and result_str:
        return {
            "task": task_str,
            "action": action_str,
            "result": result_str,
        }

    return None
