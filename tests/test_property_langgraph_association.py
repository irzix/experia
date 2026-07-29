"""Property coverage for deterministic LangGraph tool-result association."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from experia.integrations.langgraph.utils import default_messages_state_extractor


@dataclass(frozen=True)
class _AssociationCase:
    state: dict[str, Any]
    uses_ids: bool
    compatible: bool
    task: str
    declaration_names: tuple[str, ...]
    expected_result: str


@st.composite
def _association_cases(draw: st.DrawFn) -> _AssociationCase:
    uses_ids = draw(st.booleans())
    call_count = draw(st.integers(min_value=1, max_value=6))
    declaration_order = draw(st.permutations(tuple(range(call_count))))
    task = draw(
        st.text(
            alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=24,
        )
    )
    declaration_names = tuple(f"Tool{index}" for index in declaration_order)

    if uses_ids:
        call_ids = tuple(f"call-{index}" for index in declaration_order)
        tool_calls = [
            {"id": call_id, "name": name, "args": {}}
            for call_id, name in zip(call_ids, declaration_names)
        ]
        repeated_ids = draw(st.lists(st.sampled_from(call_ids), min_size=0, max_size=6))
        result_ids = draw(st.permutations((*call_ids, *repeated_ids)))
        compatible = draw(st.booleans())
        if not compatible:
            result_ids = (*result_ids, "foreign-call")
        tool_results = [
            {
                "type": "tool",
                "content": f"event-{event_index}:{call_id}",
                "tool_call_id": call_id,
            }
            for event_index, call_id in enumerate(result_ids)
        ]
        old_call = {
            "id": call_ids[0],
            "name": "OldTool",
            "args": {},
        }
        old_result = {
            "type": "tool",
            "content": "separated-old-result",
            "tool_call_id": call_ids[0],
        }
    else:
        compatible = True
        tool_calls = [{"name": name, "args": {}} for name in declaration_names]
        result_order = draw(st.permutations(tuple(range(call_count))))
        tool_results = [
            {
                "type": "tool",
                "content": f"event-{event_index}:result-{result_index}",
            }
            for event_index, result_index in enumerate(result_order)
        ]
        old_call = {"name": "OldTool", "args": {}}
        old_result = {
            "type": "tool",
            "content": "separated-old-result",
        }

    state = {
        "messages": [
            {"type": "human", "content": "old task"},
            {"type": "ai", "tool_calls": [old_call]},
            old_result,
            {"type": "human", "content": task},
            {"type": "ai", "tool_calls": tool_calls},
            *tool_results,
        ]
    }
    return _AssociationCase(
        state=state,
        uses_ids=uses_ids,
        compatible=compatible,
        task=task,
        declaration_names=declaration_names,
        expected_result="\n".join(result["content"] for result in tool_results),
    )


# Feature: open-source-project-improvements, Property 20: LangGraph association is ID-based and deterministic
@settings(max_examples=100, deadline=None)
@given(case=_association_cases())
def test_langgraph_association_is_id_based_and_deterministic(
    case: _AssociationCase,
) -> None:
    """**Validates: Requirements 6.3, 6.4**"""
    original_state = copy.deepcopy(case.state)
    first = default_messages_state_extractor(copy.deepcopy(case.state))
    second = default_messages_state_extractor(copy.deepcopy(case.state))

    assert first == second
    assert case.state == original_state

    if case.uses_ids and not case.compatible:
        assert first is None
        return

    assert first is not None
    assert first["task"] == case.task
    assert first["result"] == case.expected_result
    assert "separated-old-result" not in first["result"]

    declaration_positions = tuple(
        first["action"].index(name) for name in case.declaration_names
    )
    assert declaration_positions == tuple(sorted(declaration_positions))
