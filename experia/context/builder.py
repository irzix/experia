from typing import List

from experia.memory.models import Memory


class ContextBuilder:
    """
    Transforms raw memories into structured, usable intelligence for the agent.
    """

    def format_for_prompt(self, memories: List[Memory]) -> str:
        """
        Takes a list of memories and formats them into a text string
        that can be injected into an LLM system prompt.
        """
        if not memories:
            return ""

        context_lines = ["--- User Context & Learned Experience ---"]

        # Group by type for better readability by the LLM
        grouped_memories = {}
        for mem in memories:
            grouped_memories.setdefault(mem.type.value, []).append(mem)

        for mem_type, type_memories in grouped_memories.items():
            context_lines.append(f"\n[{mem_type.upper()}]")
            for mem in type_memories:
                context_lines.append(f"- {mem.content}")

        context_lines.append("\n---------------------------------------")
        return "\n".join(context_lines)
