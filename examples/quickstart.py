"""Executable, offline Experia quickstart using only the base installation."""

import asyncio

from experia import Learner, MemoryType, SimpleHeuristicEvaluator, SQLiteStore


async def main() -> None:
    store = SQLiteStore(":memory:")
    await store.initialize()
    try:
        learner = Learner(
            store=store,
            evaluator=SimpleHeuristicEvaluator(),
        )

        experience = await learner.record(
            task="Deploy web app",
            action="Restart Nginx",
            result="failed with config syntax error",
            context={"attempt": 1},
        )
        assert experience.task == "Deploy web app"
        assert experience.action == "Restart Nginx"
        assert experience.result == "failed with config syntax error"
        assert experience.context == {"attempt": 1}

        persisted = await store.get_experience(experience.id)
        assert persisted is not None
        assert persisted.model_dump() == experience.model_dump()

        await learner.flush()
        memories = await store.search_memories(memory_type=MemoryType.LESSON)
        assert len(memories) == 1
        lesson_memory = memories[0]
        assert lesson_memory.type is MemoryType.LESSON
        assert lesson_memory.agent_role == "default"
        assert lesson_memory.confidence == 0.6
        assert lesson_memory.source == f"experience_{experience.id}"
        assert "Restart Nginx" in lesson_memory.content

        reinforced = await learner.reinforce(lesson_memory.id, success=True)
        assert reinforced is not None
        assert reinforced.reinforcement_count == 1
        assert reinforced.success_count == 1
        assert reinforced.confidence == 0.68
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
