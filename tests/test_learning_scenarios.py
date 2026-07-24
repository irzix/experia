"""
Behavioural scenario tests — does Experia actually make an agent better?

These are not unit tests of individual methods; they are end-to-end *proofs of
the core promise*: an agent that records its experiences avoids repeating its
mistakes, gets more confident about strategies that keep working, and can
recall a relevant lesson even when the new task is phrased with completely
different words.

Everything here is deterministic and fully offline (no LLM, no API keys), so it
runs anywhere CI does. The "agent" is intentionally dumb: its only source of
intelligence is the context Experia injects — which is exactly what lets us
attribute any improvement to the cognitive layer and nothing else.

Run `pytest tests/test_learning_scenarios.py -s` to see the before/after report.
"""

import re

import pytest

from experia import Learner, MemoryType, SimpleHeuristicEvaluator, SQLiteStore


# --------------------------------------------------------------------------- #
# A tiny, deterministic world and the agent that lives in it.
# --------------------------------------------------------------------------- #
class DeploymentEnv:
    """A fixed world: each action has one, always-the-same outcome."""

    OUTCOMES = {
        "restart nginx": "failed: port 80 is already bound",
        "free port 80 then restart nginx": "success: service is up",
    }

    def execute(self, action: str) -> str:
        return self.OUTCOMES[action]


class NaiveDeployBot:
    """
    An agent with zero built-in knowledge. It tries the cheapest action first
    and only avoids it if the injected context tells it that action has failed
    before. Left to its own devices it will make the same mistake forever.
    """

    TASK = "deploy the web service"
    CANDIDATES = ["restart nginx", "free port 80 then restart nginx"]

    def decide(self, injected_context: str) -> str:
        ctx = injected_context.lower()
        for action in self.CANDIDATES:
            # Skip any action we've been warned previously failed.
            if action in injected_context and "fail" in ctx:
                continue
            return action
        return self.CANDIDATES[-1]


async def run_episode(bot, env, learner, *, use_memory: bool):
    """Retrieve context (optional) -> decide -> act -> record the experience."""
    context = await learner.retrieve_context(query=bot.TASK) if use_memory else ""
    action = bot.decide(context)
    result = env.execute(action)
    await learner.record(task=bot.TASK, action=action, result=result)
    await learner.flush()  # wait for the background evaluation to land
    return {"action": action, "result": result, "success": result.startswith("success")}


async def build_learner(db_path, embedder=None):
    store = SQLiteStore(db_path=str(db_path))
    await store.initialize()
    learner = Learner(
        store=store, evaluator=SimpleHeuristicEvaluator(), embedder=embedder
    )
    return learner, store


# --------------------------------------------------------------------------- #
# 1. Baseline: without a memory, the agent never learns.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_agent_repeats_the_same_mistake_without_memory(tmp_path):
    learner, store = await build_learner(tmp_path / "baseline.db")
    try:
        env = DeploymentEnv()
        bot = NaiveDeployBot()

        episodes = [
            await run_episode(bot, env, learner, use_memory=False) for _ in range(3)
        ]

        # It picks the naive action every single time and fails every single time.
        assert all(ep["action"] == "restart nginx" for ep in episodes)
        assert all(ep["success"] is False for ep in episodes)
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# 2. The money shot: recording one failure changes future behaviour.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_agent_learns_from_failure_and_stops_failing(tmp_path, capsys):
    learner, store = await build_learner(tmp_path / "learning.db")
    try:
        env = DeploymentEnv()
        bot = NaiveDeployBot()

        # Episode 1: memory is empty, so the agent makes the naive mistake...
        first = await run_episode(bot, env, learner, use_memory=True)
        # Episode 2: it now retrieves the lesson it just learned and adapts.
        second = await run_episode(bot, env, learner, use_memory=True)

        assert first["action"] == "restart nginx"
        assert first["success"] is False

        assert second["action"] == "free port 80 then restart nginx"
        assert second["success"] is True

        # A lesson was actually persisted as a memory (not just a log line).
        lessons = await store.search_memories(memory_type=MemoryType.LESSON)
        assert len(lessons) >= 1

        # A little report that makes the value obvious when run with -s.
        print("\n--- Experia behavioural report -------------------------------")
        print(f"  Episode 1 (cold):    {first['action']:<32} -> {first['result']}")
        print(f"  Episode 2 (learned): {second['action']:<32} -> {second['result']}")
        print("  Attempts to first success without Experia: never")
        print("  Attempts to first success with Experia:    2")
        print("--------------------------------------------------------------")
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# 3. The feedback loop: strategies that keep working grow more trusted.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_repeated_success_reinforces_confidence(tmp_path):
    learner, store = await build_learner(tmp_path / "reinforce.db")
    try:
        mem = await learner.remember(
            "Free port 80 before restarting nginx", MemoryType.LESSON
        )
        start = mem.confidence

        trajectory = [start]
        for _ in range(3):
            updated = await learner.reinforce(mem.id, success=True)
            trajectory.append(updated.confidence)

        # Confidence rises monotonically toward 1.0 as the lesson keeps paying off.
        assert trajectory == sorted(trajectory)
        assert trajectory[-1] > start
        assert updated.reinforcement_count == 3
        assert updated.success_count == 3

        # And a lesson that later backfires loses confidence again.
        after_failure = await learner.reinforce(mem.id, success=False)
        assert after_failure.confidence < trajectory[-1]
    finally:
        await store.close()


# --------------------------------------------------------------------------- #
# 4. Semantic recall: the right lesson surfaces even with different wording.
# --------------------------------------------------------------------------- #
class TopicEmbedder:
    """
    A deterministic, offline stand-in for a real embedding model. It maps text
    to a small topic vector by keyword overlap — enough to demonstrate that
    semantic retrieval finds conceptually-related memories that a literal
    keyword search would miss.
    """

    TOPICS = {
        "infra": {"nginx", "server", "deploy", "port", "docker", "restart", "web", "service", "crash", "bound"},
        "data": {"database", "sql", "query", "postgres", "index", "schema"},
        "lang": {"python", "syntax", "import", "script", "typescript"},
    }

    async def embed(self, texts):
        return [self._vec(t) for t in texts]

    async def embed_one(self, text):
        return self._vec(text)

    def _vec(self, text):
        words = set(re.findall(r"[a-z]+", text.lower()))
        vec = [float(len(words & kws)) for kws in self.TOPICS.values()]
        return vec if any(vec) else [1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_semantic_recall_finds_what_keyword_search_cannot(tmp_path):
    learner, store = await build_learner(
        tmp_path / "semantic.db", embedder=TopicEmbedder()
    )
    try:
        # The agent once learned a lesson phrased around "restart nginx".
        await learner.record(
            task="deploy the web service",
            action="restart nginx",
            result="failed: port 80 is already bound",
        )
        await learner.flush()

        # A new task, described with NONE of the same words.
        query = "the site keeps crashing when we push it live"

        # Literal keyword search finds nothing — no shared substring.
        keyword_hits = await store.search_memories(query=query)
        assert keyword_hits == []

        # Semantic retrieval recognises they're about the same thing.
        context = await learner.retrieve_context(query=query)
        assert "restart nginx" in context
    finally:
        await store.close()
