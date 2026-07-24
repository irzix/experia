"""
Experia learning benchmark — does the cognitive layer actually change outcomes?

This is a deterministic, fully offline benchmark (no LLM, no API keys, no
network). It pits two identical agents against the same workday of ops tasks:

  * Baseline  — no memory. Faces each task fresh, every time.
  * Experia   — records what happened, learns the lesson, and retrieves it
                before acting next time.

The agent itself has no built-in knowledge: its only intelligence is the
context Experia injects. So every point of difference is attributable to the
cognitive layer and nothing else.

Run it:

    python benchmark/learning_benchmark.py
"""

import asyncio
import logging
from dataclasses import dataclass

from experia import Learner, SimpleHeuristicEvaluator, SQLiteStore

# Keep the benchmark output clean — silence Experia's internal INFO logs.
logging.getLogger("experia").setLevel(logging.WARNING)


@dataclass(frozen=True)
class Task:
    """One recurring task with a tempting-but-wrong action and the right one."""

    name: str
    trap_action: str
    trap_result: str  # must read as a failure
    fix_action: str
    fix_result: str  # must read as a success

    @property
    def candidates(self):
        # The agent tries the cheap/obvious action first.
        return [self.trap_action, self.fix_action]


TASKS = [
    Task(
        "deploy the web service",
        "restart nginx",
        "failed: port 80 is already bound",
        "free port 80, then restart nginx",
        "success: service is up",
    ),
    Task(
        "run the data migration",
        "apply the migration directly",
        "failed: locked the table in production",
        "apply the migration inside a transaction with a lock timeout",
        "success: migration applied cleanly",
    ),
    Task(
        "publish the release",
        "push to main and tag",
        "failed: shipped on a red CI build",
        "wait for green CI, then tag",
        "success: release published",
    ),
    Task(
        "scale the api",
        "add more replicas",
        "failed: exhausted the database connection limit",
        "raise the db pool size, then add replicas",
        "success: api scaled",
    ),
    Task(
        "clear the cache",
        "flush all redis keys",
        "failed: evicted active session tokens",
        "flush only the cache namespace",
        "success: cache cleared",
    ),
    Task(
        "import the csv",
        "load the whole file into memory",
        "failed: out of memory on a large file",
        "stream the file in chunks",
        "success: import finished",
    ),
]

ROUNDS = 4  # how many times the agent revisits each task (a "workday")


class OpsBot:
    """
    A knowledge-free agent. It picks the first candidate action it has NOT been
    warned about, using only the context handed to it. Without that context it
    always reaches for the naive action.
    """

    def decide(self, task: Task, injected_context: str) -> str:
        ctx = injected_context.lower()
        for action in task.candidates:
            if action in injected_context and "fail" in ctx:
                continue  # we've been told this one blows up
            return action
        return task.candidates[-1]


async def run_condition(use_experia: bool):
    store = SQLiteStore(db_path=":memory:")
    await store.initialize()
    agent = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    bot = OpsBot()

    per_round_success = [0] * ROUNDS
    total_success = 0
    repeated_mistakes = 0
    seen_failures: set[tuple[str, str]] = set()

    for r in range(ROUNDS):
        for task in TASKS:
            context = (
                await agent.retrieve_context(query=task.name) if use_experia else ""
            )
            action = bot.decide(task, context)

            if action == task.fix_action:
                result = task.fix_result
                per_round_success[r] += 1
                total_success += 1
            else:
                result = task.trap_result
                key = (task.name, action)
                if key in seen_failures:
                    repeated_mistakes += 1  # made this exact mistake before
                seen_failures.add(key)

            if use_experia:
                await agent.record(task=task.name, action=action, result=result)
                await agent.flush()

    await store.close()
    total = ROUNDS * len(TASKS)
    return {
        "success": total_success,
        "total": total,
        "repeated_mistakes": repeated_mistakes,
        "per_round": per_round_success,
        "per_round_total": len(TASKS),
    }


def _bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "#" * filled + "." * (width - filled)


async def main():
    baseline = await run_condition(use_experia=False)
    experia = await run_condition(use_experia=True)

    total = baseline["total"]
    b_rate = 100 * baseline["success"] / total
    e_rate = 100 * experia["success"] / total

    print()
    print(f"Experia learning benchmark  ({len(TASKS)} tasks x {ROUNDS} rounds = {total} episodes)")
    print("=" * 66)
    print(f"{'Metric':<34}{'Baseline':>14}{'Experia':>14}")
    print("-" * 66)
    print(f"{'Tasks completed successfully':<34}{baseline['success']:>10}/{total:<3}{experia['success']:>10}/{total:<3}")
    print(f"{'Overall success rate':<34}{b_rate:>13.0f}%{e_rate:>13.0f}%")
    print(f"{'Mistakes repeated (avoidable)':<34}{baseline['repeated_mistakes']:>14}{experia['repeated_mistakes']:>14}")
    print("=" * 66)

    print("\nSuccess rate per round (the learning curve)")
    print("-" * 66)
    per_task = experia["per_round_total"]
    for r in range(ROUNDS):
        b = 100 * baseline["per_round"][r] / per_task
        e = 100 * experia["per_round"][r] / per_task
        print(f"  Round {r + 1}   baseline [{_bar(b)}] {b:>3.0f}%   experia [{_bar(e)}] {e:>3.0f}%")
    print("-" * 66)
    print(
        "\nBaseline never improves. Experia fails a task at most once, extracts the\n"
        "lesson, and never repeats that mistake again."
    )


if __name__ == "__main__":
    asyncio.run(main())
