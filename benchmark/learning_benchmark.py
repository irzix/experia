"""
Experia learning benchmark — does the cognitive layer actually change outcomes?

This is a deterministic, fully offline benchmark (no LLM, no API keys, no
network). It pits two identical agents against the same workday of ops tasks:

  * Baseline  — learning inactive. Faces each task fresh, every time.
  * Experia   — learning active. Records what happened, learns the lesson, and
                retrieves it before acting next time.

The two variants are built from the *same* scenario: identical task order,
action set, evaluator, embedder, and outcome rules. The only thing that varies
is whether learning is switched on. Each variant runs from clean persisted
state, and the run is driven by an explicit seed so the ordered outcomes are
byte-for-byte reproducible.

Run it (human-readable report)::

    python -m benchmark.learning_benchmark

Emit the machine-readable, stable-ordered report::

    python -m benchmark.learning_benchmark --json --output learning-report.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.manifest import (
    collect_benchmark_manifest,
    offline_network,
    serialize_manifest,
    validate_benchmark_manifest,
)
from benchmark.offline import deny_network
from experia import Learner, SimpleHeuristicEvaluator, SQLiteStore

# Keep the benchmark output clean — silence Experia's internal INFO logs.
logging.getLogger("experia").setLevel(logging.WARNING)

REPORT_SCHEMA = "experia.learning-benchmark-report.v1"
DEFAULT_SEED = 1729
DEFAULT_ROUNDS = 4  # how many times the agent revisits each task (a "workday")

# Identities recorded in the controlled-input block. Both variants are built
# from these exact factories so the comparison varies only learning activation.
EVALUATOR_IDENTITY = "SimpleHeuristicEvaluator"
EMBEDDER_IDENTITY = "none"


@dataclass(frozen=True)
class Task:
    """One recurring task with a tempting-but-wrong action and the right one."""

    name: str
    trap_action: str
    trap_result: str  # must read as a failure
    fix_action: str
    fix_result: str  # must read as a success

    @property
    def candidates(self) -> tuple[str, str]:
        # The agent tries the cheap/obvious action first.
        return (self.trap_action, self.fix_action)

    def outcome(self, action: str) -> tuple[str, bool]:
        """Deterministic outcome rule shared by every variant."""
        if action == self.fix_action:
            return self.fix_result, True
        return self.trap_result, False


TASKS: tuple[Task, ...] = (
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
)


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


@dataclass(frozen=True)
class BenchmarkVariant:
    """A single comparison arm. Only ``learning_enabled`` differs between arms."""

    name: str
    learning_enabled: bool


BASELINE = BenchmarkVariant(name="baseline", learning_enabled=False)
EXPERIA = BenchmarkVariant(name="experia", learning_enabled=True)
VARIANTS: tuple[BenchmarkVariant, ...] = (BASELINE, EXPERIA)


@dataclass(frozen=True)
class LearningScenario:
    """
    The fixed, shared world both variants run against. The seed selects a stable
    task order; everything else (action set, evaluator, embedder, outcome rules)
    is identical across variants by construction.
    """

    seed: int = DEFAULT_SEED
    rounds: int = DEFAULT_ROUNDS
    tasks: tuple[Task, ...] = TASKS

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, int):
            raise ValueError("rounds must be an integer")
        if self.rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if not self.tasks:
            raise ValueError("scenario requires at least one task")

    @property
    def task_order(self) -> tuple[Task, ...]:
        """A deterministic, seed-derived task order shared by every variant."""
        ordered = list(self.tasks)
        random.Random(self.seed).shuffle(ordered)
        return tuple(ordered)

    def make_evaluator(self) -> SimpleHeuristicEvaluator:
        return SimpleHeuristicEvaluator()

    def make_embedder(self) -> None:
        return None

    def controlled_inputs(self) -> dict[str, Any]:
        """The inputs that MUST be identical across baseline and Experia."""
        ordered = self.task_order
        return {
            "action_set": {task.name: list(task.candidates) for task in ordered},
            "embedder": EMBEDDER_IDENTITY,
            "evaluator": EVALUATOR_IDENTITY,
            "outcome_rules": {
                task.name: {
                    "failure": [task.trap_action, task.trap_result],
                    "success": [task.fix_action, task.fix_result],
                }
                for task in ordered
            },
            "rounds": self.rounds,
            "seed": self.seed,
            "task_order": [task.name for task in ordered],
        }


def _canonical_json(value: Any) -> str:
    """Compact, key-sorted JSON used for reproducible identity hashing."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def serialize_report(report: dict[str, Any]) -> str:
    """Stable, ordered serialization of a benchmark report."""
    return (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


async def open_clean_store() -> SQLiteStore:
    """Reset persisted state by opening a fresh, initialized in-memory store."""
    store = SQLiteStore(db_path=":memory:")
    await store.initialize()
    return store


async def persisted_state_is_clean(store: SQLiteStore) -> bool:
    """True when no experiences and no memories are persisted yet."""
    experiences = await store.get_recent_experiences(limit=1)
    memories = await store.search_memories()
    return not experiences and not memories


async def run_variant(
    scenario: LearningScenario, variant: BenchmarkVariant
) -> dict[str, Any]:
    """
    Run one comparison arm from clean persisted state and return its ordered
    outcomes. Learning activation is the only behaviour that depends on
    ``variant``; the task order, action set, evaluator, embedder, and outcome
    rules come from the shared scenario.
    """
    store = await open_clean_store()
    clean_start = await persisted_state_is_clean(store)
    learner = Learner(
        store=store,
        evaluator=scenario.make_evaluator(),
        embedder=scenario.make_embedder(),
    )
    bot = OpsBot()
    order = scenario.task_order
    tasks_per_round = len(order)

    episodes: list[dict[str, Any]] = []
    per_round_successes = [0] * scenario.rounds
    successes = 0
    repeated_mistakes = 0
    seen_failures: set[tuple[str, str]] = set()

    try:
        for round_index in range(scenario.rounds):
            for position, task in enumerate(order):
                context = (
                    await learner.retrieve_context(query=task.name)
                    if variant.learning_enabled
                    else ""
                )
                action = bot.decide(task, context)
                result, success = task.outcome(action)

                repeated = False
                if success:
                    per_round_successes[round_index] += 1
                    successes += 1
                else:
                    key = (task.name, action)
                    if key in seen_failures:
                        repeated_mistakes += 1
                        repeated = True
                    seen_failures.add(key)

                if variant.learning_enabled:
                    await learner.record(task=task.name, action=action, result=result)
                    await learner.flush()

                episodes.append(
                    {
                        "action": action,
                        "position": position,
                        "repeated_mistake": repeated,
                        "result": result,
                        "round": round_index + 1,
                        "success": success,
                        "task": task.name,
                    }
                )
    finally:
        await store.close()

    total = scenario.rounds * tasks_per_round
    return {
        "clean_start": clean_start,
        "episodes": episodes,
        "learning_enabled": variant.learning_enabled,
        "name": variant.name,
        "totals": {
            "episodes": total,
            "per_round_successes": per_round_successes,
            "repeated_mistakes": repeated_mistakes,
            "success_rate": successes / total,
            "successes": successes,
            "tasks_per_round": tasks_per_round,
        },
    }


async def run_benchmark(
    scenario: LearningScenario | None = None,
) -> dict[str, Any]:
    """Run every variant from clean state and build a deterministic report."""
    scenario = scenario or LearningScenario()
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        variants[variant.name] = await run_variant(scenario, variant)

    controlled = scenario.controlled_inputs()
    core = {
        "controlled_inputs": controlled,
        "rounds": scenario.rounds,
        "schema": REPORT_SCHEMA,
        "seed": scenario.seed,
        "variant_order": [variant.name for variant in VARIANTS],
        "variants": variants,
    }
    return {
        "controlled_inputs_id": _identity(controlled),
        "outcomes_id": _identity(core),
        **core,
    }


async def run_offline_benchmark(
    scenario: LearningScenario | None = None,
) -> dict[str, Any]:
    """Run the learning benchmark with external network access denied.

    The learning benchmark is fully offline: it uses a local heuristic
    evaluator, no embedder, and an in-memory store. Executing it inside
    :func:`benchmark.offline.deny_network` proves that classification is
    enforceable — any accidental external call fails loudly instead of quietly
    reaching a service or credential — while the deterministic report is
    identical to the unguarded :func:`run_benchmark` output.

    Requirement 11.9: the offline benchmark completes successfully with network
    access disabled and without external credentials.
    """
    with deny_network():
        return await run_benchmark(scenario)


def build_manifest(
    report: dict[str, Any],
    *,
    command: Sequence[str],
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Build the publication provenance manifest for a learning-benchmark report.

    The learning benchmark is fully offline: it uses a heuristic evaluator, no
    embedder, and reaches no network or credential. The manifest binds the
    report's controlled-input identity to the recorded environment, seed,
    evaluator, embedder, and execution command so the published numbers are
    reproducible and their provenance is auditable.
    """
    dataset_identity = {
        "controlled_inputs_id": report["controlled_inputs_id"],
        "dataset_id": report["controlled_inputs_id"],
        "kind": "learning-scenario",
        "outcomes_id": report["outcomes_id"],
        "report_schema": report["schema"],
    }
    return collect_benchmark_manifest(
        benchmark="learning",
        command=command,
        dataset_identity=dataset_identity,
        seed=report["seed"],
        evaluator=EVALUATOR_IDENTITY,
        embedder=EMBEDDER_IDENTITY,
        network=offline_network(),
        repository_root=repository_root,
    )


def _bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "#" * filled + "." * (width - filled)


def render_console(report: dict[str, Any]) -> str:
    """Human-readable rendering derived entirely from the machine report."""
    baseline = report["variants"]["baseline"]["totals"]
    experia = report["variants"]["experia"]["totals"]
    total = baseline["episodes"]
    tasks = experia["tasks_per_round"]
    rounds = report["rounds"]
    b_rate = 100 * baseline["success_rate"]
    e_rate = 100 * experia["success_rate"]

    lines = [
        "",
        f"Experia learning benchmark  ({tasks} tasks x {rounds} rounds = "
        f"{total} episodes, seed={report['seed']})",
        "=" * 66,
        f"{'Metric':<34}{'Baseline':>14}{'Experia':>14}",
        "-" * 66,
        f"{'Tasks completed successfully':<34}"
        f"{baseline['successes']:>10}/{total:<3}{experia['successes']:>10}/{total:<3}",
        f"{'Overall success rate':<34}{b_rate:>13.0f}%{e_rate:>13.0f}%",
        f"{'Mistakes repeated (avoidable)':<34}"
        f"{baseline['repeated_mistakes']:>14}{experia['repeated_mistakes']:>14}",
        "=" * 66,
        "",
        "Success rate per round (the learning curve)",
        "-" * 66,
    ]
    for r in range(rounds):
        b = 100 * baseline["per_round_successes"][r] / tasks
        e = 100 * experia["per_round_successes"][r] / tasks
        lines.append(
            f"  Round {r + 1}   baseline [{_bar(b)}] {b:>3.0f}%   "
            f"experia [{_bar(e)}] {e:>3.0f}%"
        )
    lines.extend(
        [
            "-" * 66,
            "",
            "Both variants start from clean state and share identical inputs "
            "(id: " + report["controlled_inputs_id"][:12] + ").",
            "Only learning activation differs. Baseline never improves; Experia "
            "fails a task at most once,",
            "extracts the lesson, and never repeats that mistake again.",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic, offline Experia learning benchmark."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Explicit seed for the reproducible task order.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="How many times each task is revisited.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the stable-ordered machine-readable report to stdout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the stable-ordered machine-readable report to a file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Write the machine-readable provenance manifest (commit, versions, "
            "dataset identity, seed, evaluator, embedder, command, and offline "
            "classification) to a file."
        ),
    )
    parser.add_argument(
        "--require-publishable",
        action="store_true",
        help=(
            "Validate the manifest as a publication gate and exit non-zero when "
            "it is incomplete or mismatched (requires --manifest)."
        ),
    )
    return parser


async def _main_async(arguments: argparse.Namespace) -> int:
    if arguments.require_publishable and arguments.manifest is None:
        raise SystemExit("--require-publishable requires --manifest")

    scenario = LearningScenario(seed=arguments.seed, rounds=arguments.rounds)
    # Enforce the offline classification: the run itself denies external network
    # access, so a misclassified benchmark fails loudly rather than silently
    # reaching a service or credential.
    report = await run_offline_benchmark(scenario)
    serialized = serialize_report(report)

    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")

    if arguments.manifest is not None:
        manifest = build_manifest(
            report,
            command=(
                sys.executable,
                "-m",
                "benchmark.learning_benchmark",
                *sys.argv[1:],
            ),
        )
        if arguments.require_publishable:
            validate_benchmark_manifest(
                manifest,
                expected_dataset_id=report["controlled_inputs_id"],
            )
        arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
        arguments.manifest.write_text(serialize_manifest(manifest), encoding="utf-8")

    if arguments.json:
        sys.stdout.write(serialized)
    else:
        print(render_console(report))
    return 0


def main() -> int:
    return asyncio.run(_main_async(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
