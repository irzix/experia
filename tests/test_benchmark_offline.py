"""Focused tests for enforceable offline, credential-free benchmarking.

These cover the guarantees task 16.3 adds on top of the deterministic learning
benchmark (16.1) and the manifest layer (16.2):

  * a network-denial harness (:func:`benchmark.offline.deny_network`) that makes
    any attempted outbound network access fail loudly, and restores the socket
    module afterwards;
  * local, deterministic evaluator and embedder fixtures that are reproducible
    and reach no network or credential;
  * the offline learning benchmark completes successfully with network access
    disabled and without external credentials (Requirement 11.9);
  * the explicit, separate classification for benchmarks that legitimately
    require services or credential categories is preserved (Requirement 11.8).

Everything here is fully offline (no LLM, no network, no credentials).

Validates: Requirements 11.8, 11.9
"""

from __future__ import annotations

import socket

import pytest

from benchmark.learning_benchmark import (
    EMBEDDER_IDENTITY,
    EVALUATOR_IDENTITY,
    LearningScenario,
    build_manifest,
    run_benchmark,
    run_offline_benchmark,
)
from benchmark.manifest import offline_network, service_network
from benchmark.offline import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    OFFLINE_EMBEDDER_IDENTITY,
    OFFLINE_EVALUATOR_IDENTITY,
    DeterministicOfflineEmbedder,
    DeterministicOfflineEvaluator,
    NetworkAccessDeniedError,
    deny_network,
)
from experia import Learner, SQLiteStore
from experia.experience.models import ExperienceRecord

COMMAND = ("python", "-m", "benchmark.learning_benchmark", "--manifest", "m.json")


# --------------------------------------------------------------------------- #
# Network-denial harness: attempted external access fails loudly
# --------------------------------------------------------------------------- #
def test_deny_network_blocks_socket_connect() -> None:
    with deny_network():
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(NetworkAccessDeniedError):
                connection.connect(("example.com", 80))
        finally:
            connection.close()


def test_deny_network_blocks_socket_connect_ex() -> None:
    with deny_network():
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(NetworkAccessDeniedError):
                connection.connect_ex(("example.com", 80))
        finally:
            connection.close()


def test_deny_network_blocks_create_connection() -> None:
    with deny_network():
        with pytest.raises(NetworkAccessDeniedError):
            socket.create_connection(("example.com", 80), timeout=0.01)


def test_deny_network_restores_socket_after_block() -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    with deny_network():
        assert socket.socket.connect is not original_connect

    assert socket.socket.connect is original_connect
    assert socket.socket.connect_ex is original_connect_ex
    assert socket.create_connection is original_create_connection


def test_deny_network_restores_socket_even_when_block_raises() -> None:
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    with pytest.raises(RuntimeError, match="boom"):
        with deny_network():
            raise RuntimeError("boom")

    assert socket.socket.connect is original_connect
    assert socket.create_connection is original_create_connection


def test_deny_network_allows_connection_free_local_primitives() -> None:
    # socketpair does not use connect(), so local, connection-free primitives
    # keep working and the benchmark itself can still run under denial.
    with deny_network():
        left, right = socket.socketpair()
        try:
            left.sendall(b"offline")
            assert right.recv(16) == b"offline"
        finally:
            left.close()
            right.close()


# --------------------------------------------------------------------------- #
# Deterministic offline evaluator fixture
# --------------------------------------------------------------------------- #
def _record(result: str) -> ExperienceRecord:
    return ExperienceRecord(task="deploy", action="restart", result=result)


async def test_offline_evaluator_is_deterministic_for_failures() -> None:
    evaluator = DeterministicOfflineEvaluator()
    experience = _record("failed: port already bound")

    first = await evaluator.evaluate(experience)
    second = await evaluator.evaluate(experience)

    assert first is not None and second is not None
    assert first.content == second.content
    assert first.confidence == second.confidence == 0.6
    assert first.experience_id == experience.id


async def test_offline_evaluator_recognises_success_and_neutral() -> None:
    evaluator = DeterministicOfflineEvaluator()

    success = await evaluator.evaluate(_record("success: service is up"))
    assert success is not None
    assert success.confidence == 0.8

    neutral = await evaluator.evaluate(_record("nothing notable happened"))
    assert neutral is None


async def test_offline_evaluator_runs_under_network_denial() -> None:
    evaluator = DeterministicOfflineEvaluator()
    with deny_network():
        lesson = await evaluator.evaluate(_record("error: timeout"))
    assert lesson is not None
    assert evaluator.identity == OFFLINE_EVALUATOR_IDENTITY


# --------------------------------------------------------------------------- #
# Deterministic offline embedder fixture
# --------------------------------------------------------------------------- #
async def test_offline_embedder_is_deterministic_and_unit_length() -> None:
    embedder = DeterministicOfflineEmbedder()

    first = await embedder.embed_one("free port 80")
    second = await embedder.embed_one("free port 80")

    assert first == second
    assert len(first) == DEFAULT_EMBEDDING_DIMENSIONS
    norm = sum(value * value for value in first) ** 0.5
    assert norm == pytest.approx(1.0)


async def test_offline_embedder_separates_distinct_texts() -> None:
    embedder = DeterministicOfflineEmbedder()
    vectors = await embedder.embed(["restart nginx", "stream the file"])

    assert len(vectors) == 2
    assert vectors[0] != vectors[1]


async def test_offline_embedder_runs_under_network_denial() -> None:
    embedder = DeterministicOfflineEmbedder(dimensions=8)
    with deny_network():
        vector = await embedder.embed_one("no network needed")
    assert len(vector) == 8
    assert embedder.identity == OFFLINE_EMBEDDER_IDENTITY


@pytest.mark.parametrize("dimensions", [0, -1, True, 2.0])
def test_offline_embedder_rejects_invalid_dimensions(dimensions) -> None:
    with pytest.raises(ValueError, match="dimensions must be a positive integer"):
        DeterministicOfflineEmbedder(dimensions=dimensions)


# --------------------------------------------------------------------------- #
# The offline benchmark completes under network denial (Requirement 11.9)
# --------------------------------------------------------------------------- #
async def test_offline_benchmark_completes_with_network_disabled() -> None:
    scenario = LearningScenario(seed=2024, rounds=3)

    guarded = await run_offline_benchmark(scenario)
    unguarded = await run_benchmark(scenario)

    # Denying the network does not change the deterministic report.
    assert guarded == unguarded
    assert guarded["variants"]["experia"]["totals"]["successes"] > 0
    assert all(
        variant["clean_start"] is True for variant in guarded["variants"].values()
    )


async def test_offline_benchmark_manifest_is_classified_offline() -> None:
    report = await run_offline_benchmark(LearningScenario(seed=7, rounds=2))
    manifest = build_manifest(report, command=COMMAND)

    assert manifest["network"] == offline_network()
    assert manifest["network"]["offline"] is True
    assert manifest["network"]["required_services"] == []
    assert manifest["network"]["credential_categories"] == []
    # The offline benchmark uses only local, credential-free identities.
    assert manifest["evaluator"] == EVALUATOR_IDENTITY
    assert manifest["embedder"] == EMBEDDER_IDENTITY


async def test_learner_runs_fully_offline_with_local_fixtures() -> None:
    # A full record -> evaluate -> remember -> retrieve cycle driven entirely by
    # the local deterministic fixtures, with external network access denied.
    with deny_network():
        store = SQLiteStore(db_path=":memory:")
        await store.initialize()
        learner = Learner(
            store=store,
            evaluator=DeterministicOfflineEvaluator(),
            embedder=DeterministicOfflineEmbedder(),
            background_evaluation=False,
        )
        try:
            await learner.record(
                task="deploy the web service",
                action="restart nginx",
                result="failed: port 80 is already bound",
            )
            memories = await store.search_memories()
            context = await learner.retrieve_context(query="deploy the web service")
        finally:
            await store.close()

    assert memories, "the local evaluator should have produced a lesson memory"
    assert isinstance(context, str)


# --------------------------------------------------------------------------- #
# Preserve the separate classification for service/credential benchmarks (11.8)
# --------------------------------------------------------------------------- #
def test_service_classification_remains_distinct_from_offline() -> None:
    offline = offline_network()
    service = service_network(
        required_services=["evaluator-api"],
        credential_categories=["api-key"],
    )

    assert offline["offline"] is True
    assert service["offline"] is False
    # A non-offline benchmark documents exactly what it requires; the offline
    # path documents that it requires nothing.
    assert service["required_services"] == ["evaluator-api"]
    assert service["credential_categories"] == ["api-key"]
    assert offline["required_services"] == []
    assert offline["credential_categories"] == []


def test_service_classification_requires_documentation() -> None:
    with pytest.raises(ValueError, match="must document at least one"):
        service_network(required_services=[], credential_categories=[])
