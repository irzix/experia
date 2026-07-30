"""Offline enforcement for reproducible, credential-free benchmarks.

Task 16.3 makes the offline benchmark *enforceably* credential- and
network-free. This module supplies the two pieces that guarantee it:

  * **Local, deterministic fixtures** — a :class:`DeterministicOfflineEvaluator`
    and a :class:`DeterministicOfflineEmbedder` that never touch a network or a
    credential. Identical inputs always yield identical outputs, so a full
    learning or retrieval run can be driven end-to-end with no external service.

  * **A network-denial harness** — :func:`deny_network` makes any attempted
    outbound socket connection fail loudly with :class:`NetworkAccessDeniedError`.
    An accidental external call during an "offline" benchmark is therefore
    caught immediately instead of silently succeeding. Connection-free local
    primitives (in-memory SQLite, asyncio's self-pipe, ``socket.socketpair``)
    are unaffected, so the benchmark itself still runs.

Benchmarks that legitimately need a network service or a credential category
are a **separate, explicitly classified** case. They are declared through
:func:`benchmark.manifest.service_network` and are never executed under
:func:`deny_network`. Nothing in this module weakens that separation: the
offline path denies the network, the non-offline path documents exactly what it
requires.
"""

from __future__ import annotations

import hashlib
import math
import socket
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Optional

from experia.core.interfaces import Evaluator
from experia.experience.models import ExperienceRecord, Lesson

# Identities recorded in a benchmark manifest when these fixtures are used. They
# are stable strings so the provenance record stays reproducible.
OFFLINE_EVALUATOR_IDENTITY = "DeterministicOfflineEvaluator"
OFFLINE_EMBEDDER_IDENTITY = "deterministic-offline-v1"

# A small, fixed embedding width keeps the fixture cheap while still exercising
# the store's vector path. It is deterministic, not tuned for quality.
DEFAULT_EMBEDDING_DIMENSIONS = 16


class NetworkAccessDeniedError(RuntimeError):
    """Raised when code attempts external network access under ``deny_network``.

    This is intentionally loud: an offline benchmark that tries to open a socket
    connection is misclassified, and we want the run to fail rather than quietly
    reach out to a service or leak a credential.
    """


@contextmanager
def deny_network() -> Iterator[None]:
    """Deny outbound network access for the duration of the ``with`` block.

    Any attempt to open an outbound socket connection (``socket.connect``,
    ``socket.connect_ex``, or :func:`socket.create_connection`) raises
    :class:`NetworkAccessDeniedError`. Local, connection-free primitives are
    left alone, so in-memory SQLite, the asyncio self-pipe, and
    ``socket.socketpair`` continue to work and the offline benchmark still runs.

    The patch is applied to the pure-Python ``socket.socket`` subclass and the
    module-level ``socket.create_connection`` and is fully restored on exit,
    including when the block raises, so it never leaks into surrounding code.
    """

    def _denied(*_args: object, **_kwargs: object) -> None:
        raise NetworkAccessDeniedError(
            "External network access is denied during the offline benchmark. "
            "This benchmark must run credential- and network-free; a benchmark "
            "that legitimately requires a service or credential must be "
            "classified as non-offline via benchmark.manifest.service_network."
        )

    sentinel = object()
    original_methods = {
        name: socket.socket.__dict__.get(name, sentinel)
        for name in ("connect", "connect_ex")
    }
    original_create_connection = socket.create_connection

    for name in original_methods:
        setattr(socket.socket, name, _denied)
    socket.create_connection = _denied  # type: ignore[assignment]
    try:
        yield
    finally:
        for name, value in original_methods.items():
            if value is sentinel:
                # The method was inherited from the C base, not defined on the
                # Python subclass; remove our shadow to restore that state.
                delattr(socket.socket, name)
            else:
                setattr(socket.socket, name, value)
        socket.create_connection = original_create_connection  # type: ignore[assignment]


class DeterministicOfflineEvaluator(Evaluator):
    """A local, deterministic evaluator for offline benchmarks.

    Emits a fixed lesson for recognisably failed or successful outcomes and no
    lesson otherwise. It performs no model call, reaches no network, uses no
    credential, and does not depend on wall-clock time or randomness, so the
    same experience always produces the same lesson content and confidence.
    """

    identity = OFFLINE_EVALUATOR_IDENTITY

    async def evaluate(self, experience: ExperienceRecord) -> Optional[Lesson]:
        lowered = experience.result.lower()
        if "fail" in lowered or "error" in lowered:
            content = (
                f"The action '{experience.action}' failed during "
                f"'{experience.task}'. Review prerequisites before retrying."
            )
            confidence = 0.6
        elif "success" in lowered:
            content = (
                f"The action '{experience.action}' succeeded for "
                f"'{experience.task}'. Prefer this strategy next time."
            )
            confidence = 0.8
        else:
            return None
        return Lesson(
            experience_id=experience.id,
            content=content,
            confidence=confidence,
        )


class DeterministicOfflineEmbedder:
    """A local, deterministic embedder for offline benchmarks.

    Produces a fixed-dimension, unit-length vector by expanding a SHA-256 digest
    of the input text into floats in ``[-1, 1]``. It is fully deterministic and
    offline: identical text always yields the identical vector, and no network
    or credential is ever used. It implements the :class:`experia.Embedder`
    protocol (``embed`` / ``embed_one``).
    """

    identity = OFFLINE_EMBEDDER_IDENTITY

    def __init__(self, *, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> None:
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions < 1
        ):
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        raw = bytearray()
        counter = 0
        # Deterministically stretch the digest to cover every dimension.
        while len(raw) < self.dimensions * 4:
            block = f"{counter}:{text}".encode("utf-8")
            raw.extend(hashlib.sha256(block).digest())
            counter += 1
        values = [
            (int.from_bytes(raw[index * 4 : index * 4 + 4], "big") / 0xFFFFFFFF) * 2 - 1
            for index in range(self.dimensions)
        ]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return values
        return [value / norm for value in values]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_one(self, text: str) -> list[float]:
        return self._vector(text)
