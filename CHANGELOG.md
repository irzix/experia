# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-07-24

### Added
- **Semantic retrieval:** New pluggable `Embedder` protocol with a default `LiteLLMEmbedder`. Memories are embedded on write and ranked by cosine similarity (blended with importance); keyword search remains the fallback when no embedder is configured.
- **Feedback loop:** `Learner.reinforce(memory_id, success)` and `SQLiteStore.update_memory_feedback` update memory confidence via an exponential moving average and track `reinforcement_count` / `success_count`.
- **Memory lifecycle:** Near-duplicate memories are de-duplicated on write (existing memory reinforced instead of copied); `Learner.prune()` / `SQLiteStore.prune_expired()` sweep expired memories.
- **Non-blocking capture:** `record()` now persists the experience immediately and evaluates in the background by default; `Learner.flush()` awaits pending evaluations. Configurable via `background_evaluation`.

### Changed
- **SQLiteStore:** Reuses a single connection (WAL best-effort) with a write lock instead of reconnecting per operation, adds indexes on hot columns, and writes a lesson and its derived memory atomically in one transaction (`save_lesson_and_memory`). Added `get_memory`, `close`, and expiry-aware search.
- **Package metadata:** Added the reachable repository project URLs, exact Python 3.10–3.12 classifiers, SPDX `MIT` expression, packaged license declaration, and shipped `py.typed` marker; removed the unimplemented `openai` extra and the unverified author email.
- **Dependency change records:** Added a credential-free changelog gate that requires the affected extra plus exact old and new declarations whenever an existing optional dependency range changes. Python 3.10 uses the exactly pinned `tomli==2.2.1` development dependency for this check.
- Top-level `experia` package now exports `Learner`, `SQLiteStore`, `SimpleHeuristicEvaluator`, `Embedder`, `LiteLLMEmbedder`, `Memory`, and `MemoryType`.

### Docs
- README now has a **Project Status** section clarifying which backends are implemented versus planned (Postgres/pgvector, Mem0, Zep, distributed mode).
- Synchronized the canonical public API snapshot and generated API Reference constructor contracts with installed 0.7.0 behavior, including required `Learner.evaluator`, lifecycle methods, and the keyword-only embedding failure policy; added an offline asserted quickstart with linked automated evidence.

## [0.2.1] - 2026-07-24

### Added
- **PyPI Trusted Publishing (OIDC):** Added a GitHub Action workflow (`publish.yml`) to automatically publish the package to PyPI on new tags without requiring API tokens.

## [0.2.0] - 2026-07-24

### Added
- **LLM Integrations (`litellm`):** Added optional `[llm]` dependency to enable advanced cognitive features across any LLM provider.
- **Root Cause Analysis:** Experiences now store a `root_cause` field to track *why* an action succeeded or failed.
- **LLMEvaluator:** A new evaluator that uses an LLM to deeply analyze experiences, identify root causes, and extract high-confidence lessons.
- **Rule Generator:** Automatically consolidates strong lessons into permanent, high-priority `RULE` memories.
- **Reflection Engine:** A developer-controlled batch processing engine that analyzes past experiences and lessons to generate overarching `STRATEGY` memories.
- **MemoryType Enhancements:** Added `STRATEGY` to the memory types.

### Changed
- Refactored `SQLiteStore` to support seamless migrations (e.g., automatically adding the `root_cause` column to existing databases).
- Updated the main `Learner` API to accept `RuleGenerator` via dependency injection and added the `agent.reflect()` manual trigger method.

## [0.1.0] - 2026-07-24

### Added
- **Core Cognitive Loop:** Initial implementation of the `Learner` class to orchestrate experience capture and memory retrieval.
- **Async Architecture:** Fully asynchronous `asyncio`-based execution model.
- **Dependency Injection:** Defined strict `Protocol` interfaces for `Evaluator` and `MemoryStore`.
- **Memory Storage:** Initial `SQLiteStore` implementation using `aiosqlite`.
- **Heuristic Evaluation:** Basic `SimpleHeuristicEvaluator` for extracting lessons without an LLM.
- **Testing and CI:** Comprehensive test suite with `pytest` and GitHub Actions CI workflow.
