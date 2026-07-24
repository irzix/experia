# Experia API Reference

Welcome to the Experia API reference. Experia provides a core async cognitive layer designed to augment existing agents (LangChain, LangGraph, etc.) with long-term memory, experience learning, and rule generation.

---

## Core Classes

### `Learner`
The primary orchestration engine that connects Memory, Evaluation, and Reflection.

**Initialization**
```python
from experia.core.learner import Learner

agent = Learner(
    store=MemoryStore,
    evaluator=Optional[Evaluator],
    rule_generator=Optional[RuleGenerator],
    embedder=Optional[Embedder],
    agent_role="default",
    background_evaluation=True,
    dedup_threshold=0.95,
)
```
- `store`: A class implementing the `MemoryStore` protocol (e.g. `SQLiteStore`).
- `evaluator`: A class implementing the `Evaluator` protocol (e.g. `LLMEvaluator`).
- `rule_generator`: (Optional) Extracts structured rules from lessons.
- `embedder`: (Optional) A class implementing the `Embedder` protocol (e.g. `LiteLLMEmbedder`). Enables semantic retrieval and de-duplication. Without it, retrieval falls back to keyword search.
- `agent_role`: (Optional) The role of the agent. Used in Multi-Agent shared memory to isolate context.
- `background_evaluation`: (Optional, default `True`) When true, `record()` returns immediately and evaluation runs in the background.
- `dedup_threshold`: (Optional, default `0.95`) Cosine similarity above which a new memory is treated as a duplicate and reinforces the existing one.

**Methods**
- `async record(task: str, action: str, result: str, context: dict = None) -> ExperienceRecord`
  Persists a raw experience immediately and (by default) evaluates it in the background.
- `async flush() -> None`
  Awaits all pending background evaluations.
- `async retrieve_context(query: str = "", limit: int = 5) -> str`
  Fetches context formatted as a prompt-ready string, filtering for `agent_role` and global `STRATEGY` memories. Uses semantic ranking when an embedder is configured.
- `async reflect(model: str = "gpt-4o", batch_size: int = 50) -> None`
  Manually triggers reflection to consolidate lessons into a unified Strategy.
- `async remember(content: str, memory_type: MemoryType = MemoryType.FACT) -> Memory`
  Manually inject a memory (de-duplicated when an embedder is configured).
- `async reinforce(memory_id: UUID, success: bool) -> Optional[Memory]`
  Close the feedback loop: nudge a memory's confidence toward 1.0 (success) or 0.0 (failure).
- `async prune() -> int`
  Remove expired memories; returns the number pruned.

### `Embedder`
Protocol for pluggable embedding backends. `LiteLLMEmbedder` is the default implementation, backed by litellm (any supported provider).

```python
from experia.memory.embeddings import LiteLLMEmbedder

embedder = LiteLLMEmbedder(model="text-embedding-3-small")
```
- `async embed(texts: list[str]) -> list[list[float]]`
- `async embed_one(text: str) -> list[float]`

---

## Memory

### `SQLiteStore`
A local SQLite-based implementation of `MemoryStore`. It provides async, thread-safe access to memories using `aiosqlite`.

**Initialization**
```python
from experia.memory.store import SQLiteStore

store = SQLiteStore(db_path="agent_memory.db")
await store.initialize()
```

### Models

#### `ExperienceRecord`
Represents a raw execution cycle.
- `id`: UUID
- `task`: Initial intent.
- `action`: The tool or code executed.
- `result`: The output or error.
- `agent_role`: The agent that performed the action.
- `context`: Any runtime state.
- `created_at`: Datetime.

#### `Lesson`
Extracted by an `Evaluator`.
- `content`: The string lesson.
- `root_cause`: Optional extraction of why it failed.
- `agent_role`: The agent that learned this.
- `confidence`: 0.0 - 1.0.

#### `Memory`
A unified knowledge structure.
- `content`: The actual knowledge.
- `type`: `MemoryType` (`FACT`, `LESSON`, `RULE`, `STRATEGY`, `PREFERENCE`).
- `agent_role`: Role isolation.
- `confidence`: Certainty (updated by the feedback loop).
- `importance`: Relevance scoring.
- `reinforcement_count` / `success_count`: How often the memory has been validated by an outcome, and how many were successful.
- `embedding`: Optional vector of `content` used for semantic search (excluded from prompt serialization).
- `expires_at`: Optional expiry; swept by `prune`.

**Relevant `SQLiteStore` methods**
- `async search_memories(query="", memory_type=None, agent_role=None, limit=10, query_embedding=None, include_expired=False)` — keyword or semantic retrieval.
- `async find_similar_memory(embedding, memory_type=None, agent_role=None, threshold=0.95)` — nearest existing memory above threshold.
- `async update_memory_feedback(memory_id, success, alpha=0.2)` — EMA confidence update + counters.
- `async save_lesson_and_memory(lesson, memory)` — atomic write in one transaction.
- `async prune_expired() -> int` — delete expired memories.
- `async close()` — close the underlying connection.

---

## Integrations

### LangGraph

```python
from experia.integrations.langgraph.nodes import ExperiaContextNode, ExperiaLearningNode
```

**`ExperiaContextNode(agent: Learner, limit: int = 5)`**
- Position: Pre-agent.
- Function: Fetches context using `agent.retrieve_context` based on the latest human message and injects a `SystemMessage` into the graph State.

**`ExperiaLearningNode(agent: Learner, extractor: Optional[Callable] = None)`**
- Position: Post-tools.
- Function: Scans the state to extract `task`, `action`, and `result`, then calls `agent.record()`.

### LangChain

```python
from experia.integrations.langchain.callbacks import ExperiaCallbackHandler
from experia.integrations.langchain.retrievers import ExperiaLearningRetriever
```

**`ExperiaCallbackHandler(agent: Learner)`**
- A state machine that listens to `on_chain_start`, `on_tool_start`, and `on_tool_end` to construct cohesive experiences automatically.

**`ExperiaLearningRetriever(agent: Learner, search_kwargs: dict)`**
- Inherits from LangChain's `BaseRetriever`. Overrides `_aget_relevant_documents` to seamlessly integrate with LCEL workflows.
