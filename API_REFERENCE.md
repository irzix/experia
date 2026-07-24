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
    agent_role="default",
)
```
- `store`: A class implementing the `MemoryStore` protocol (e.g. `SQLiteStore`).
- `evaluator`: A class implementing the `Evaluator` protocol (e.g. `LLMEvaluator`).
- `rule_generator`: (Optional) Extracts structured rules from lessons.
- `agent_role`: (Optional) The role of the agent. Used in Multi-Agent shared memory to isolate context.

**Methods**
- `async record(task: str, action: str, result: str, context: dict = None) -> ExperienceRecord`
  Saves a raw experience and kicks off asynchronous evaluation.
- `async retrieve_context(query: str = "", limit: int = 5) -> str`
  Fetches context formatted as a prompt-ready string, filtering for `agent_role` and global `STRATEGY` memories.
- `async reflect(model: str = "gpt-4o", batch_size: int = 50) -> None`
  Manually triggers reflection to consolidate lessons into a unified Strategy.
- `async remember(content: str, memory_type: MemoryType = MemoryType.FACT) -> Memory`
  Manually inject a memory.

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
- `confidence`: Certainty.
- `importance`: Relevance scoring.

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
