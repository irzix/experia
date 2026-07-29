# Experia API Reference

Experia 0.7.0 is an async experience-learning layer with a local SQLite backend
and optional LLM and framework integrations. The canonical machine-readable
public contract is [`api-snapshot.json`](api-snapshot.json); the generated
constructor tables below expose its parameter names, kinds, types, required
status, and defaults without changing runtime signatures.

## Installation and executable example

The offline core needs only the base package: `pip install experia`. The
canonical [offline quickstart](examples/quickstart.py) uses the package-level
imports `Learner`, `MemoryType`, `SimpleHeuristicEvaluator`, and `SQLiteStore`.
It initializes the store, supplies the required evaluator, records and reloads
an `ExperienceRecord`, flushes background evaluation, asserts the derived
`Memory`, reinforces it, and closes the store.

Optional installation extras are:

| Feature | Extra | Documented import | Installed constructor example |
|---|---|---|---|
| LiteLLM evaluation, embeddings, rules, and reflection | `experia[llm]` | `experia.experience.llm_evaluator.LLMEvaluator`, `experia.LiteLLMEmbedder`, `experia.improvement.rules.RuleGenerator` | [`examples/llm_extra.py`](examples/llm_extra.py) |
| LangChain callbacks and retrieval | `experia[langchain]` | `experia.integrations.langchain.callbacks.ExperiaCallbackHandler`, `experia.integrations.langchain.retrievers.ExperiaLearningRetriever` | [`examples/langchain_extra.py`](examples/langchain_extra.py) |
| LangGraph nodes | `experia[langgraph]` | `experia.integrations.langgraph.nodes.ExperiaContextNode`, `experia.integrations.langgraph.nodes.ExperiaLearningNode` | [`examples/langgraph_extra.py`](examples/langgraph_extra.py) |

The gate contract in [`examples/installed-examples.json`](examples/installed-examples.json)
declares the exact extras for these examples and the base
[`examples/quickstart.py`](examples/quickstart.py). Invoking an optional feature
without its extra raises `ConfigurationError`
naming the required extra. LLM-backed operations can additionally require the
credential category expected by the selected provider. Planned adapters are
not installation extras and raise `UnavailableFeatureError` when constructed.

<!-- BEGIN GENERATED OUTBOUND DATA CONTRACT -->
## Canonical outbound-data contract

Generated from [`outbound-data.json`](outbound-data.json) and checked
against every current production `protect_sink` call site. Request fields
are logical fields within provider payloads; fixed framing and control values
are listed alongside caller-derived values.

**No sanitizer configured:** every request and metadata field below is
value-equivalent **pass-through**. Experia still makes a defensive copy before
the sink, but it does not redact or transform values.

### User-provided embedder

- Sink: `experia.core.learner.Learner._call_embedder`
- Service/implementation: User-provided Embedder implementation
- Network: `implementation-dependent` — The injected Embedder may run locally or contact an implementation-selected service; Experia imposes no endpoint.
- Credential category: `implementation-dependent` — The injected Embedder decides whether it needs an API key, cloud identity, another credential, or no credential.
- Associated metadata emission: `not-emitted`

| Request field | Source | Sanitizer configured | No sanitizer |
|---|---|---|---|
| `text` | Caller-derived text passed to Embedder.embed_one | `sanitized` | `pass-through` |

| Associated metadata field | Source | Sanitizer configured | No sanitizer |
|---|---|---|---|
| `feature` | Experia feature identifier prepared with the protected request | `sanitized` | `pass-through` |
| `operation` | Experia operation identifier prepared with the protected request | `sanitized` | `pass-through` |
| `text_count` | Experia-computed number of submitted texts | `sanitized` | `pass-through` |

### LiteLLM embedding

- Sink: `experia.memory.embeddings.LiteLLMEmbedder.embed`
- Service/implementation: Embedding provider selected through LiteLLM
- Network: `provider-dependent` — LiteLLM may use a local provider or outbound network access to the endpoint selected by the configured model and provider.
- Credential category: `provider-dependent-api-credential` — The selected LiteLLM provider may require an API key or cloud identity; a local provider may require no credential.
- Associated metadata emission: `logging`

| Request field | Source | Sanitizer configured | No sanitizer |
|---|---|---|---|
| `input.texts[]` | Caller-derived text list submitted for embedding | `sanitized` | `pass-through` |
| `model` | Configured LiteLLM embedding model identifier | `pass-through` | `pass-through` |

| Associated metadata field | Source | Sanitizer configured | No sanitizer |
|---|---|---|---|
| `feature` | Experia feature identifier | `sanitized` | `pass-through` |
| `model` | Configured model identifier | `sanitized` | `pass-through` |
| `operation` | Experia operation identifier | `sanitized` | `pass-through` |
| `text_count` | Experia-computed number of submitted texts | `sanitized` | `pass-through` |

### LLM evaluation

- Sink: `experia.experience.llm_evaluator.LLMEvaluator.evaluate`
- Service/implementation: Completion provider selected through LiteLLM
- Network: `provider-dependent` — LiteLLM may use a local provider or outbound network access to the endpoint selected by the configured model and provider.
- Credential category: `provider-dependent-api-credential` — The selected LiteLLM provider may require an API key or cloud identity; a local provider may require no credential.
- Associated metadata emission: `logging`

| Request field | Source | Sanitizer configured | No sanitizer |
|---|---|---|---|
| `messages.system.content` | Experia-controlled evaluator instruction and response schema | `pass-through` | `pass-through` |
| `messages.system.role` | Experia-controlled system role | `pass-through` | `pass-through` |
| `messages.user.content.action` | ExperienceRecord action | `sanitized` | `pass-through` |
| `messages.user.content.context` | Complete nested ExperienceRecord context | `sanitized` | `pass-through` |
| `messages.user.content.framing` | Experia-controlled task, action, result, and context labels | `pass-through` | `pass-through` |
| `messages.user.content.result` | ExperienceRecord result | `sanitized` | `pass-through` |
| `messages.user.content.task` | ExperienceRecord task | `sanitized` | `pass-through` |
| `messages.user.role` | Experia-controlled user role | `pass-through` | `pass-through` |
| `model` | Configured LiteLLM completion model identifier | `pass-through` | `pass-through` |
| `response_format.type` | Experia-controlled JSON response format selector | `pass-through` | `pass-through` |
| `temperature` | Experia-controlled sampling temperature | `pass-through` | `pass-through` |

| Associated metadata field | Source | Sanitizer configured | No sanitizer |
|---|---|---|---|
| `experience_id` | Experia-generated experience identifier | `sanitized` | `pass-through` |
| `feature` | Experia feature identifier | `sanitized` | `pass-through` |
| `model` | Configured model identifier | `sanitized` | `pass-through` |
| `operation` | Experia operation identifier | `sanitized` | `pass-through` |

### Reflection

- Sink: `experia.reflection.consolidation.ReflectionEngine.reflect`
- Service/implementation: Completion provider selected through LiteLLM
- Network: `provider-dependent` — LiteLLM may use a local provider or outbound network access to the endpoint selected by the configured model and provider.
- Credential category: `provider-dependent-api-credential` — The selected LiteLLM provider may require an API key or cloud identity; a local provider may require no credential.
- Associated metadata emission: `logging`

| Request field | Source | Sanitizer configured | No sanitizer |
|---|---|---|---|
| `messages.system.content` | Experia-controlled reflection instruction | `pass-through` | `pass-through` |
| `messages.system.role` | Experia-controlled system role | `pass-through` | `pass-through` |
| `messages.user.content.experiences[].action` | Action from each retrieved ExperienceRecord | `sanitized` | `pass-through` |
| `messages.user.content.experiences[].result` | Result from each retrieved ExperienceRecord | `sanitized` | `pass-through` |
| `messages.user.content.experiences[].task` | Task from each retrieved ExperienceRecord | `sanitized` | `pass-through` |
| `messages.user.content.framing` | Experia-controlled batch and field labels | `pass-through` | `pass-through` |
| `messages.user.role` | Experia-controlled user role | `pass-through` | `pass-through` |
| `model` | Configured LiteLLM completion model identifier | `pass-through` | `pass-through` |
| `temperature` | Experia-controlled sampling temperature | `pass-through` | `pass-through` |

| Associated metadata field | Source | Sanitizer configured | No sanitizer |
|---|---|---|---|
| `batch_size` | Caller-selected retrieval batch limit | `sanitized` | `pass-through` |
| `experience_count` | Experia-computed number of transmitted experiences | `sanitized` | `pass-through` |
| `feature` | Experia feature identifier | `sanitized` | `pass-through` |
| `model` | Configured model identifier | `sanitized` | `pass-through` |
| `operation` | Experia operation identifier | `sanitized` | `pass-through` |

### Rule generation

- Sink: `experia.improvement.rules.RuleGenerator.consolidate_lesson`
- Service/implementation: Completion provider selected through LiteLLM
- Network: `provider-dependent` — LiteLLM may use a local provider or outbound network access to the endpoint selected by the configured model and provider.
- Credential category: `provider-dependent-api-credential` — The selected LiteLLM provider may require an API key or cloud identity; a local provider may require no credential.
- Associated metadata emission: `logging`

| Request field | Source | Sanitizer configured | No sanitizer |
|---|---|---|---|
| `messages.system.content` | Experia-controlled rule-generation instruction | `pass-through` | `pass-through` |
| `messages.system.role` | Experia-controlled system role | `pass-through` | `pass-through` |
| `messages.user.content.content` | Lesson content | `sanitized` | `pass-through` |
| `messages.user.content.framing` | Experia-controlled lesson and root-cause labels | `pass-through` | `pass-through` |
| `messages.user.content.root_cause` | Lesson root cause or the Experia fallback label | `sanitized` | `pass-through` |
| `messages.user.role` | Experia-controlled user role | `pass-through` | `pass-through` |
| `model` | Configured LiteLLM completion model identifier | `pass-through` | `pass-through` |
| `temperature` | Experia-controlled sampling temperature | `pass-through` | `pass-through` |

| Associated metadata field | Source | Sanitizer configured | No sanitizer |
|---|---|---|---|
| `experience_id` | Identifier of the lesson source experience | `sanitized` | `pass-through` |
| `feature` | Experia feature identifier | `sanitized` | `pass-through` |
| `lesson_id` | Experia-generated lesson identifier | `sanitized` | `pass-through` |
| `model` | Configured model identifier | `sanitized` | `pass-through` |
| `operation` | Experia operation identifier | `sanitized` | `pass-through` |

<!-- END GENERATED OUTBOUND DATA CONTRACT -->

<!-- BEGIN GENERATED LIFECYCLE AND CONTRACT REFERENCE -->
## Lifecycle, failure, and contract reference

Generated from the repository's validated sources ([`lifecycle-contract.json`](lifecycle-contract.json), [`api-snapshot.json`](api-snapshot.json), [`outbound-data.json`](outbound-data.json), and [`schema-support.json`](tests/fixtures/sqlite/schema-support.json)) so these tables stay consistent with the installed behavior.

### Lifecycle operations

Call order, postconditions, pending Background_Job state, and idempotence
for initialization, `flush()`, the shutdown operation, and store close.

| Operation | Call order | Postconditions | Pending background jobs | Idempotence |
|---|---|---|---|---|
| Store initialization (`experia.memory.store.SQLiteStore.initialize`) | Await once after constructing the store and before any read or write; Learner never calls it implicitly. | Opens the SQLite connection, enables foreign keys, best-effort WAL, and migrates the schema forward to the current version before returning. | Creates no Background_Job; no background work is pending on return. | Safe to await more than once; an already-open connection is reused and forward-only migration is a no-op at the current version. |
| Learner.flush() (`experia.core.learner.Learner.flush`) | Await after record() or remember() to complete outstanding evaluation, embedding, rule-generation, and reflection work; returns a FlushReport. | Captures a cutoff snapshot of every non-terminal Background_Job accepted before the call plus its causal descendants, returns only after each reaches a terminal state, and raises a typed EvaluationFailure that aggregates downstream failures while retaining persisted experiences. | Every Background_Job in the flush snapshot is terminal on return; jobs accepted after the cutoff stay pending and are unaffected. | Repeatable; each call opens a new cutoff generation, and a call with no outstanding work returns an empty report. |
| Shutdown operation (Learner.shutdown()) (`experia.core.learner.Learner.shutdown`) | Await once when the Learner is finished to stop accepting new Background_Job work; select the drain or cancel policy; returns a ShutdownReport. Learner.aclose() wraps this and can also close the store. | Atomically closes submissions and, under drain, awaits every accepted job or, under cancel, requests cancellation of every non-terminal job and awaits its terminal transition; drain surfaces an aggregated EvaluationFailure only after all jobs are terminal. | Every accepted Background_Job is terminal on return; submissions during and after shutdown are rejected with a typed LifecycleError. | Idempotent; the first caller selects the policy and concurrent or later callers await the same completion. The Learner is not reopened after shutdown; construct a new instance. |
| Store close (SQLiteStore.close()) (`experia.memory.store.SQLiteStore.close`) | Await when the store owner is finished, after any Learner.shutdown(); Learner.aclose(close_store=True) performs this ordering. Learner.shutdown() does not close the caller-owned store implicitly. | Waits for in-flight store operations to drain, closes the SQLite connection, and marks the store closed; later store operations raise a typed StorageError with operation "lifecycle". | Closing the store does not manage Learner Background_Job work; drain or cancel it with Learner.shutdown() or flush() first so no evaluation job races the connection close. | Idempotent; concurrent callers await one close completion, and calls after the first successful close are no-ops with persisted data unchanged. |

### Typed failure contract

Every documented failure names its trigger, the typed error raised, the
resulting state, and the retry behavior. Typed errors are the exception
classes recorded in [`api-snapshot.json`](api-snapshot.json).

| Trigger | Typed error | Resulting state | Retry behavior |
|---|---|---|---|
| Serialization or commit fails while saving an experience, memory, or feedback update. | `experia.StorageError` | The write transaction is rolled back; no partial record is persisted and no evaluation job is created. | Retry after correcting the data or database; a documented idempotency policy controls duplicate identifiers. |
| Inserting or committing the lesson-and-derived-memory pair fails. | `experia.StorageError` | Neither the lesson nor the derived memory is persisted; a previously persisted raw experience is retained. | Evaluation may be retried; the caller controls duplicate identifiers. |
| Stored JSON, enum, UUID, or timestamp data is malformed when loaded. | `experia.StorageError` | Only the read fails, carrying the table, record, and field context; persisted state is unchanged. | Retry after repairing or migrating the stored data. |
| A schema migration step or its commit fails. | `experia.StorageError` | The pre-migration schema and records are preserved through rollback. | Re-run initialization after resolving the cause; forward migration is idempotent. |
| A public operation is invoked on a closing or closed store. | `experia.StorageError` | Store state is unchanged. | Initialize before closing, or construct a new store after close. |
| Evaluation, embedding, rule generation, or reflection fails for a background job. | `experia.EvaluationFailure` | The persisted experience is retained; the job reaches a failure terminal state and is reported by flush() or shutdown(drain). | The caller may retry the specific operation identified by the failure context. |
| Background_Job work is submitted during or after shutdown. | `experia.LifecycleError` | No new job or write is created. | Construct a new Learner; the shut-down instance is not reopened. |
| An invalid limit, prompt budget, or configuration value is supplied. | `experia.ConfigurationError` | No I/O or mutation occurs. | Retry with a valid configuration value. |
| An optional feature is invoked without its installation extra. | `experia.ConfigurationError` | No Experia state changes before the error, which names the required extra. | Install the named extra and invoke the feature again. |
| A planned placeholder adapter or integration is constructed or invoked. | `experia.UnavailableFeatureError` | No Experia state changes. | Available only after a release marks the feature implemented. |
| The configured sanitizer fails before an external sink or log emission. | `experia.SanitizationError` | Neither the external request nor its log metadata is emitted, and caller-provided values are left unchanged. | Retry after fixing the sanitizer or data; automatic bypass is not allowed. |

### Network and credential summary

Per-feature network, credential, and metadata-emission behavior generated
from [`outbound-data.json`](outbound-data.json). Without a configured
sanitizer every transmitted and emitted field is **pass-through**. The
field-level tables above list each transmitted field and its sanitized/pass-through classification.

| Feature | Sink | Network requirement | Credential category | Metadata emission |
|---|---|---|---|---|
| User-provided embedder | `experia.core.learner.Learner._call_embedder` | `implementation-dependent` | `implementation-dependent` | `not-emitted` |
| LiteLLM embedding | `experia.memory.embeddings.LiteLLMEmbedder.embed` | `provider-dependent` | `provider-dependent-api-credential` | `logging` |
| LLM evaluation | `experia.experience.llm_evaluator.LLMEvaluator.evaluate` | `provider-dependent` | `provider-dependent-api-credential` | `logging` |
| Reflection | `experia.reflection.consolidation.ReflectionEngine.reflect` | `provider-dependent` | `provider-dependent-api-credential` | `logging` |
| Rule generation | `experia.improvement.rules.RuleGenerator.consolidate_lesson` | `provider-dependent` | `provider-dependent-api-credential` | `logging` |

### Public API stability

The current major version is **0** (package version `0.7.0`).
Within this major version, every supported import path and compatible signature recorded in [`api-snapshot.json`](api-snapshot.json) is preserved. Removals or narrowed signatures ship only under a greater major version accompanied by a migration guide.

### SQLite schema support window

Schema version 3 supports forward, upgrade-only migration from every version in the inclusive window 0 through 3. The machine-readable source of truth is [`schema-support.json`](tests/fixtures/sqlite/schema-support.json), which must agree with `experia.memory.migrations`.

| Schema version | Status | Forward migration |
|---:|---|---|
| 0 | Supported legacy | Migrates through version(s) 1, 2, 3 |
| 1 | Supported | Migrates through version(s) 2, 3 |
| 2 | Supported | Migrates through version(s) 3 |
| 3 | Current | No migration required |

### Deprecation window

A deprecated public API stays callable for at least 2 consecutive minor releases while emitting a deprecation warning that names its replacement. Removal before that window elapses is permitted only under a greater major version, and the migration guide identifies the removal and its replacement.

No public API is deprecated in the current major version.

<!-- END GENERATED LIFECYCLE AND CONTRACT REFERENCE -->

## Core classes and methods

### `Learner`

`Learner` is the primary orchestration class. Both `store: MemoryStore` and
`evaluator: Evaluator` are required. In particular, `evaluator` has no default
and is not `Optional`. `rule_generator` and `embedder` are optional and default
to `None`; `embedding_failure` is keyword-only and defaults to `"fallback"`.
Other defaults are recorded in the generated contract below.

- `async record(task: str, action: str, result: str, context: dict[str, Any] | None = None) -> ExperienceRecord` persists the raw experience before returning and schedules evaluation when background evaluation is enabled.
- `async flush() -> FlushReport` waits for the current background-work cutoff and reports its terminal outcomes.
- `async shutdown(policy: ShutdownPolicy | str = ShutdownPolicy.DRAIN) -> ShutdownReport` stops new work and drains or cancels accepted jobs according to the selected policy.
- `async aclose(policy: ShutdownPolicy | str = ShutdownPolicy.DRAIN, *, close_store: bool = False) -> None` performs shutdown and optionally closes the caller-owned store.
- `async retrieve_context(query: str = "", limit: int = 5) -> str` retrieves role-aware memories and formats complete untrusted-data blocks.
- `async remember(content: str, memory_type: MemoryType = MemoryType.FACT) -> Memory` stores an explicit memory and de-duplicates it when an embedder is configured.
- `async reinforce(memory_id: UUID, success: bool) -> Memory | None` updates confidence and feedback counters.
- `async prune() -> int` removes expired memories and returns the count removed.
- `async reflect(model: str = "gpt-4o", batch_size: int = 50) -> None` invokes optional LLM-backed reflection and therefore requires `experia[llm]` plus provider configuration.

### `SQLiteStore`

`SQLiteStore(db_path: str = "experia.db")` is the local async `MemoryStore`.
Call `await store.initialize()` before use and `await store.close()` when the
owner is finished. Relevant methods are `save_experience`, `get_experience`,
`save_memory`, `get_memory`, `save_lesson_and_memory`, `search_memories`,
`find_similar_memory`, `update_memory_feedback`, and `prune_expired`; their
canonical signatures are in the snapshot.

### Evaluators and embeddings

`SimpleHeuristicEvaluator` is available in the base installation and performs
no network calls. `LLMEvaluator(model: str = "gpt-4o")` and
`LiteLLMEmbedder(model: str = "text-embedding-3-small")` require
`experia[llm]`. `RuleGenerator(store: MemoryStore, model: str = "gpt-4o")`
also uses that extra when consolidation is invoked. No LLM-backed implementation
is implicitly selected by `Learner`.

## Models

`ExperienceRecord` requires keyword arguments `task`, `action`, and `result`.
`Lesson` requires `experience_id` and `content`. `Memory` requires `content` and
`type`. IDs and timestamps use factories, while all other model constructor
defaults and exposed types are listed in the generated tables.

`MemoryType` values are `FACT`, `PREFERENCE`, `LESSON`, `RULE`, `STRATEGY`, and
`EXPERIENCE`.

## Integrations

LangChain constructors require `experia[langchain]`. `ExperiaCallbackHandler`
requires `agent`. `ExperiaLearningRetriever` requires the keyword-only `agent`
argument and has keyword-only `limit=5`; inherited `name`, `tags`, and
`metadata` default to `None`.

LangGraph constructors require `experia[langgraph]`. `ExperiaContextNode`
requires `agent` and defaults `limit` to `5`. `ExperiaLearningNode` requires
`agent` and defaults `extractor` to `None`.

<!-- BEGIN GENERATED CONSTRUCTOR CONTRACTS -->
## Canonical public constructor contracts

Generated from [`api-snapshot.json`](api-snapshot.json). Do not edit this
section independently of the canonical snapshot. Required/default/type
metadata reflects the installed package; `Learner.evaluator` is required.

### `experia.core.exceptions.ConfigurationError`

Supported import paths: `experia.ConfigurationError`, `experia.core.ConfigurationError`, `experia.core.exceptions.ConfigurationError`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `*args` | `var_positional` | `object` | no | `variadic` |
| `feature` | `keyword_only` | `str \| None` | no | `None` |
| `parameter` | `keyword_only` | `str \| None` | no | `None` |
| `extra` | `keyword_only` | `str \| None` | no | `None` |

### `experia.core.exceptions.EvaluationFailure`

Supported import paths: `experia.EvaluationFailure`, `experia.core.EvaluationFailure`, `experia.core.exceptions.EvaluationFailure`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `message` | `positional_or_keyword` | `str \| None` | no | `None` |
| `job_id` | `keyword_only` | `uuid.UUID` | yes | `—` |
| `operation` | `keyword_only` | `str` | yes | `—` |
| `experience_id` | `keyword_only` | `uuid.UUID \| None` | no | `None` |
| `failures` | `keyword_only` | `collections.abc.Iterable[experia.core.exceptions.FailureDetail]` | no | `()` |

### `experia.core.exceptions.FailureDetail`

Supported import paths: `experia.FailureDetail`, `experia.core.FailureDetail`, `experia.core.exceptions.FailureDetail`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `job_id` | `positional_or_keyword` | `uuid.UUID` | yes | `—` |
| `operation` | `positional_or_keyword` | `str` | yes | `—` |
| `experience_id` | `positional_or_keyword` | `uuid.UUID \| None` | no | `None` |
| `error_type` | `positional_or_keyword` | `str` | no | `'evaluation_failure'` |

### `experia.core.exceptions.LifecycleError`

Supported import paths: `experia.LifecycleError`, `experia.core.LifecycleError`, `experia.core.exceptions.LifecycleError`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `message` | `positional_or_keyword` | `str \| None` | no | `None` |
| `state` | `keyword_only` | `str` | yes | `—` |
| `operation` | `keyword_only` | `str` | yes | `—` |

### `experia.core.exceptions.SanitizationError`

Supported import paths: `experia.SanitizationError`, `experia.core.SanitizationError`, `experia.core.exceptions.SanitizationError`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `message` | `positional_or_keyword` | `str \| None` | no | `None` |
| `path` | `keyword_only` | `collections.abc.Iterable[int \| str]` | no | `()` |
| `operation` | `keyword_only` | `str \| None` | no | `None` |

### `experia.core.exceptions.StorageError`

Supported import paths: `experia.StorageError`, `experia.core.StorageError`, `experia.core.exceptions.StorageError`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `*args` | `var_positional` | `object` | no | `variadic` |
| `operation` | `keyword_only` | `str \| None` | no | `None` |
| `table` | `keyword_only` | `str \| None` | no | `None` |
| `record_ids` | `keyword_only` | `collections.abc.Iterable[str \| uuid.UUID] \| str \| uuid.UUID` | no | `()` |
| `migration` | `keyword_only` | `str \| None` | no | `None` |
| `field` | `keyword_only` | `str \| None` | no | `None` |

### `experia.core.exceptions.UnavailableFeatureError`

Supported import paths: `experia.UnavailableFeatureError`, `experia.core.UnavailableFeatureError`, `experia.core.exceptions.UnavailableFeatureError`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `feature` | `positional_or_keyword` | `str` | yes | `—` |
| `status` | `keyword_only` | `str` | no | `'planned'` |
| `message` | `keyword_only` | `str \| None` | no | `None` |

### `experia.core.interfaces.Evaluator`

Supported import paths: `experia.core.interfaces.Evaluator`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `*args` | `var_positional` | `unannotated` | no | `variadic` |
| `**kwargs` | `var_keyword` | `unannotated` | no | `variadic` |

### `experia.core.interfaces.MemoryStore`

Supported import paths: `experia.core.interfaces.MemoryStore`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `*args` | `var_positional` | `unannotated` | no | `variadic` |
| `**kwargs` | `var_keyword` | `unannotated` | no | `variadic` |

### `experia.core.learner.Learner`

Supported import paths: `experia.Learner`, `experia.core.learner.Learner`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `store` | `positional_or_keyword` | `experia.core.interfaces.MemoryStore` | yes | `—` |
| `evaluator` | `positional_or_keyword` | `experia.core.interfaces.Evaluator` | yes | `—` |
| `rule_generator` | `positional_or_keyword` | `experia.improvement.rules.RuleGenerator \| None` | no | `None` |
| `embedder` | `positional_or_keyword` | `experia.memory.embeddings.Embedder \| None` | no | `None` |
| `agent_role` | `positional_or_keyword` | `str` | no | `'default'` |
| `background_evaluation` | `positional_or_keyword` | `bool` | no | `True` |
| `dedup_threshold` | `positional_or_keyword` | `float` | no | `0.95` |
| `embedding_failure` | `keyword_only` | `typing.Literal['fallback', 'raise']` | no | `'fallback'` |
| `data_protection` | `keyword_only` | `experia.security.protection.DataProtectionLayer \| None` | no | `None` |
| `lifecycle_observer` | `keyword_only` | `collections.abc.Callable[[<class 'experia.core.logging.LifecycleEvent'>], collections.abc.Awaitable[None] \| None] \| None` | no | `None` |

### `experia.experience.evaluator.SimpleHeuristicEvaluator`

Supported import paths: `experia.SimpleHeuristicEvaluator`, `experia.experience.evaluator.SimpleHeuristicEvaluator`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|

### `experia.experience.llm_evaluator.LLMEvaluator`

Supported import paths: `experia.experience.llm_evaluator.LLMEvaluator`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `model` | `positional_or_keyword` | `str` | no | `'gpt-4o'` |
| `data_protection` | `keyword_only` | `experia.security.protection.DataProtectionLayer \| None` | no | `None` |

### `experia.experience.models.ExperienceRecord`

Supported import paths: `experia.experience.models.ExperienceRecord`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `id` | `keyword_only` | `uuid.UUID` | no | `factory: uuid.uuid4` |
| `task` | `keyword_only` | `str` | yes | `—` |
| `action` | `keyword_only` | `str` | yes | `—` |
| `result` | `keyword_only` | `str` | yes | `—` |
| `agent_role` | `keyword_only` | `str` | no | `'default'` |
| `context` | `keyword_only` | `dict[str, typing.Any] \| None` | no | `factory: builtins.dict` |
| `created_at` | `keyword_only` | `datetime.datetime` | no | `factory: experia.experience.models.ExperienceRecord.<lambda>` |

### `experia.experience.models.Lesson`

Supported import paths: `experia.experience.models.Lesson`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `id` | `keyword_only` | `uuid.UUID` | no | `factory: uuid.uuid4` |
| `experience_id` | `keyword_only` | `uuid.UUID` | yes | `—` |
| `content` | `keyword_only` | `str` | yes | `—` |
| `agent_role` | `keyword_only` | `str` | no | `'default'` |
| `root_cause` | `keyword_only` | `str \| None` | no | `None` |
| `confidence` | `keyword_only` | `float` | no | `0.8` |
| `created_at` | `keyword_only` | `datetime.datetime` | no | `factory: experia.experience.models.Lesson.<lambda>` |

### `experia.improvement.rules.RuleGenerator`

Supported import paths: `experia.improvement.rules.RuleGenerator`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `store` | `positional_or_keyword` | `experia.core.interfaces.MemoryStore` | yes | `—` |
| `model` | `positional_or_keyword` | `str` | no | `'gpt-4o'` |
| `data_protection` | `keyword_only` | `experia.security.protection.DataProtectionLayer \| None` | no | `None` |

### `experia.integrations.langchain.callbacks.ExperiaCallbackHandler`

Supported import paths: `experia.integrations.langchain.callbacks.ExperiaCallbackHandler`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `agent` | `positional_or_keyword` | `experia.core.learner.Learner` | yes | `—` |
| `callback_mode` | `keyword_only` | `typing.Literal['background', 'durable']` | no | `'background'` |

### `experia.integrations.langchain.retrievers.ExperiaLearningRetriever`

Supported import paths: `experia.integrations.langchain.retrievers.ExperiaLearningRetriever`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `*args` | `var_positional` | `typing.Any` | no | `variadic` |
| `name` | `keyword_only` | `str \| None` | no | `None` |
| `tags` | `keyword_only` | `list[str] \| None` | no | `None` |
| `metadata` | `keyword_only` | `dict[str, typing.Any] \| None` | no | `None` |
| `agent` | `keyword_only` | `experia.core.learner.Learner` | yes | `—` |
| `limit` | `keyword_only` | `int` | no | `5` |

### `experia.integrations.langgraph.nodes.ExperiaContextNode`

Supported import paths: `experia.integrations.langgraph.nodes.ExperiaContextNode`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `agent` | `positional_or_keyword` | `experia.core.learner.Learner` | yes | `—` |
| `limit` | `positional_or_keyword` | `int` | no | `5` |

### `experia.integrations.langgraph.nodes.ExperiaLearningNode`

Supported import paths: `experia.integrations.langgraph.nodes.ExperiaLearningNode`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `agent` | `positional_or_keyword` | `experia.core.learner.Learner` | yes | `—` |
| `extractor` | `positional_or_keyword` | `collections.abc.Callable[[Dict[str, Any]], dict[str, str] \| None] \| None` | no | `None` |
| `callback_mode` | `keyword_only` | `typing.Literal['background', 'durable']` | no | `'background'` |

### `experia.memory.embeddings.Embedder`

Supported import paths: `experia.Embedder`, `experia.memory.embeddings.Embedder`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `*args` | `var_positional` | `unannotated` | no | `variadic` |
| `**kwargs` | `var_keyword` | `unannotated` | no | `variadic` |

### `experia.memory.embeddings.LiteLLMEmbedder`

Supported import paths: `experia.LiteLLMEmbedder`, `experia.memory.embeddings.LiteLLMEmbedder`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `model` | `positional_or_keyword` | `str` | no | `'text-embedding-3-small'` |
| `data_protection` | `keyword_only` | `experia.security.protection.DataProtectionLayer \| None` | no | `None` |

### `experia.memory.models.Memory`

Supported import paths: `experia.Memory`, `experia.memory.models.Memory`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `id` | `keyword_only` | `uuid.UUID` | no | `factory: uuid.uuid4` |
| `content` | `keyword_only` | `str` | yes | `—` |
| `type` | `keyword_only` | `experia.memory.models.MemoryType` | yes | `—` |
| `agent_role` | `keyword_only` | `str` | no | `'default'` |
| `confidence` | `keyword_only` | `float` | no | `0.8` |
| `importance` | `keyword_only` | `float` | no | `0.5` |
| `source` | `keyword_only` | `str \| None` | no | `None` |
| `metadata` | `keyword_only` | `dict[str, typing.Any] \| None` | no | `factory: builtins.dict` |
| `created_at` | `keyword_only` | `datetime.datetime` | no | `factory: experia.memory.models.Memory.<lambda>` |
| `updated_at` | `keyword_only` | `datetime.datetime` | no | `factory: experia.memory.models.Memory.<lambda>` |
| `expires_at` | `keyword_only` | `datetime.datetime \| None` | no | `None` |
| `reinforcement_count` | `keyword_only` | `int` | no | `0` |
| `success_count` | `keyword_only` | `int` | no | `0` |
| `embedding` | `keyword_only` | `list[float] \| None` | no | `None` |

### `experia.memory.store.SQLiteStore`

Supported import paths: `experia.SQLiteStore`, `experia.memory.store.SQLiteStore`

| Parameter | Kind | Exposed type | Required | Default |
|---|---|---|---:|---|
| `db_path` | `positional_or_keyword` | `str` | no | `'experia.db'` |

<!-- END GENERATED CONSTRUCTOR CONTRACTS -->
