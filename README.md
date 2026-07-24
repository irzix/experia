# Experia AI

The open-source experience learning layer for AI agents.

Experia enables agents to learn from past actions, failures, and outcomes.
It provides experience capture, lesson extraction, behavioral improvement,
and long-term cognitive memory without replacing existing agent frameworks.

## Vision

Current AI agents start from zero on every interaction. Experia adds an **experience learning loop** around agents. It allows agents to remember what happened, understand why it worked or failed, and improve future decisions.

**Observation → Action → Result → Experience → Lesson → Memory → Better Future Action**

```mermaid
flowchart TD
    subgraph Agent Application
        Agent[AI Agent / LLM]
        Task[Execute Task]
        Developer[Developer / Cron Job]
        Agent -- Takes Action --> Task
    end

    subgraph Experia AI Cognitive Layer [Experia Async Cognitive Layer]
        Store[(MemoryStore)]
        Eval[LLM Evaluator\nRoot Cause Analysis]
        RuleGen[Rule Generator]
        Reflect[Reflection Engine\nBatch Analysis]
        Ctx[Context Builder]
        
        Task -- 1. await record() --> Store
        Store -- 2. Evaluate Outcome --> Eval
        Eval -- 3. Extract Lesson --> Store
        Eval -- 4. Consolidate --> RuleGen
        RuleGen -- 5. Generate RULE --> Store
        Developer -. 6. await reflect() .-> Reflect
        Reflect -- 7. Generate STRATEGY --> Store
        Store -- 8. await retrieve_context() --> Ctx
    end
    
    Ctx -- 9. Inject Knowledge --> Agent
```

## Integrations

Experia acts as a cognitive plugin. It does not replace your agent frameworks (like LangChain, AutoGen, CrewAI), it enhances them by managing long-term memory, learned experiences, user knowledge, and behavioral patterns.

## Getting Started

Experia uses an **asynchronous** and **pluggable** architecture. To use the advanced cognitive features (Root Cause Analysis, Rules, and Reflection), install with the `llm` extra:

```bash
pip install "experia[llm]"
```

*Note: You will need an `OPENAI_API_KEY` (or other litellm supported keys) exported in your environment.*

```python
import asyncio
from experia.core.learner import Learner
from experia.memory.store import SQLiteStore
from experia.experience.llm_evaluator import LLMEvaluator
from experia.improvement.rules import RuleGenerator


async def main():
    # 1. Dependency Injection setup
    store = SQLiteStore("my_agent.db")
    await store.initialize()

    # 2. Advanced LLM Cognition
    evaluator = LLMEvaluator(model="gpt-4o-mini")
    rule_generator = RuleGenerator(store=store, model="gpt-4o-mini")
    
    agent = Learner(store=store, evaluator=evaluator, rule_generator=rule_generator)

    # 3. Use the async API to record actions (Evaluates and generates rules automatically)
    await agent.record(
        task="Deploy web app",
        action="Restart Nginx",
        result="failed with config syntax error",
    )

    # 4. Retrieve memory context for your LLM's next action
    context = await agent.retrieve_context()
    print(context)

    # 5. Nightly/Scheduled Reflection
    # Analyzes the last N experiences to find patterns and extract global strategies.
    # Developer-controlled to manage AI reasoning costs.
    await agent.reflect(model="gpt-4o-mini", batch_size=50)


if __name__ == "__main__":
    asyncio.run(main())
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
