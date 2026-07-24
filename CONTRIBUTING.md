# Contributing to Experia AI

First off, thank you for considering contributing to Experia! It's people like you that make Experia a powerful cognitive layer for the entire open-source AI community.

## Development Environment Setup

Experia uses modern Python tooling. To get started:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/irzix/experia.git
   cd experia
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies (including dev and LLM tools):**
   ```bash
   pip install -e ".[dev,llm]"
   ```

## Code Style and Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for extremely fast linting and code formatting.

Before committing, please run:
```bash
ruff format .
ruff check --fix .
```

## Running Tests

We use `pytest` for testing. Tests that require LLM API calls are mocked using `unittest.mock.AsyncMock` so that you do not need an active API key or internet connection to run the test suite.

To run all tests:
```bash
pytest
```

## Making a Pull Request (PR)

1. **Fork the repository** and create your branch from `main`.
2. **Write tests** for any new features or bug fixes.
3. **Ensure the test suite passes:** `pytest`
4. **Ensure your code is formatted:** `ruff format .`
5. **Update the CHANGELOG.md** if your change is user-facing.
6. **Submit your PR** with a clear description of the problem it solves or the feature it adds.

## Architectural Guidelines

- **Asynchronous First:** All core IO-bound operations (database calls, LLM requests) must be `async`.
- **Dependency Injection:** If you add a new storage backend (e.g., PostgreSQL, Mem0) or a new evaluation engine, it must implement the exact `Protocol` defined in `experia.core.interfaces`.
- **No Agent Execution:** Experia is a *cognitive layer*, not an agent orchestrator. We do not manage tool calling, workflows, or prompts. We only manage memory, experience, and learning.

## Questions?

If you are unsure if a feature aligns with the Experia roadmap, please open an Issue first to discuss it before writing code. We're happy to help!
