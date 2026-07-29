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

3. **Install the development dependencies from the fresh checkout.** These are the only setup commands you need. They are credential-free and are the exact commands every supported Python version (3.10, 3.11, and 3.12) runs in CI, so there are no hidden setup steps:

<!-- BEGIN SETUP COMMANDS -->
```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```
<!-- END SETUP COMMANDS -->

The `[dev]` extra installs every tool the Quality Gate uses (ruff, pytest, coverage, and the build backend), so no further installation is required.

## Quality Gate

The required local checks are the same credential-free commands run for every supported Python version in CI. Run them from the repository root after activating the virtual environment:

<!-- BEGIN QUALITY GATE COMMANDS -->
```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
python scripts/coverage_gate.py .coverage.granular.json
python scripts/artifact_gate.py --output-dir dist/quality-gate
```
<!-- END QUALITY GATE COMMANDS -->

The artifact gate builds the wheel and source distribution, installs the wheel in a clean environment, and smoke-tests it outside the source tree. No command requires an API key or external service credential.

To apply formatting before rerunning the gate:

```bash
python -m ruff format .
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
