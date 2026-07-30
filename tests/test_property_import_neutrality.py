"""Property tests for import-time logging neutrality."""

import json
import os
import subprocess
import sys
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_IMPORT_SCRIPT = """
import importlib
import json
import logging
import sys

configuration = json.loads(sys.argv[1])


class NoOpHandler(logging.Handler):
    def emit(self, record):
        pass


handler_types = {
    "handler": NoOpHandler,
    "null": logging.NullHandler,
    "stream": logging.StreamHandler,
}
root = logging.getLogger()
configured_handlers = [
    handler_types[kind]() for kind in configuration["handler_kinds"]
]
root.handlers[:] = configured_handlers
root.setLevel(configuration["level"])
root.propagate = configuration["propagate"]

before_handlers = tuple(root.handlers)
before_handler_ids = tuple(id(handler) for handler in root.handlers)
before_handler_count = len(root.handlers)
before_propagation = root.propagate
before_level = root.level

importlib.import_module("experia")

after_handlers = tuple(root.handlers)
assert len(after_handlers) == before_handler_count
assert tuple(id(handler) for handler in after_handlers) == before_handler_ids
assert all(before is after for before, after in zip(before_handlers, after_handlers))
assert root.propagate is before_propagation
assert root.level == before_level
"""


# Feature: open-source-project-improvements, Property 24: Import is neutral to the root logger
@given(
    handler_kinds=st.lists(
        st.sampled_from(("handler", "null", "stream")),
        min_size=0,
        max_size=5,
    ),
    level=st.integers(min_value=0, max_value=60),
    propagate=st.booleans(),
)
def test_import_is_neutral_to_root_logger(
    handler_kinds: list[str], level: int, propagate: bool
) -> None:
    configuration = json.dumps(
        {
            "handler_kinds": handler_kinds,
            "level": level,
            "propagate": propagate,
        }
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(_REPOSITORY_ROOT), environment.get("PYTHONPATH", "")),
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT, configuration],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
