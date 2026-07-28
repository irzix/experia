from collections import OrderedDict
from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel

from experia.core.exceptions import SanitizationError
from experia.security import DataProtectionLayer


class NestedPayload(BaseModel):
    credential: str
    values: list[int]


class RecordingSanitizer:
    def __init__(self) -> None:
        self.paths: list[tuple[str | int, ...]] = []

    def sanitize(self, value: Any, *, path: tuple[str | int, ...]) -> Any:
        self.paths.append(path)
        if isinstance(value, str):
            return f"protected:{'/'.join(map(str, path))}"
        return value


def test_pass_through_returns_value_equivalent_deep_copy():
    model = NestedPayload(credential="secret", values=[1, 2])
    payload = OrderedDict(
        task="deploy",
        context={
            "items": [model, ("tuple-value",)],
            "labels": {"one", "two"},
            "frozen": frozenset({1, 2}),
        },
    )

    protected = DataProtectionLayer().protect_external(payload)

    assert protected == payload
    assert protected is not payload
    assert protected["context"] is not payload["context"]
    assert protected["context"]["items"] is not payload["context"]["items"]
    protected_model = protected["context"]["items"][0]
    assert protected_model is not model
    assert protected_model.values is not model.values


def test_sanitizer_receives_every_nested_leaf_path_and_copied_values():
    sanitizer = RecordingSanitizer()
    model = NestedPayload(credential="model-secret", values=[3])
    payload = {
        "task": "deploy",
        "context": {
            "token": "raw-token",
            "attempts": [1, "retry"],
            "model": model,
            "labels": ("first",),
        },
    }

    protected = DataProtectionLayer(sanitizer).protect_external(payload)

    assert sanitizer.paths == [
        ("task",),
        ("context", "token"),
        ("context", "attempts", 0),
        ("context", "attempts", 1),
        ("context", "model", "credential"),
        ("context", "model", "values", 0),
        ("context", "labels", 0),
    ]
    assert protected["task"] == "protected:task"
    assert protected["context"]["token"] == "protected:context/token"
    assert protected["context"]["model"].credential == (
        "protected:context/model/credential"
    )
    assert payload["task"] == "deploy"
    assert payload["context"]["token"] == "raw-token"
    assert model.credential == "model-secret"


def test_set_members_are_copied_sanitized_and_preserve_set_kinds():
    sanitizer = RecordingSanitizer()
    payload = {
        "mutable": {"alpha", "beta"},
        "immutable": frozenset({1, 2}),
    }

    protected = DataProtectionLayer(sanitizer).protect_metadata(payload)

    assert isinstance(protected["mutable"], set)
    assert isinstance(protected["immutable"], frozenset)
    assert all(value.startswith("protected:mutable/") for value in protected["mutable"])
    assert {path[:-1] for path in sanitizer.paths} == {
        ("mutable",),
        ("immutable",),
    }


def test_sanitizer_failure_is_safe_and_does_not_mutate_caller_input():
    sensitive_value = bytearray(b"raw-api-key")
    payload = {"context": {"credential": sensitive_value}}
    before = deepcopy(payload)

    class MutatingFailingSanitizer:
        def sanitize(self, value: Any, *, path: tuple[str | int, ...]) -> Any:
            value.extend(b"-changed")
            raise RuntimeError(f"failed for {value!r}")

    with pytest.raises(SanitizationError) as caught:
        DataProtectionLayer(MutatingFailingSanitizer()).protect_metadata(payload)

    error = caught.value
    assert payload == before
    assert sensitive_value == bytearray(b"raw-api-key")
    assert error.path == ("context", "credential")
    assert error.operation == "log_metadata"
    assert str(error) == "Data sanitization failed."
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "raw-api-key" not in repr(error.__dict__)
