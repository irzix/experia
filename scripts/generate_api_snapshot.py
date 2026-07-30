"""Generate the canonical, versioned Experia public API snapshot."""

from __future__ import annotations

import argparse
import collections.abc
import dataclasses
import importlib
import inspect
import json
import math
import types
import typing
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, ForwardRef, get_args, get_origin, get_type_hints

SCHEMA_VERSION = 1
PACKAGE_NAME = "experia"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "api-snapshot.json"
DEFAULT_API_REFERENCE = ROOT / "API_REFERENCE.md"
CONSTRUCTOR_BLOCK_START = "<!-- BEGIN GENERATED CONSTRUCTOR CONTRACTS -->"
CONSTRUCTOR_BLOCK_END = "<!-- END GENERATED CONSTRUCTOR CONTRACTS -->"

# Public paths documented outside the package-level ``__all__`` declarations.
_DOCUMENTED_EXPORT_PATHS = (
    "experia.core.interfaces.Evaluator",
    "experia.core.interfaces.MemoryStore",
    "experia.experience.llm_evaluator.LLMEvaluator",
    "experia.experience.models.ExperienceRecord",
    "experia.experience.models.Lesson",
    "experia.improvement.rules.RuleGenerator",
    "experia.integrations.langchain.callbacks.ExperiaCallbackHandler",
    "experia.integrations.langchain.retrievers.ExperiaLearningRetriever",
    "experia.integrations.langgraph.nodes.ExperiaContextNode",
    "experia.integrations.langgraph.nodes.ExperiaLearningNode",
)

# Class members that are part of the documented public contract. Keeping this
# list intentional avoids treating implementation helpers or inherited framework
# internals as Experia APIs.
_PUBLIC_MEMBERS_BY_TARGET = {
    "experia.core.interfaces.Evaluator": ("evaluate",),
    "experia.core.interfaces.MemoryStore": (
        "find_similar_memory",
        "get_experience",
        "get_memory",
        "get_recent_experiences",
        "prune_expired",
        "save_experience",
        "save_lesson",
        "save_lesson_and_memory",
        "save_memory",
        "search_memories",
        "update_memory_feedback",
    ),
    "experia.core.learner.Learner": (
        "aclose",
        "flush",
        "prune",
        "record",
        "reflect",
        "reinforce",
        "remember",
        "retrieve_context",
        "shutdown",
    ),
    "experia.experience.evaluator.SimpleHeuristicEvaluator": ("evaluate",),
    "experia.experience.llm_evaluator.LLMEvaluator": ("evaluate",),
    "experia.improvement.rules.RuleGenerator": ("consolidate_lesson",),
    "experia.integrations.langchain.callbacks.ExperiaCallbackHandler": (
        "on_chain_end",
        "on_chain_error",
        "on_chain_start",
        "on_tool_end",
        "on_tool_error",
        "on_tool_start",
    ),
    "experia.integrations.langchain.retrievers.ExperiaLearningRetriever": (
        "_aget_relevant_documents",
        "_get_relevant_documents",
    ),
    "experia.integrations.langgraph.nodes.ExperiaContextNode": ("__call__",),
    "experia.integrations.langgraph.nodes.ExperiaLearningNode": ("__call__",),
    "experia.memory.embeddings.Embedder": ("embed", "embed_one"),
    "experia.memory.embeddings.LiteLLMEmbedder": ("embed", "embed_one"),
    "experia.memory.store.SQLiteStore": (
        "close",
        "find_similar_memory",
        "get_experience",
        "get_memory",
        "get_recent_experiences",
        "initialize",
        "prune_expired",
        "save_experience",
        "save_lesson",
        "save_lesson_and_memory",
        "save_memory",
        "search_memories",
        "update_memory_feedback",
    ),
}

# This registry is deliberately explicit. Future deprecations must identify both
# the release that introduced the warning and the supported replacement path.
_DEPRECATIONS: dict[str, dict[str, str]] = {}


@dataclass(frozen=True)
class ExportSpec:
    """One import path included in the compatibility baseline."""

    path: str
    members: tuple[str, ...] = ()


def _canonical_path(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return type(value).__name__


def resolve_export(path: str) -> Any:
    """Resolve an importable dotted path without relying on attribute re-exports."""
    parts = path.split(".")
    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            value: Any = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
            continue
        for attribute in parts[split_at:]:
            value = getattr(value, attribute)
        return value
    raise ImportError(f"Unable to resolve public API path: {path}")


def supported_export_specs() -> tuple[ExportSpec, ...]:
    """Return the intentional public surface in deterministic path order."""
    paths = set(_DOCUMENTED_EXPORT_PATHS)
    for namespace_name in ("experia", "experia.core"):
        namespace = importlib.import_module(namespace_name)
        for name in namespace.__all__:
            export_path = f"{namespace_name}.{name}"
            paths.add(export_path)
            paths.add(_canonical_path(getattr(namespace, name)))

    specs = []
    for path in sorted(paths):
        target = _canonical_path(resolve_export(path))
        specs.append(
            ExportSpec(path=path, members=_PUBLIC_MEMBERS_BY_TARGET.get(target, ()))
        )
    return tuple(specs)


def _canonical_annotation(annotation: Any) -> str | None:
    if annotation is inspect.Parameter.empty or annotation is inspect.Signature.empty:
        return None
    if isinstance(annotation, str):
        return annotation.strip("'\"")
    if isinstance(annotation, ForwardRef):
        return annotation.__forward_arg__
    if annotation is None or annotation is type(None):
        return "None"
    if annotation is Any or annotation is typing.Any:
        return "typing.Any"

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is typing.Annotated:
        return _canonical_annotation(arguments[0])
    if origin in (typing.Union, types.UnionType):
        members = sorted(
            (_canonical_annotation(argument) or "typing.Any" for argument in arguments),
            key=lambda value: (value == "None", value),
        )
        return " | ".join(members)
    if origin is typing.Literal:
        values = ", ".join(repr(argument) for argument in arguments)
        return f"typing.Literal[{values}]"

    origin_names = {
        list: "list",
        dict: "dict",
        tuple: "tuple",
        set: "set",
        frozenset: "frozenset",
        type: "type",
        collections.abc.Callable: "collections.abc.Callable",
        collections.abc.Iterable: "collections.abc.Iterable",
        collections.abc.Mapping: "collections.abc.Mapping",
        collections.abc.Sequence: "collections.abc.Sequence",
    }
    if origin is not None:
        name = origin_names.get(origin, _canonical_path(origin))
        if not arguments:
            return name
        rendered = ", ".join(
            _canonical_annotation(argument) or "typing.Any" for argument in arguments
        )
        return f"{name}[{rendered}]"

    if inspect.isclass(annotation):
        if annotation.__module__ == "builtins":
            return annotation.__qualname__
        return _canonical_path(annotation)
    return str(annotation).replace("typing.", "")


def _callable_path(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return _canonical_path(value)


def _stable_value(value: Any) -> dict[str, Any]:
    if value is None or isinstance(value, (bool, int, str)):
        return {"kind": "value", "value": value}
    if isinstance(value, float):
        if math.isfinite(value):
            return {"kind": "value", "value": value}
        return {"kind": "value", "value": str(value)}
    if isinstance(value, Enum):
        return {
            "kind": "enum",
            "type": _canonical_path(type(value)),
            "member": value.name,
            "value": value.value,
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_stable_value(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_stable_value(item) for item in value]}
    if isinstance(value, dict):
        return {
            "kind": "mapping",
            "items": [
                {"key": str(key), "value": _stable_value(item)}
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ],
        }
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if callable(value):
        return {"kind": "callable", "path": _callable_path(value)}
    if repr(value) == "<factory>":
        return {"kind": "factory"}
    return {"kind": "object", "type": _canonical_path(type(value))}


def _default_metadata(
    default: Any,
    *,
    owner: type[Any] | None,
    parameter_name: str,
    parameter_kind: inspect._ParameterKind,
) -> dict[str, Any]:
    if default is inspect.Parameter.empty:
        if parameter_kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return {"kind": "variadic"}
        return {"kind": "required"}

    model_fields = getattr(owner, "model_fields", {}) if owner is not None else {}
    field = model_fields.get(parameter_name)
    default_factory = getattr(field, "default_factory", None)
    if default_factory is not None:
        return {"kind": "factory", "path": _callable_path(default_factory)}
    return _stable_value(default)


def _resolved_hints(callable_value: Any, owner: type[Any] | None) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    if owner is not None:
        try:
            hints.update(get_type_hints(owner, include_extras=True))
        except (NameError, TypeError):
            pass
        init = getattr(owner, "__init__", None)
        if init is not None:
            try:
                hints.update(get_type_hints(init, include_extras=True))
            except (NameError, TypeError):
                pass
    if not inspect.isclass(callable_value):
        try:
            hints.update(get_type_hints(callable_value, include_extras=True))
        except (NameError, TypeError):
            pass
    return hints


def _snapshot_signature(
    callable_value: Any,
    *,
    owner: type[Any] | None = None,
    drop_bound_parameter: bool = False,
) -> dict[str, Any] | None:
    try:
        signature = inspect.signature(callable_value)
    except (TypeError, ValueError):
        return None

    hints = _resolved_hints(callable_value, owner)
    parameters = list(signature.parameters.values())
    if drop_bound_parameter and parameters and parameters[0].name in {"self", "cls"}:
        parameters = parameters[1:]

    return {
        "parameters": [
            {
                "name": parameter.name,
                "kind": parameter.kind.name.lower(),
                "annotation": _canonical_annotation(
                    hints.get(parameter.name, parameter.annotation)
                ),
                "required": parameter.default is inspect.Parameter.empty
                and parameter.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ),
                "default": _default_metadata(
                    parameter.default,
                    owner=owner,
                    parameter_name=parameter.name,
                    parameter_kind=parameter.kind,
                ),
            }
            for parameter in parameters
        ],
        "return": _canonical_annotation(
            hints.get("return", signature.return_annotation)
        ),
    }


def _kind(value: Any) -> str:
    if inspect.isclass(value):
        if issubclass(value, Enum):
            return "enum"
        if getattr(value, "_is_protocol", False):
            return "protocol"
        if issubclass(value, BaseException):
            return "exception"
        if dataclasses.is_dataclass(value):
            return "dataclass"
        return "class"
    if inspect.isfunction(value):
        return "function"
    return "object"


def _deprecation(path: str, value: Any) -> dict[str, Any]:
    explicit = _DEPRECATIONS.get(path, {})
    message = explicit.get("message") or getattr(value, "__deprecated__", None)
    since = explicit.get("since") or getattr(value, "__deprecated_since__", None)
    replacement = explicit.get("replacement") or getattr(
        value, "__deprecated_replacement__", None
    )
    return {
        "is_deprecated": bool(message or since or replacement),
        "message": message,
        "replacement": replacement,
        "since": since,
    }


def _snapshot_member(owner: type[Any], name: str) -> dict[str, Any]:
    value = getattr(owner, name)
    return {
        "name": name,
        "kind": "method",
        "async": inspect.iscoroutinefunction(value),
        "signature": _snapshot_signature(
            value,
            owner=owner,
            drop_bound_parameter=True,
        ),
        "deprecation": _deprecation(f"{_canonical_path(owner)}.{name}", value),
    }


def _snapshot_export(spec: ExportSpec) -> dict[str, Any]:
    value = resolve_export(spec.path)
    kind = _kind(value)
    signature = (
        None
        if kind == "enum"
        else _snapshot_signature(
            value,
            owner=value if inspect.isclass(value) else None,
        )
    )
    export = {
        "path": spec.path,
        "target": _canonical_path(value),
        "kind": kind,
        "async": inspect.iscoroutinefunction(value),
        "signature": signature,
        "members": [
            _snapshot_member(value, member_name) for member_name in sorted(spec.members)
        ],
        "deprecation": _deprecation(spec.path, value),
    }
    if kind == "enum":
        export["values"] = [
            {"name": member.name, "value": member.value} for member in value
        ]
    return export


def build_snapshot(
    export_specs: tuple[ExportSpec, ...] | None = None,
    *,
    package_version: str | None = None,
) -> dict[str, Any]:
    """Build the in-memory canonical snapshot."""
    version = package_version or metadata.version(PACKAGE_NAME)
    specs = export_specs or supported_export_specs()
    return {
        "schema_version": SCHEMA_VERSION,
        "package": PACKAGE_NAME,
        "package_version": version,
        "major_version": int(version.split(".", 1)[0]),
        "exports": [
            _snapshot_export(spec) for spec in sorted(specs, key=lambda item: item.path)
        ],
    }


def canonical_json(snapshot: dict[str, Any]) -> str:
    """Serialize a snapshot to stable UTF-8 JSON bytes represented as text."""
    return (
        json.dumps(
            snapshot,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _render_default(default: dict[str, Any]) -> str:
    kind = default["kind"]
    if kind == "required":
        return "—"
    if kind == "variadic":
        return "variadic"
    if kind == "value":
        return repr(default["value"])
    if kind == "factory":
        return f"factory: {default.get('path', '<runtime>')}"
    if kind == "tuple":
        items = ", ".join(_render_default(item) for item in default["items"])
        return f"({items}{',' if len(default['items']) == 1 else ''})"
    if kind == "list":
        return f"[{', '.join(_render_default(item) for item in default['items'])}]"
    if kind == "mapping":
        return "mapping"
    if kind == "enum":
        return f"{default['type']}.{default['member']}"
    if kind == "ellipsis":
        return "..."
    if kind == "callable":
        return default["path"]
    if kind == "object":
        return f"instance of {default['type']}"
    raise ValueError(f"Unsupported canonical default kind: {kind!r}")


def render_constructor_contracts(snapshot: dict[str, Any]) -> str:
    """Render every inspectable public constructor from a canonical snapshot.

    Aliases are grouped by canonical target so required/default/type metadata has
    one documentation source while every supported import path remains visible.
    """
    exports_by_target: dict[str, list[dict[str, Any]]] = {}
    for export in snapshot["exports"]:
        if export["signature"] is not None:
            exports_by_target.setdefault(export["target"], []).append(export)

    lines = [
        CONSTRUCTOR_BLOCK_START,
        "## Canonical public constructor contracts",
        "",
        "Generated from [`api-snapshot.json`](api-snapshot.json). Do not edit this",
        "section independently of the canonical snapshot. Required/default/type",
        "metadata reflects the installed package; `Learner.evaluator` is required.",
    ]
    for target in sorted(exports_by_target):
        exports = exports_by_target[target]
        signature = exports[0]["signature"]
        import_paths = ", ".join(
            f"`{export['path']}`"
            for export in sorted(exports, key=lambda item: item["path"])
        )
        lines.extend(
            [
                "",
                f"### `{target}`",
                "",
                f"Supported import paths: {import_paths}",
                "",
                "| Parameter | Kind | Exposed type | Required | Default |",
                "|---|---|---|---:|---|",
            ]
        )
        for parameter in signature["parameters"]:
            kind = parameter["kind"]
            name = parameter["name"]
            if kind == "var_positional":
                name = f"*{name}"
            elif kind == "var_keyword":
                name = f"**{name}"
            annotation = parameter["annotation"] or "unannotated"
            default = _render_default(parameter["default"])
            lines.append(
                "| "
                f"`{_markdown_cell(name)}` | `{_markdown_cell(kind)}` | "
                f"`{_markdown_cell(annotation)}` | "
                f"{'yes' if parameter['required'] else 'no'} | "
                f"`{_markdown_cell(default)}` |"
            )
    lines.extend(["", CONSTRUCTOR_BLOCK_END])
    return "\n".join(lines)


def sync_api_reference(
    snapshot: dict[str, Any],
    path: Path = DEFAULT_API_REFERENCE,
) -> Path:
    """Replace the generated constructor block with data from ``snapshot``."""
    reference = path.read_text(encoding="utf-8")
    if (
        reference.count(CONSTRUCTOR_BLOCK_START) != 1
        or reference.count(CONSTRUCTOR_BLOCK_END) != 1
    ):
        raise ValueError(
            f"{path} must contain exactly one generated constructor contract block."
        )
    prefix, remainder = reference.split(CONSTRUCTOR_BLOCK_START, 1)
    _, suffix = remainder.split(CONSTRUCTOR_BLOCK_END, 1)
    synchronized = prefix + render_constructor_contracts(snapshot) + suffix
    path.write_text(synchronized, encoding="utf-8")
    return path


def write_snapshot(
    output: Path = DEFAULT_OUTPUT,
    *,
    snapshot: dict[str, Any] | None = None,
) -> Path:
    """Generate and write the canonical snapshot."""
    output.write_text(canonical_json(snapshot or build_snapshot()), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"snapshot destination (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--api-reference",
        type=Path,
        help="also synchronize the generated constructor block in this API reference",
    )
    arguments = parser.parse_args()
    snapshot = build_snapshot()
    output = write_snapshot(arguments.output, snapshot=snapshot)
    print(output)
    if arguments.api_reference is not None:
        print(sync_api_reference(snapshot, arguments.api_reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
