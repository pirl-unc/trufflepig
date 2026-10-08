"""Shared primitives for immutable domain data and public serialization.

Domain records should normally be frozen dataclasses. Mutable inputs are
snapshotted into immutable value types at construction. ``public_dict()``
methods use :func:`dataclass_public_dict` when their public schema is a direct
projection of fields, and remain explicit when the public schema adds, removes,
or transforms fields.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar


Key = TypeVar("Key")
Value = TypeVar("Value")


@dataclass(frozen=True, init=False, eq=False)
class FrozenMapping(Mapping[Key, Value], Generic[Key, Value]):
    """An immutable, hashable snapshot of a small mapping."""

    _items: tuple[tuple[Key, Value], ...]

    def __init__(self, values: Mapping[Key, Value] | None = None):
        object.__setattr__(self, "_items", tuple(dict(values or {}).items()))

    def __getitem__(self, key: Key) -> Value:
        for current, value in self._items:
            if current == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[Key]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    def __hash__(self) -> int:
        return hash(frozenset(self._items))


def public_value(value: Any) -> Any:
    """Return fresh JSON-compatible containers for a domain value.

    Unsupported object types fail loudly instead of leaking implementation
    objects into manifests or relying on ``json.dumps(default=...)``.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Public mapping keys must be strings")
        return {key: public_value(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_public_dict(value)
    if isinstance(value, (tuple, list)):
        return [public_value(item) for item in value]
    raise TypeError(f"Unsupported public data value: {type(value).__name__}")


def dataclass_public_dict(value: Any) -> dict[str, Any]:
    """Serialize dataclass fields into a fresh JSON-compatible dictionary."""
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("dataclass_public_dict requires a dataclass instance")
    return {
        field.name: public_value(getattr(value, field.name))
        for field in fields(value)
    }
