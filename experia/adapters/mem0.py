"""Import-compatible placeholder for the planned Mem0 adapter."""

from experia.core.exceptions import UnavailableFeatureError


class Mem0Adapter:
    """Planned Mem0 adapter; unavailable in the current release."""

    def __new__(cls, *args: object, **kwargs: object) -> "Mem0Adapter":
        raise UnavailableFeatureError("mem0", status="planned")


Mem0MemoryAdapter = Mem0Adapter

__all__ = ["Mem0Adapter", "Mem0MemoryAdapter"]
