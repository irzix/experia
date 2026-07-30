"""Import-compatible placeholder for the planned Zep adapter."""

from experia.core.exceptions import UnavailableFeatureError


class ZepAdapter:
    """Planned Zep adapter; unavailable in the current release."""

    def __new__(cls, *args: object, **kwargs: object) -> "ZepAdapter":
        raise UnavailableFeatureError("zep", status="planned")


ZepMemoryAdapter = ZepAdapter

__all__ = ["ZepAdapter", "ZepMemoryAdapter"]
