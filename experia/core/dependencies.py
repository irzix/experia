"""Internal guards for feature-scoped optional dependencies."""

from experia.core.exceptions import ConfigurationError


def require_optional_dependency(
    available: bool,
    *,
    feature: str,
    extra: str,
) -> None:
    """Raise a contextual error when an invoked feature's extra is unavailable."""
    if available:
        return

    raise ConfigurationError(
        f"{feature} requires the optional installation extra '{extra}'.",
        feature=feature,
        parameter="dependency",
        extra=extra,
    )
