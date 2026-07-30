"""Import-compatible placeholder for the planned PostgreSQL adapter."""

from experia.core.exceptions import UnavailableFeatureError


class PostgresAdapter:
    """Planned PostgreSQL adapter; unavailable in the current release."""

    def __new__(cls, *args: object, **kwargs: object) -> "PostgresAdapter":
        raise UnavailableFeatureError("postgres", status="planned")


PostgresStore = PostgresAdapter
PostgreSQLStore = PostgresAdapter

__all__ = ["PostgresAdapter", "PostgresStore", "PostgreSQLStore"]
