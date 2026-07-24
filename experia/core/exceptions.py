class ExperiaError(Exception):
    """Base exception for all Experia errors."""

    pass


class StorageError(ExperiaError):
    """Raised when a storage operation fails."""

    pass


class EvaluationError(ExperiaError):
    """Raised when an experience evaluation fails."""

    pass


class ConfigurationError(ExperiaError):
    """Raised when Experia is misconfigured."""

    pass
