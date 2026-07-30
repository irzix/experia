"""Import-compatible placeholder for the planned CrewAI integration."""

from experia.core.exceptions import UnavailableFeatureError


class CrewAIIntegration:
    """Planned CrewAI integration; unavailable in the current release."""

    def __new__(cls, *args: object, **kwargs: object) -> "CrewAIIntegration":
        raise UnavailableFeatureError("crewai", status="planned")


CrewAIAdapter = CrewAIIntegration

__all__ = ["CrewAIAdapter", "CrewAIIntegration"]
