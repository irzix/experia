"""Shared deterministic local, CI, concurrency, and extended profiles."""

import os

from hypothesis import settings

from tests.quality_profiles import (
    CI_PROFILE,
    DETERMINISTIC_HYPOTHESIS_SETTINGS,
    HYPOTHESIS_PROFILE_MIN_EXAMPLES,
    LOCAL_PROFILE,
)

for profile_name, minimum_examples in HYPOTHESIS_PROFILE_MIN_EXAMPLES.items():
    settings.register_profile(
        profile_name,
        max_examples=minimum_examples,
        **DETERMINISTIC_HYPOTHESIS_SETTINGS,
    )

_default_profile = CI_PROFILE if os.environ.get("CI") else LOCAL_PROFILE
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", _default_profile))
