"""Pytest configuration for the test suite.

Configures hypothesis settings profile for reproducible property-based tests.
Use --hypothesis-seed=0 in CI for deterministic runs.
"""

from hypothesis import settings, HealthCheck

# Register a CI profile with deterministic seed and reasonable runtime
settings.register_profile(
    "ci",
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)

# Default profile for local development
settings.register_profile(
    "default",
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile("default")
