"""Shared pytest fixtures and test environment setup.

Sets a dummy API key in the environment *before* any test module imports the
app. config.Settings() requires API_KEY at import time and would otherwise fail
collection. This value is a test-only fixture, not a secret.
"""

import os

# Must run before document_brain.config is imported anywhere.
os.environ["API_KEY"] = "test-api-key"
