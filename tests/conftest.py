"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from cog_statistics.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)
