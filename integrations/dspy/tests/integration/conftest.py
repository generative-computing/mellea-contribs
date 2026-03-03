"""Shared fixtures for integration tests."""

import dspy
import pytest
from mellea import start_session
from mellea_dspy import MelleaLM


@pytest.fixture(scope="session")
def mellea_session():
    """Create a Mellea session for the entire test session."""
    return start_session()


@pytest.fixture
def lm(mellea_session):
    """Create a MelleaLM instance and configure DSPy."""
    lm_instance = MelleaLM(mellea_session=mellea_session, model="mellea-test")
    dspy.configure(lm=lm_instance)
    return lm_instance
