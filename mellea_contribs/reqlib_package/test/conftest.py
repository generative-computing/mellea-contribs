import os

import pytest


@pytest.fixture(scope="session")
def gh_run() -> int:
    """Fixture indicating if tests are running in GitHub Actions CI."""
    return int(os.environ.get("CICD", 0))
