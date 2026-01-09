import asyncio
import pytest

from mellea_va import Pareto

from mellea import MelleaSession, start_session
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import CBlock, SimpleContext
from mellea.stdlib.requirement import Requirement, simple_validate

# @pytest.fixture(scope="module")
# def m() -> MelleaSession:
#     return MelleaSession(backend=OllamaModelBackend(), ctx=ChatContext())

@pytest.fixture(scope="function")
def m():
    """Fresh Ollama session for each test."""
    session = start_session()
    yield session
    session.reset()



async def test_pareto(m: MelleaSession):
    """"""

    MelleaSession.powerup(Pareto)

    assert await m.amax("Is country X larger than country Y by area?",
                        ["France", "United States", "Australia", "Singapore"]) == "United States"
    assert await m.amax("Is the country X more Asian than the country Y?",
                        ["France", "United States", "Australia", "Singapore"]) == "Singapore"

    assert set(await m.apareto(["Is country X larger than country Y by area?",
                                "Is the country X more Asian than the country Y?"],
                               ["France", "United States", "Australia", "Singapore"])) == set(["United States", "Singapore"])


if __name__ == "__main__":
    pytest.main([__file__])
