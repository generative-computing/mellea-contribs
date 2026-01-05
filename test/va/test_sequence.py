import asyncio
import pytest

from mellea_va import Sequence

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



async def test_sequence(m: MelleaSession):
    """"""

    MelleaSession.powerup(Sequence)

    assert await m.amap("X", "a noun corresponding to X with the opposite sex", ["ox", "rooster"]) == ["cow", "hen"]

    assert await m.afind("X", "X is a plant", ["ox", "hen", "carrot", "car"]) == "carrot"

    assert await m.asort("Is country X larger than country Y by area?",
                         ["France", "United States", "Nigeria", "Singapore"]) == ["United States", "Nigeria", "France", "Singapore"]
    assert await m.amax("Is country X larger than country Y by area?",
                        ["France", "United States", "Australia", "Singapore"]) == "United States"
    assert await m.amedian("Is the latitude of city X larger than the latitude of city Y?",
                           ["Yakutsk","Taipei","Edinburgh","Singapore","Melbourne"]) == "Taipei"


if __name__ == "__main__":
    pytest.main([__file__])
