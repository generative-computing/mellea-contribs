import asyncio
import pytest

from mellea_va import Relation

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



async def test_relation(m: MelleaSession):
    """"""

    MelleaSession.powerup(Relation)

    assert await m.agt("Is the number X larger than the number Y?", "2", "1"), "number test 2>1"
    assert await m.agt("Is the number X smaller than the number Y?", "1", "2"), "number test 1<2"
    assert await m.agt("Is the country X larger than the country Y by area?", "United States", "Japan"), "area test US > Japan"
    assert await m.agt("Is the country X more densely populated than the country Y?", "Japan", "United States"), "population density test US > Japan"




if __name__ == "__main__":
    pytest.main([__file__])
