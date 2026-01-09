import asyncio
import pytest

import time

from mellea_va import Core

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



async def test_core(m: MelleaSession):
    """"""

    MelleaSession.powerup(Core)

    assert await m.abool("Is 1+1=2?")

    assert await m.achoice("Which city is in the United States?",
                           ["Tokyo","Boston","Paris","Melbourne"]) == "Boston"

    t1 = time.time()
    await m.abool("Is 1+1=2?", vote=1)
    t2 = time.time()
    dt1 = t2-t1

    t1 = time.time()
    await m.abool("Is 1+1=2?", vote=5)
    t2 = time.time()
    dt2 = t2-t1

    r = dt2 / dt1
    print(f"asynchronouos efficiency for 5 calls: {r}")
    assert 3 < r < 5, "asynchronous call efficiency test"

    assert m.bool("Is 1+1=2?")

    assert m.choice("Which city is in the United States?",
                    ["Tokyo","Boston","Paris","Melbourne"]) == "Boston"


if __name__ == "__main__":
    pytest.main([__file__])
