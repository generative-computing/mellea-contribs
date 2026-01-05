import asyncio
import pytest

from mellea_va import Cluster

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


def rand_index(clusters_pred:dict[tuple[float, float], int],
               clusters_true:dict[tuple[float, float], int]):
    """
    Measures the agreement between two clusters.
    """

    count = 0
    total = 0
    for i1, (e1, cid_pred1) in enumerate(sorted(clusters_pred.items())):
        cid_true1 = clusters_true[e1]
        for i2, (e2, cid_pred2) in enumerate(sorted(clusters_pred.items())):
            cid_true2 = clusters_true[e2]

            if i1 >= i2:
                continue

            same_pred = (cid_pred1 == cid_pred2)
            same_true = (cid_true1 == cid_true2)

            total += 1
            if same_pred == same_true: # both true or both false
                count += 1

    N = len(clusters_pred)
    assert total == ((N * (N-1)) // 2)
    return count / ((N * (N-1)) // 2)


async def test_cluster(m: MelleaSession):
    """"""

    MelleaSession.powerup(Cluster)

    assert await m.atriplet(
        "Is country Z culturally closer to country X than is to country Y?",
        "United States", "Japan", "Canada"), "cultural similarity test US/Canada/Japan"

    communities = await m.acluster(
        # "Is country Z geographically closer to country X than is to country Y?",
        # ["Canada", "United States",
        #  "Spain", "France",
        #  "China", "Japan"],
        "Is color Z more similar to color X than is to color Y?",
        ["red", "pink",
         "blue", "cyan",
         "green", "lime"],
        k=5,
        n_clusters=3)

    clusters_pred = dict()
    for cid, comm in enumerate(communities):
        print(cid, comm)
        for node in comm:
            clusters_pred[node] = cid

    clusters_true = {
        # "Canada":0,
        # "United States":0,
        # "Spain":1,
        # "France":1,
        # "China":2,
        # "Japan":2,
        "red":0,
        "pink":0,
        "blue":1,
        "cyan":1,
        "green":2,
        "lime":2,
    }
    assert rand_index(clusters_pred, clusters_true) > 0.9


if __name__ == "__main__":
    pytest.main([__file__])
