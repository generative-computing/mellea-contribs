import random
import functools
import itertools
import asyncio
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mellea import MelleaSession
from mellea.backends import Backend
from mellea.stdlib.base import Context
from mellea.stdlib.functional import ainstruct
from mellea.helpers.fancy_logger import FancyLogger
from mellea.helpers.event_loop_helper import _run_async_in_thread

from pydantic import BaseModel

from typing import (
    Literal,
    Callable,
    TypeVar,
    List,
)

import numpy as np

from .util import session_wrapper, sync_wrapper
from .core import Core, abool

T = TypeVar("T")

async def delaunay(elems:list[T], criterion:Callable[[T,T,T],bool], k:int=3) -> nx.Graph:

    assert len(elems) >= 2

    def select(elems:list[T]):
        assert len(elems) >= 2
        _elems = elems.copy()
        i1 = random.randint(0, len(_elems)-1)
        r1 = _elems.pop(i1)
        i2 = random.randint(0, len(_elems)-1)
        r2 = _elems.pop(i2)
        return r1, r2, _elems

    async def split(x:T, y:T, S:list[T]):
        """Split a set S into Sx and Sy, which are closer to x or y, respectively"""

        Sx = []
        Sy = []
        tasks = [
            criterion(x, y, z)
            for z in S
        ]
        results = await asyncio.gather(*tasks)
        for z, r in zip(S, results):
            if r:
                Sx.append(z)
            else:
                Sy.append(z)

        return Sx, Sy

    async def construct(parent, elems):
        if len(elems) < 2:
            g = nx.Graph()
            for elem in elems:
                g.add_edge(parent, elem)
            return g

        c1, c2, _elems = select(elems)
        elems1, elems2 = await split(c1, c2, _elems)
        g1, g2 = await asyncio.gather(
            construct(c1, elems1),
            construct(c2, elems2))

        g = nx.compose(g1, g2)
        g.add_edge(parent, c1)
        g.add_edge(parent, c2)
        return g

    async def tree():
        r1, r2, _elems = select(elems)
        elems1, elems2 = await split(r1, r2, _elems)
        g1, g2 = await asyncio.gather(
            construct(r1, elems1),
            construct(r2, elems2))
        g = nx.compose(g1, g2)
        g.add_edge(r1, r2)
        return g

    g = nx.Graph()
    trees = await asyncio.gather(*[tree() for _ in range(k)])
    for t in trees:
        g = nx.compose(g, t)
    return g


async def atriplet(backend:Backend, ctx:Context, prompt: str, x:str, y:str, z:str, **kwargs) -> bool:
    """Given a triplet comparison query, perform the query using the LLM.

    It returns True if Z is closer to X than is to Y.
    """
    return await abool(backend, ctx, prompt + f"\nZ: {z}\nX: {x}\nY: {y}\n", **kwargs)


async def acluster(backend:Backend, ctx:Context, criterion:str, elems:list[str],
                   *,
                   k:int = 3,
                   n_clusters:int = 3,
                   **kwargs) -> list[set[str]]:
    """Generate an approximate Delaunay graph and perform graph clustering on it.

    The graph construction method follows the n log n algorithm by Haghiri et. al. [1]

    Args:
        criterion: triplet comparison criterion
        elems: list of strings to cluster
        k: k for k-ANNS for Delaunay Graph
        n_clusters: the number of clusters

        **kwargs: accepts vote, positional, shuffle.

    Returns:
        A cluster representation as list[set[str]]


    [1] Haghiri, Siavash, Debarghya Ghoshdastidar, and Ulrike von Luxburg.
    "Comparison-based nearest neighbor search." Artificial Intelligence and Statistics. PMLR, 2017.

    """

    async def fn(x:str, y:str, z:str) -> bool:
        return await atriplet(backend, ctx, criterion, x, y, z, **kwargs)

    g = await delaunay(elems, fn, k=k)

    communities = list(nx.algorithms.community.greedy_modularity_communities(g, cutoff=n_clusters, best_n=n_clusters))

    return communities


class Cluster(Core):
    pass


Cluster.atriplet = session_wrapper(atriplet)
Cluster.acluster = session_wrapper(acluster)
Cluster.triplet = sync_wrapper(Cluster.atriplet)
Cluster.cluster = sync_wrapper(Cluster.acluster)

