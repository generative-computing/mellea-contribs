import random
import functools
import itertools
import asyncio
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mellea import MelleaSession
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

from .util import sync_wrapper
from .relation import Relation

T = TypeVar("T")

async def delaunay(elems:list[T], criteria:Callable[[T,T,T],bool], k:int=3) -> nx.Graph:

    assert len(elems) >= 2

    g = nx.Graph()
    for elem in elems:
        g.add_node(elem)

    for _ in range(k):

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
            for z in S:
                if await criteria(x, y, z):
                    Sx.append(z)
                else:
                    Sy.append(z)

            return Sx, Sy

        async def construct(parent, elems):
            if len(elems) < 2:
                for elem in elems:
                    g.add_edge(parent, elem)
                return

            c1, c2, _elems = select(elems)
            g.add_edge(parent, c1)
            g.add_edge(parent, c2)

            elems1, elems2 = split(c1, c2, _elems)
            asyncio.gather(
                construct(c1, elems1),
                construct(c2, elems2))

        r1, r2, _elems = select(elems)
        g.add_edge(r1, r2)
        elems1, elems2 = split(r1, r2, _elems)
        asyncio.gather(
            construct(c1, elems1),
            construct(c2, elems2))

    return g


class Cluster(Relation):

    async def atriplet(m:MelleaSession, prompt: str, x:str, y:str, z:str, **kwargs) -> bool:
        """Given a triplet comparison query, perform the query using the LLM.

        It returns True if Z is closer to X than is to Y.
        """
        answer = await m.achoice(prompt + f"\nZ: {z}\nX: {x}\nY: {y}\n" , ["X", "Y"], **kwargs)
        return answer == "X"


    async def acluster(m:MelleaSession, criteria:str, elems:list[str],
                       *,
                       k:int = 3,
                       **kwargs) -> list[set[str]]:
        """Generate an approximate Delaunay graph and perform graph clustering on it.

        The graph construction method follows the n log n algorithm by Haghiri et. al. [1]

        Args:
            criteria: triplet comparison criteria
            elems: list of strings to cluster
            k: k for k-ANNS for Delaunay Graph

            **kwargs: accepts vote, positional, shuffle.

        Returns:
            A cluster representation as list[set[str]]


        [1] Haghiri, Siavash, Debarghya Ghoshdastidar, and Ulrike von Luxburg.
        "Comparison-based nearest neighbor search." Artificial Intelligence and Statistics. PMLR, 2017.

        """

        async def fn(x:str, y:str, z:str) -> bool:
            return await m.atriplet(criteria, x, y, z, **kwargs)

        g = delaunay(elems, fn, k=k)

        communities = list(nx.algorithms.community.greedy_modularity_communities(g))

        return communities


Cluster.cluster = sync_wrapper(Cluster.acluster)


# Testing Delaunay Graph Clustering approach on 2D points.
# (for VA, we replace the triplet comparison with LLM-based one)

Point = tuple[float,float]

def points(
    n_clusters=5,
    points_per_cluster=20,
    radius=10.0,
    cluster_std=0.5,
    seed=None,
):
    """
    Generate 2D points in clusters centered at the vertices of a regular polyhedra.

    Returns
    -------
    points : np.ndarray of shape (n_clusters*points_per_cluster, 2)
        The generated 2D points.
    """
    if seed is not None:
        np.random.seed(seed)

    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centers = np.column_stack([radius * np.cos(angles),
                               radius * np.sin(angles)])

    # Generate clusters
    points = []
    for cx, cy in centers:
        cluster = np.random.normal(
            loc=[cx, cy],
            scale=cluster_std,
            size=(points_per_cluster, 2)
        )
        points.append(cluster)

    return np.vstack(points)

async def criteria(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return np.square(x-z).sum() < np.square(y-z).sum()

def plot(g):
    """
    Plot the graph g whose nodes are 2D points [x, y].
    Also compute greedy modularity communities and color
    nodes by community assignment.
    """

    # --- Compute communities ---
    communities = list(nx.algorithms.community.greedy_modularity_communities(g))

    # Assign a color index to each node
    node_color = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            node_color[node] = cid

    # Color palette
    # If many clusters, matplotlib cycles automatically
    colors = [node_color[n] for n in g.nodes]

    # --- Extract node positions ---
    xs = [node[0] for node in g.nodes]
    ys = [node[1] for node in g.nodes]

    plt.figure(figsize=(7, 7))

    # --- Draw edges ---
    for u, v in g.edges:
        plt.plot([u[0], v[0]], [u[1], v[1]], linewidth=0.8, color="gray", alpha=0.5)

    # --- Draw nodes ---
    sc = plt.scatter(xs, ys, c=colors, cmap="tab10", s=35)

    plt.gca().set_aspect('equal', 'box')
    plt.title("Graph with Greedy Modularity Communities")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    cbar = plt.colorbar(sc)
    cbar.set_label("Community ID")

    plt.show()

def main():
    elems = points(points_per_cluster=30)
    elems = [ tuple(p) for p in elems ]
    g = delaunay(elems)
    plot(g)

if __name__ == "__main__":

    main()

