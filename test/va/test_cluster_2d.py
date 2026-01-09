"""
Testing Delaunay Graph Clustering approach on 2D points.
(for VA, we replace the triplet comparison with LLM-based one)
"""

import pytest
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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

    return np.vstack(points), points

def criteria(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return np.square(x-z).sum() < np.square(y-z).sum()

def delaunay(elems, k=7):

    assert len(elems) >= 2

    g = nx.Graph()
    for elem in elems:
        g.add_node(elem)

    for _ in range(k):

        def select(elems:list[Point]):
            assert len(elems) >= 2
            _elems = elems.copy()
            i1 = random.randint(0, len(_elems)-1)
            r1 = _elems.pop(i1)
            i2 = random.randint(0, len(_elems)-1)
            r2 = _elems.pop(i2)
            return r1, r2, _elems

        def split(x:Point, y:Point, S:list[Point]):
            """Split a set S into Sx and Sy, which are closer to x or y, respectively"""

            Sx = []
            Sy = []
            for z in S:
                if criteria(x, y, z):
                    Sx.append(z)
                else:
                    Sy.append(z)

            return Sx, Sy

        def construct(parent, elems):
            if len(elems) < 2:
                for elem in elems:
                    g.add_edge(parent, elem)
                return

            c1, c2, _elems = select(elems)
            g.add_edge(parent, c1)
            g.add_edge(parent, c2)

            elems1, elems2 = split(c1, c2, _elems)
            construct(c1, elems1)
            construct(c2, elems2)

        r1, r2, _elems = select(elems)
        g.add_edge(r1, r2)
        elems1, elems2 = split(r1, r2, _elems)
        construct(r1, elems1)
        construct(r2, elems2)

    return g

def plot(g, communities):
    """
    Plot the graph g whose nodes are 2D points [x, y].
    Also compute greedy modularity communities and color
    nodes by community assignment.
    """

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
        plt.plot([u[0], v[0]], [u[1], v[1]], linewidth=0.8, color="gray", alpha=0.2)

    # --- Draw nodes ---
    sc = plt.scatter(xs, ys, c=colors, cmap="tab10", s=35)

    plt.gca().set_aspect('equal', 'box')
    plt.title("Graph with Greedy Modularity Communities")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    cbar = plt.colorbar(sc)
    cbar.set_label("Community ID")

    plt.savefig("test_cluster_2d.png")
    # plt.show()

    pass

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

def test_cluster_2d():
    elems, clusters = points(points_per_cluster=30)
    elems = [ tuple(p) for p in elems ]
    g = delaunay(elems, k=7)

    # To obtain exactly n communities, set both cutoff and best_n to n.
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html
    communities = list(nx.algorithms.community.greedy_modularity_communities(g, cutoff=5, best_n=5))

    plot(g, communities)

    clusters_pred = dict()
    for cid, comm in enumerate(communities):
        for node in comm:
            clusters_pred[(node[0], node[1])] = cid

    clusters_true = dict()
    for cid, comm in enumerate(clusters):
        for node in comm:
            clusters_true[(node[0], node[1])] = cid


    # for (e1, cid_pred1), (e2, cid_pred2) in zip(sorted(clusters_pred.items()),
    #                                             sorted(clusters_true.items()),):
    #     assert e1 == e2
    #     print(e1, cid_pred1, cid_pred2)

    assert len(clusters) == len(communities)
    assert len(clusters_true) == len(clusters_pred)
    assert set(clusters_true.keys()) == set(clusters_pred.keys())

    assert rand_index(clusters_pred, clusters_true) > 0.9

if __name__ == "__main__":
    pytest.main([__file__])
