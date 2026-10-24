"""Data splitting and leakage prevention module for recommendation evaluation.

Ensures:
- Exact reproducible train/test interaction splitting.
- Only eligible multi-repository contributors (degree >= 2) have interactions held out for testing.
- Single-repository contributors (degree == 1) remain strictly in the training graph so the graph structure
  and repository degree distributions are preserved without disconnected isolate artifacts.
- Training graph and test ground-truth sets are cleanly isolated.
- Community detection on the training graph is recalculated strictly from training edges to prevent
  any data leakage into community-aware recommenders.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import pandas as pd

from src.community_detection.detector import RepositoryCommunityDetector


@dataclass
class DatasetSplit:
    """Contains partitioned training graph, held-out ground truth interactions, and metadata."""

    train_graph: nx.Graph
    train_repo_projection: nx.Graph
    train_leiden_communities: Dict[str, int]
    train_louvain_communities: Dict[str, int]
    test_interactions: Dict[str, Set[str]]  # contributor_node_id -> set of held-out repo_node_ids
    test_contributors: List[str]
    total_train_edges: int
    total_test_edges: int
    split_strategy: str
    seed: int


def create_train_test_split(
    bipartite_graph: nx.Graph,
    test_ratio: float = 0.20,
    min_contributor_degree: int = 2,
    seed: int = 42,
) -> DatasetSplit:
    """Create a leakage-free train/test split of contributor-repository interactions.
    
    Leakage Prevention Controls:
    1. Multi-repo contributors (degree >= 2) have a subset of their repository edges held out (leaving at least 1 edge in train).
    2. Contributors with degree == 1 have 0 edges held out, retaining connectivity in training without leakage.
    3. The training graph G_train contains ONLY training edges.
    4. The training repository projection G_R_train is constructed ONLY from G_train.
    5. Leiden community detection is computed ONLY on G_R_train, ensuring zero leakage of test edges into community boundaries.
    """
    rng = np.random.default_rng(seed)

    c_nodes = [
        n for n, d in bipartite_graph.nodes(data=True)
        if d.get("node_type") == "contributor" or d.get("bipartite") == 0
    ]
    r_nodes = [
        n for n, d in bipartite_graph.nodes(data=True)
        if d.get("node_type") == "repository" or d.get("bipartite") == 1
    ]

    # Eligible contributors for recommendation evaluation: degree >= min_contributor_degree
    multi_c_nodes = [c for c in c_nodes if bipartite_graph.degree(c) >= min_contributor_degree]

    train_edges: Set[Tuple[str, str]] = set()
    test_interactions: Dict[str, Set[str]] = {}

    for c in c_nodes:
        neighbors = list(bipartite_graph.neighbors(c))
        if len(neighbors) < min_contributor_degree:
            # Degree-1 contributor: all edges stay in train
            for r in neighbors:
                train_edges.add((c, r))
        else:
            # Multi-degree contributor: hold out a fraction of interactions (at least 1, leaving at least 1 in train)
            num_holdout = max(1, int(round(len(neighbors) * test_ratio)))
            if num_holdout >= len(neighbors):
                num_holdout = len(neighbors) - 1

            shuffled = list(neighbors)
            rng.shuffle(shuffled)
            held_out = set(shuffled[:num_holdout])
            keep_train = set(shuffled[num_holdout:])

            for r in keep_train:
                train_edges.add((c, r))

            test_interactions[c] = held_out

    # Construct strict training bipartite graph
    G_train = nx.Graph()
    for n, d in bipartite_graph.nodes(data=True):
        G_train.add_node(n, **d)

    for c, r in train_edges:
        edge_data = bipartite_graph.get_edge_data(c, r) or {}
        G_train.add_edge(c, r, **edge_data)

    # Construct training unipartite repository projection
    G_R_train = bipartite.weighted_projected_graph(G_train, r_nodes)

    # Run Leiden community detection strictly on training graph projection
    isolated_in_train = {n for n in r_nodes if G_R_train.degree(n) == 0}
    connected_in_train = [n for n in r_nodes if n not in isolated_in_train]
    sub_train = G_R_train.subgraph(connected_in_train).copy()

    # We use igraph + leidenalg on sub_train
    import igraph as ig
    import leidenalg

    node_to_idx = {n: i for i, n in enumerate(sub_train.nodes())}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in sub_train.edges()]
    weights = [float(d.get("weight", 1.0)) for u, v, d in sub_train.edges(data=True)]

    ig_g = ig.Graph(n=len(sub_train), edges=edges, directed=False)
    ig_g.es["weight"] = weights

    partition = leidenalg.find_partition(
        ig_g,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=seed,
        n_iterations=5,
    )

    train_leiden_comms: Dict[str, int] = {}
    sorted_comms = sorted([list(comm) for comm in partition], key=len, reverse=True)
    for c_id, comm in enumerate(sorted_comms, 1):
        for idx in comm:
            train_leiden_comms[idx_to_node[idx]] = c_id
    for n in isolated_in_train:
        train_leiden_comms[n] = -1

    # Also compute Louvain for comparison on train graph
    from networkx.algorithms.community import louvain_communities
    louv_comms = louvain_communities(sub_train, weight="weight", seed=seed)
    train_louvain_comms: Dict[str, int] = {}
    sorted_l_comms = sorted(louv_comms, key=len, reverse=True)
    for c_id, comm in enumerate(sorted_l_comms, 1):
        for n in comm:
            train_louvain_comms[n] = c_id
    for n in isolated_in_train:
        train_louvain_comms[n] = -1

    total_test_edges = sum(len(repos) for repos in test_interactions.values())

    return DatasetSplit(
        train_graph=G_train,
        train_repo_projection=G_R_train,
        train_leiden_communities=train_leiden_comms,
        train_louvain_communities=train_louvain_comms,
        test_interactions=test_interactions,
        test_contributors=list(test_interactions.keys()),
        total_train_edges=G_train.number_of_edges(),
        total_test_edges=total_test_edges,
        split_strategy=f"multi_contributor_heldout_ratio_{test_ratio}",
        seed=seed,
    )
