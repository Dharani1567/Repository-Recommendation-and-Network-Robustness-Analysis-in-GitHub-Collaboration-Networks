"""Network Structural and Community Robustness Analyzer.

Calculates:
- Network topological degradation:
  * Remaining contributor count, repository count, edge count
  * Number of connected components
  * Largest Connected Component (GCC) size and fraction
  * Network density
  * Number of isolated repositories (degree 0) and isolated contributors
  * Network fragmentation index: 1.0 - (GCC_nodes / Total_nodes)
  * Average Shortest Path Length computed strictly on the Giant Connected Component
- Repository Projection Rebuilding and Community Robustness:
  * Rebuilds weighted unipartite repository projection from remaining graph
  * Recalculates Leiden community partition strictly on remaining graph
  * Community count, modularity Q, community size distribution
  * Adjusted Rand Index (ARI) measuring partition drift from Part 4 baseline
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import igraph as ig
import leidenalg
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.community import modularity
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


class NetworkRobustnessAnalyzer:
    """Evaluates topological breakdown and community restructuring after contributor removal."""

    def __init__(self, baseline_repo_communities: Dict[str, int]):
        self.baseline_repo_communities = baseline_repo_communities

    def evaluate_structural_metrics(
        self,
        graph: nx.Graph,
        experiment_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute bipartite structural robustness metrics."""
        c_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "contributor" or d.get("bipartite") == 0]
        r_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "repository" or d.get("bipartite") == 1]

        total_nodes = graph.number_of_nodes()
        total_edges = graph.number_of_edges()

        # Isolated nodes
        isolated_repos = [r for r in r_nodes if graph.degree(r) == 0]
        isolated_contribs = [c for c in c_nodes if graph.degree(c) == 0]

        # Bipartite density
        possible_edges = len(c_nodes) * len(r_nodes)
        density = float(total_edges / possible_edges) if possible_edges > 0 else 0.0

        # Connected components on bipartite graph
        ccs = list(nx.connected_components(graph))
        num_ccs = len(ccs)
        cc_sizes = sorted([len(c) for c in ccs], reverse=True)
        gcc_size = cc_sizes[0] if cc_sizes else 0
        gcc_fraction = float(gcc_size / total_nodes) if total_nodes > 0 else 0.0
        fragmentation = float(1.0 - gcc_fraction)

        # Average shortest path length strictly on Giant Connected Component (C-accelerated via igraph)
        avg_path_length_gcc = None
        if gcc_size > 1:
            try:
                node_to_idx = {n: i for i, n in enumerate(graph.nodes())}
                edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
                ig_g = ig.Graph(n=len(graph), edges=edges, directed=False)
                ig_gcc = ig_g.connected_components().giant()
                avg_path_length_gcc = round(float(ig_gcc.average_path_length()), 4)
            except Exception:
                avg_path_length_gcc = None

        return {
            **experiment_meta,
            "remaining_contributors": len(c_nodes),
            "remaining_repositories": len(r_nodes),
            "remaining_edges": total_edges,
            "connected_components_count": num_ccs,
            "gcc_nodes": gcc_size,
            "gcc_fraction": round(gcc_fraction, 4),
            "network_density": round(density, 6),
            "fragmentation_index": round(fragmentation, 4),
            "isolated_repositories_count": len(isolated_repos),
            "isolated_contributors_count": len(isolated_contribs),
            "avg_shortest_path_length_gcc": avg_path_length_gcc,
        }

    def evaluate_community_robustness(
        self,
        graph: nx.Graph,
        experiment_meta: Dict[str, Any],
        seed: int = 42,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Rebuild repository projection, recalculate Leiden partition, and evaluate modularity & drift."""
        r_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "repository" or d.get("bipartite") == 1]

        # Rebuild repository projection strictly from remaining graph
        repo_proj = bipartite.weighted_projected_graph(graph, r_nodes)
        num_proj_edges = repo_proj.number_of_edges()

        # Identify isolated repositories in projection
        isolated_in_proj = {r for r in r_nodes if repo_proj.degree(r) == 0}
        connected_in_proj = [r for r in r_nodes if r not in isolated_in_proj]

        new_communities: Dict[str, int] = {}
        for r in isolated_in_proj:
            new_communities[r] = -1

        modularity_score = 0.0
        community_sizes: List[int] = []

        if connected_in_proj:
            sub_proj = repo_proj.subgraph(connected_in_proj).copy()

            # Convert to igraph
            node_to_idx = {n: i for i, n in enumerate(sub_proj.nodes())}
            idx_to_node = {i: n for n, i in node_to_idx.items()}
            edges = [(node_to_idx[u], node_to_idx[v]) for u, v in sub_proj.edges()]
            weights = [float(d.get("weight", 1.0)) for u, v, d in sub_proj.edges(data=True)]

            ig_g = ig.Graph(n=len(sub_proj), edges=edges, directed=False)
            ig_g.es["weight"] = weights

            partition = leidenalg.find_partition(
                ig_g,
                leidenalg.ModularityVertexPartition,
                weights="weight",
                seed=seed,
                n_iterations=5,
            )

            sorted_comms = sorted([list(c) for c in partition], key=len, reverse=True)
            for comm_id, comm in enumerate(sorted_comms, 1):
                for idx in comm:
                    new_communities[idx_to_node[idx]] = comm_id

            community_sizes = [len(c) for c in sorted_comms]

            # Calculate modularity on connected projection subgraph
            comm_sets = [{idx_to_node[i] for i in comm} for comm in sorted_comms]
            modularity_score = round(float(modularity(sub_proj, comm_sets, weight="weight")), 4)

        # Measure partition stability/drift relative to baseline Part 4 Leiden assignments
        ordered_repos = sorted(r_nodes)
        baseline_labels = [self.baseline_repo_communities.get(r, -1) for r in ordered_repos]
        new_labels = [new_communities.get(r, -1) for r in ordered_repos]
        ari_drift = round(float(adjusted_rand_score(baseline_labels, new_labels)), 4)

        results = {
            **experiment_meta,
            "projection_edges": num_proj_edges,
            "isolated_repositories_in_projection": len(isolated_in_proj),
            "leiden_communities_count": len(community_sizes),
            "modularity_q": modularity_score,
            "largest_community_size": community_sizes[0] if community_sizes else 0,
            "community_sizes_distribution": str(community_sizes),
            "ari_vs_baseline_partition": ari_drift,
        }

        return results, new_communities
