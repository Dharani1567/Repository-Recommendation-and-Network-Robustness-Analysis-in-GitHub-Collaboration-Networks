"""Dataset analysis and network suitability evaluation tool.

Inspects:
- Repository, contributor, and contribution edge counts
- Degree distributions (contributors per repo, repos per contributor)
- Bipartite graph properties: density, connected components, Giant Connected Component (GCC)
- Unipartite projections (Contributor-Contributor co-working, Repo-Repo co-contribution)
- Community detectability feasibility
- Data quality, duplicates, and missing identifiers
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd


def analyze_dataset_and_network(
    data_dir: Path,
    dataset_name: str = "Current Dataset",
) -> Dict[str, Any]:
    """Run comprehensive analysis on repositories, contributors, and contributions datasets."""
    repos_file = data_dir / "repositories.csv"
    contribs_file = data_dir / "contributors.csv"
    edges_file = data_dir / "contributions.csv"

    if not repos_file.exists() or not contribs_file.exists() or not edges_file.exists():
        raise FileNotFoundError(f"Missing one or more required dataset CSVs in {data_dir}")

    df_repos = pd.read_csv(repos_file)
    df_contribs = pd.read_csv(contribs_file)
    df_edges = pd.read_csv(edges_file)

    # 1. Dataset Counts
    num_repos = len(df_repos)
    num_unique_repos = df_repos["repo_id"].nunique()
    num_contribs = len(df_contribs)
    num_unique_contribs = df_contribs["contributor_id"].nunique()
    num_edges = len(df_edges)
    num_unique_edges = len(df_edges.drop_duplicates(subset=["contributor_id", "repo_id"]))

    # 2. Duplicate & Missing Checks
    dup_repos = num_repos - num_unique_repos
    dup_contribs = num_contribs - num_unique_contribs
    dup_edges = num_edges - num_unique_edges

    missing_repo_ids = int(df_repos["repo_id"].isna().sum())
    missing_contrib_ids = int(df_contribs["contributor_id"].isna().sum())
    missing_edge_ids = int(df_edges["contributor_id"].isna().sum() + df_edges["repo_id"].isna().sum())

    # 3. Distribution Metrics
    c_per_r = df_edges.groupby("repo_id")["contributor_id"].count()
    r_per_c = df_edges.groupby("contributor_id")["repo_id"].count()

    single_repo_contributors = int((r_per_c == 1).sum())
    multi_repo_contributors = int((r_per_c > 1).sum())
    multi_repo_ratio = float(multi_repo_contributors / len(r_per_c)) if len(r_per_c) > 0 else 0.0

    # 4. Construct Bipartite Graph
    B = nx.Graph()
    for _, r in df_repos.iterrows():
        r_node = f"R_{r['repo_id']}"
        B.add_node(r_node, bipartite=1, label=r.get("full_name", r_node), type="repo")

    for _, c in df_contribs.iterrows():
        c_node = f"C_{c['contributor_id']}"
        B.add_node(c_node, bipartite=0, label=c.get("login", c_node), type="contributor")

    for _, e in df_edges.iterrows():
        c_node = f"C_{e['contributor_id']}"
        r_node = f"R_{e['repo_id']}"
        w = float(e.get("contributions_count", 1))
        B.add_edge(c_node, r_node, weight=w)

    num_nodes = B.number_of_nodes()
    total_edges = B.number_of_edges()

    # Connected Components
    components = list(nx.connected_components(B))
    num_components = len(components)
    comp_sizes = [len(c) for c in sorted(components, key=len, reverse=True)]
    gcc_nodes = comp_sizes[0] if comp_sizes else 0
    gcc_fraction = float(gcc_nodes / num_nodes) if num_nodes > 0 else 0.0

    # Unipartite Repository Projection (shared contributor overlap)
    repo_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]
    contrib_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]

    repo_proj = bipartite.weighted_projected_graph(B, repo_nodes)
    repo_proj_edges = repo_proj.number_of_edges()
    repo_proj_ccs = list(nx.connected_components(repo_proj))

    # Feasibility of Community Detection
    # If repositories are isolated or connected components are tiny isolated clusters,
    # community detection cannot cross-pollinate recommendations.
    suitability_assessment = {
        "sufficient_for_community_detection": (num_repos >= 50 and multi_repo_ratio >= 0.10 and gcc_fraction >= 0.60),
        "sufficient_for_recommendation": (num_repos >= 50 and multi_repo_contributors >= 100),
        "verdict_reasons": [],
    }

    if num_repos < 50:
        suitability_assessment["verdict_reasons"].append(
            f"Dataset size ({num_repos} repos) is insufficient for complex graph partitioning; recommended threshold is 200–500 repos."
        )
    if multi_repo_ratio < 0.10:
        suitability_assessment["verdict_reasons"].append(
            f"Only {multi_repo_ratio*100:.1f}% ({multi_repo_contributors}) of contributors span multiple repos. Without shared bridge contributors, the contributor-contributor projection is heavily fragmented."
        )
    if num_components > 1 and gcc_fraction < 0.60:
        suitability_assessment["verdict_reasons"].append(
            f"Network is partitioned into {num_components} disconnected components. Giant Connected Component contains only {gcc_fraction*100:.1f}% of nodes."
        )

    results = {
        "dataset_name": dataset_name,
        "counts": {
            "repositories": num_repos,
            "unique_repositories": num_unique_repos,
            "contributors": num_contribs,
            "unique_contributors": num_unique_contribs,
            "contributions": num_edges,
            "unique_contributions": num_unique_edges,
        },
        "quality": {
            "duplicate_repositories": dup_repos,
            "duplicate_contributors": dup_contribs,
            "duplicate_contributions": dup_edges,
            "missing_repo_ids": missing_repo_ids,
            "missing_contributor_ids": missing_contrib_ids,
            "missing_edge_ids": missing_edge_ids,
        },
        "distributions": {
            "contributors_per_repo": {
                "min": int(c_per_r.min()) if not c_per_r.empty else 0,
                "max": int(c_per_r.max()) if not c_per_r.empty else 0,
                "mean": float(c_per_r.mean()) if not c_per_r.empty else 0.0,
                "median": float(c_per_r.median()) if not c_per_r.empty else 0.0,
            },
            "repositories_per_contributor": {
                "min": int(r_per_c.min()) if not r_per_c.empty else 0,
                "max": int(r_per_c.max()) if not r_per_c.empty else 0,
                "mean": float(r_per_c.mean()) if not r_per_c.empty else 0.0,
                "single_repo_contributors": single_repo_contributors,
                "multi_repo_contributors": multi_repo_contributors,
                "multi_repo_percentage": round(multi_repo_ratio * 100, 2),
            },
        },
        "network_topology": {
            "total_nodes": num_nodes,
            "contributor_nodes": len(contrib_nodes),
            "repository_nodes": len(repo_nodes),
            "bipartite_edges": total_edges,
            "connected_components_count": num_components,
            "largest_connected_component_nodes": gcc_nodes,
            "largest_connected_component_fraction": round(gcc_fraction, 4),
            "component_sizes_top_5": comp_sizes[:5],
            "repository_projection_edges": repo_proj_edges,
            "repository_projection_components": len(repo_proj_ccs),
        },
        "suitability": suitability_assessment,
    }
    return results
