"""Bipartite Graph Builder and Network Validator.

Constructs, validates, computes statistics, and serializes bipartite
Contributor-Repository collaboration networks using NetworkX.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import pandas as pd

from src.config.settings import Config

logger = logging.getLogger(__name__)


@dataclass
class GraphValidationReport:
    """Stores validation checks and health statistics for the constructed bipartite graph."""

    total_nodes: int = 0
    contributor_nodes: int = 0
    repository_nodes: int = 0
    total_edges: int = 0
    duplicate_nodes_detected: int = 0
    duplicate_edges_detected: int = 0
    invalid_contributor_ids: int = 0
    invalid_repo_ids: int = 0
    unexpected_node_types: int = 0
    is_strictly_bipartite: bool = False
    validation_status: str = "PENDING"
    validation_messages: List[str] = None

    def __post_init__(self):
        if self.validation_messages is None:
            self.validation_messages = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BipartiteCollaborationNetwork:
    """NetworkX-based Bipartite Contributor-Repository Collaboration Network."""

    CONTRIBUTOR_NODE_TYPE = "contributor"
    REPOSITORY_NODE_TYPE = "repository"

    def __init__(self):
        self.graph = nx.Graph()
        self.validation_report = GraphValidationReport()
        self.statistics: Dict[str, Any] = {}

    @classmethod
    def format_contributor_node_id(cls, contributor_id: int) -> str:
        """Deterministic, collision-free node identifier for contributors."""
        return f"C_{contributor_id}"

    @classmethod
    def format_repository_node_id(cls, repo_id: int) -> str:
        """Deterministic, collision-free node identifier for repositories."""
        return f"R_{repo_id}"

    def build_from_dataframes(
        self,
        df_repos: pd.DataFrame,
        df_contribs: pd.DataFrame,
        df_edges: pd.DataFrame,
        filter_bots: bool = True,
    ) -> Tuple[nx.Graph, GraphValidationReport]:
        """Construct bipartite graph from tabular dataframes with validation."""
        self.graph.clear()
        report = GraphValidationReport()

        # Identify bots if filter requested
        bot_ids: Set[int] = set()
        if filter_bots and "is_bot" in df_contribs.columns:
            bot_ids = set(df_contribs[df_contribs["is_bot"] == True]["contributor_id"])
            report.validation_messages.append(
                f"Filtered {len(bot_ids)} bot accounts from graph nodes and incident edges."
            )

        # 1. Add Contributor Nodes (bipartite=0)
        seen_contributor_nodes: Set[str] = set()
        for _, row in df_contribs.iterrows():
            cid = row.get("contributor_id")
            if pd.isna(cid) or int(cid) <= 0:
                report.invalid_contributor_ids += 1
                continue
            cid = int(cid)
            if cid in bot_ids:
                continue

            node_id = self.format_contributor_node_id(cid)
            if node_id in seen_contributor_nodes:
                report.duplicate_nodes_detected += 1
                continue
            seen_contributor_nodes.add(node_id)

            self.graph.add_node(
                node_id,
                bipartite=0,
                node_type=self.CONTRIBUTOR_NODE_TYPE,
                contributor_id=cid,
                username=str(row.get("login", "")),
                user_type=str(row.get("user_type", "User")),
                site_admin=bool(row.get("site_admin", False)),
                html_url=str(row.get("html_url", "")),
            )

        # 2. Add Repository Nodes (bipartite=1)
        seen_repo_nodes: Set[str] = set()
        for _, row in df_repos.iterrows():
            rid = row.get("repo_id")
            if pd.isna(rid) or int(rid) <= 0:
                report.invalid_repo_ids += 1
                continue
            rid = int(rid)

            node_id = self.format_repository_node_id(rid)
            if node_id in seen_repo_nodes:
                report.duplicate_nodes_detected += 1
                continue
            seen_repo_nodes.add(node_id)

            self.graph.add_node(
                node_id,
                bipartite=1,
                node_type=self.REPOSITORY_NODE_TYPE,
                repo_id=rid,
                repo_name=str(row.get("repo_name", "")),
                full_name=str(row.get("full_name", "")),
                language=str(row.get("language", "")),
                stars_count=int(row.get("stars_count", 0)),
                forks_count=int(row.get("forks_count", 0)),
                topics=str(row.get("topics", "")),
            )

        # 3. Add Edges (Contributions)
        seen_edges: Set[Tuple[str, str]] = set()
        for _, row in df_edges.iterrows():
            cid = row.get("contributor_id")
            rid = row.get("repo_id")
            if pd.isna(cid) or pd.isna(rid):
                continue
            cid, rid = int(cid), int(rid)
            if cid in bot_ids:
                continue

            c_node = self.format_contributor_node_id(cid)
            r_node = self.format_repository_node_id(rid)

            # Ensure both incident endpoints exist
            if c_node not in self.graph:
                report.invalid_contributor_ids += 1
                continue
            if r_node not in self.graph:
                report.invalid_repo_ids += 1
                continue

            edge_key = (c_node, r_node)
            if edge_key in seen_edges:
                report.duplicate_edges_detected += 1
                continue
            seen_edges.add(edge_key)

            weight = float(row.get("contributions_count", 1.0))
            self.graph.add_edge(
                c_node,
                r_node,
                weight=weight,
                relationship_type=str(row.get("relationship_type", "commits")),
            )

        # 4. Perform Graph Validation
        c_nodes = {n for n, d in self.graph.nodes(data=True) if d.get("node_type") == self.CONTRIBUTOR_NODE_TYPE}
        r_nodes = {n for n, d in self.graph.nodes(data=True) if d.get("node_type") == self.REPOSITORY_NODE_TYPE}

        report.total_nodes = self.graph.number_of_nodes()
        report.contributor_nodes = len(c_nodes)
        report.repository_nodes = len(r_nodes)
        report.total_edges = self.graph.number_of_edges()

        # Check for unexpected node types
        unknown_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") not in (self.CONTRIBUTOR_NODE_TYPE, self.REPOSITORY_NODE_TYPE)]
        report.unexpected_node_types = len(unknown_nodes)

        # Verify strict bipartiteness using NetworkX
        report.is_strictly_bipartite = bipartite.is_bipartite(self.graph)

        if report.is_strictly_bipartite and report.unexpected_node_types == 0 and report.invalid_contributor_ids == 0 and report.invalid_repo_ids == 0:
            report.validation_status = "PASSED"
            report.validation_messages.append("Strict bipartite graph conditions satisfied: no intra-set edges exist.")
        else:
            report.validation_status = "FAILED"
            report.validation_messages.append("Bipartite validation failed.")

        self.validation_report = report
        return self.graph, report

    def calculate_statistics(self) -> Dict[str, Any]:
        """Compute network statistics distinguishing contributor and repository degrees."""
        if self.graph.number_of_nodes() == 0:
            return {}

        c_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("bipartite") == 0]
        r_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("bipartite") == 1]

        num_c = len(c_nodes)
        num_r = len(r_nodes)
        num_e = self.graph.number_of_edges()

        # Network Density (Bipartite density = E / (N_c * N_r))
        max_possible_edges = num_c * num_r
        density = float(num_e / max_possible_edges) if max_possible_edges > 0 else 0.0

        # Connected Components
        ccs = list(nx.connected_components(self.graph))
        num_ccs = len(ccs)
        cc_sizes = sorted([len(c) for c in ccs], reverse=True)
        gcc_size = cc_sizes[0] if cc_sizes else 0
        gcc_fraction = float(gcc_size / self.graph.number_of_nodes()) if self.graph.number_of_nodes() > 0 else 0.0

        # Degree calculations
        c_degrees = [d for n, d in self.graph.degree(c_nodes)]
        r_degrees = [d for n, d in self.graph.degree(r_nodes)]

        def get_degree_stats(degrees: List[int]) -> Dict[str, float]:
            if not degrees:
                return {}
            arr = np.array(degrees)
            return {
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
            }

        c_deg_stats = get_degree_stats(c_degrees)
        r_deg_stats = get_degree_stats(r_degrees)

        # Multi-repo and single-repo contributor distributions
        single_repo_c = sum(1 for d in c_degrees if d == 1)
        multi_repo_c = sum(1 for d in c_degrees if d > 1)

        # Unipartite Repository Projection Statistics
        repo_proj = bipartite.weighted_projected_graph(self.graph, r_nodes)
        repo_proj_edges = repo_proj.number_of_edges()
        repo_proj_ccs = nx.number_connected_components(repo_proj)

        stats = {
            "summary": {
                "total_nodes": self.graph.number_of_nodes(),
                "contributor_nodes": num_c,
                "repository_nodes": num_r,
                "total_edges": num_e,
                "bipartite_density": round(density, 6),
                "connected_components_count": num_ccs,
                "largest_connected_component_nodes": gcc_size,
                "largest_connected_component_fraction": round(gcc_fraction, 4),
                "top_5_component_sizes": cc_sizes[:5],
            },
            "contributor_degree_distribution": {
                **c_deg_stats,
                "single_repo_contributors": single_repo_c,
                "multi_repo_contributors": multi_repo_c,
                "multi_repo_percentage": round(multi_repo_c / num_c * 100, 2) if num_c > 0 else 0.0,
            },
            "repository_degree_distribution": r_deg_stats,
            "repository_projection": {
                "projection_nodes": num_r,
                "projection_edges": repo_proj_edges,
                "projection_connected_components": repo_proj_ccs,
            },
        }

        self.statistics = stats
        return stats

    def save_graph_and_tables(self, output_dir: Path) -> Dict[str, Path]:
        """Serialize graph to GraphML, GEXF, and tabular node/edge CSVs."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # 1. Save GraphML (standard reusable graph interchange format)
        graphml_path = output_dir / "collaboration_network.graphml"
        nx.write_graphml(self.graph, graphml_path)
        paths["graphml"] = graphml_path

        # 2. Save GEXF (for Gephi / interactive viewers)
        gexf_path = output_dir / "collaboration_network.gexf"
        nx.write_gexf(self.graph, gexf_path)
        paths["gexf"] = gexf_path

        # 3. Save Node Table CSV
        nodes_data = []
        for n, d in self.graph.nodes(data=True):
            node_row = {
                "node_id": n,
                "bipartite": d.get("bipartite"),
                "node_type": d.get("node_type"),
                "contributor_id": d.get("contributor_id"),
                "username": d.get("username"),
                "repo_id": d.get("repo_id"),
                "repo_name": d.get("repo_name"),
                "full_name": d.get("full_name"),
                "degree": self.graph.degree(n),
            }
            nodes_data.append(node_row)
        df_nodes = pd.DataFrame(nodes_data)
        nodes_csv_path = output_dir / "network_nodes.csv"
        df_nodes.to_csv(nodes_csv_path, index=False)
        paths["nodes_csv"] = nodes_csv_path

        # 4. Save Edge Table CSV
        edges_data = []
        for u, v, d in self.graph.edges(data=True):
            # Ensure contributor is source, repo is target
            c_node = u if self.graph.nodes[u].get("bipartite") == 0 else v
            r_node = v if self.graph.nodes[v].get("bipartite") == 1 else u
            edges_data.append({
                "source_contributor": c_node,
                "target_repository": r_node,
                "contributor_id": self.graph.nodes[c_node].get("contributor_id"),
                "username": self.graph.nodes[c_node].get("username"),
                "repo_id": self.graph.nodes[r_node].get("repo_id"),
                "repo_name": self.graph.nodes[r_node].get("full_name"),
                "weight": d.get("weight", 1.0),
                "relationship_type": d.get("relationship_type", "commits"),
            })
        df_edges = pd.DataFrame(edges_data)
        edges_csv_path = output_dir / "network_edges.csv"
        df_edges.to_csv(edges_csv_path, index=False)
        paths["edges_csv"] = edges_csv_path

        # 5. Save validation report and network stats
        with open(output_dir / "graph_validation_report.json", "w", encoding="utf-8") as f:
            json.dump(self.validation_report.to_dict(), f, indent=2)

        with open(output_dir / "network_statistics.json", "w", encoding="utf-8") as f:
            json.dump(self.statistics, f, indent=2)

        return paths
