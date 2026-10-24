"""Pipeline runner to build, validate, calculate statistics, and visualize the Bipartite Collaboration Network."""

import json
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from src.config.settings import Config
from src.network_construction.graph_builder import BipartiteCollaborationNetwork
from src.network_construction.visualizer import create_subgraph_visualization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_network_construction")


def run_network_construction_pipeline(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    filter_bots: bool = True,
) -> dict:
    """Build, validate, compute statistics, and visualize the bipartite network."""
    input_path = data_dir or (Config.DATA_DIR / "expanded_processed")
    out_path = output_dir or (Config.DATA_DIR / "network")
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading datasets from {input_path}...")
    df_repos = pd.read_csv(input_path / "repositories.csv")
    df_contribs = pd.read_csv(input_path / "contributors.csv")
    df_edges = pd.read_csv(input_path / "contributions.csv")

    logger.info(f"Building Bipartite Graph from {len(df_repos)} repos, {len(df_contribs)} contributors, {len(df_edges)} edges...")
    builder = BipartiteCollaborationNetwork()
    B, validation_report = builder.build_from_dataframes(
        df_repos=df_repos,
        df_contribs=df_contribs,
        df_edges=df_edges,
        filter_bots=filter_bots,
    )

    logger.info("Computing network topology statistics...")
    stats = builder.calculate_statistics()

    logger.info(f"Saving graph files and node/edge tables to {out_path}...")
    saved_files = builder.save_graph_and_tables(out_path)

    logger.info("Generating static and interactive network visualizations...")
    viz_files = create_subgraph_visualization(
        B=B,
        output_dir=out_path / "visualizations",
        seed_repos=["pytorch", "transformers", "scikit-learn", "numpy", "pandas", "fastapi", "flask"],
        max_repos=10,
        max_contributors=70,
    )

    summary = {
        "status": "success",
        "validation": validation_report.to_dict(),
        "statistics": stats,
        "saved_graph_files": {k: str(v) for k, v in saved_files.items()},
        "visualization_files": {k: str(v) for k, v in viz_files.items()},
    }

    # Save summary markdown report
    report_md = f"""# Bipartite Network Construction & Validation Report

## 1. Graph Validation Summary
- **Validation Status**: {validation_report.validation_status}
- **Is Strictly Bipartite**: {validation_report.is_strictly_bipartite}
- **Total Nodes**: {validation_report.total_nodes:,}
  - **Contributor Nodes (Type: contributor)**: {validation_report.contributor_nodes:,}
  - **Repository Nodes (Type: repository)**: {validation_report.repository_nodes:,}
- **Total Edges (Contributions)**: {validation_report.total_edges:,}
- **Duplicate Nodes Detected**: {validation_report.duplicate_nodes_detected}
- **Duplicate Edges Detected**: {validation_report.duplicate_edges_detected}
- **Invalid Contributor IDs**: {validation_report.invalid_contributor_ids}
- **Invalid Repository IDs**: {validation_report.invalid_repo_ids}
- **Unexpected Node Types**: {validation_report.unexpected_node_types}

## 2. Network Statistics
- **Bipartite Density**: {stats['summary']['bipartite_density']}
- **Connected Components Count**: {stats['summary']['connected_components_count']}
- **Largest Connected Component (GCC)**: {stats['summary']['largest_connected_component_nodes']:,} nodes ({stats['summary']['largest_connected_component_fraction']*100:.2f}% of network)
- **Component Sizes (Top 5)**: {stats['summary']['top_5_component_sizes']}

### Contributor Degree Distribution (Repositories per Developer)
- **Min**: {stats['contributor_degree_distribution']['min']}
- **Max**: {stats['contributor_degree_distribution']['max']}
- **Mean**: {stats['contributor_degree_distribution']['mean']:.3f}
- **Median**: {stats['contributor_degree_distribution']['median']}
- **Std Dev**: {stats['contributor_degree_distribution']['std']:.3f}
- **Single-Repo Contributors**: {stats['contributor_degree_distribution']['single_repo_contributors']:,} ({100 - stats['contributor_degree_distribution']['multi_repo_percentage']:.2f}%)
- **Multi-Repo Bridge Contributors**: {stats['contributor_degree_distribution']['multi_repo_contributors']:,} ({stats['contributor_degree_distribution']['multi_repo_percentage']:.2f}%)

### Repository Degree Distribution (Contributors per Repo)
- **Min**: {stats['repository_degree_distribution']['min']}
- **Max**: {stats['repository_degree_distribution']['max']}
- **Mean**: {stats['repository_degree_distribution']['mean']:.2f}
- **Median**: {stats['repository_degree_distribution']['median']}
- **Std Dev**: {stats['repository_degree_distribution']['std']:.2f}

### Projected Repository Co-Contribution Graph
- **Projection Nodes**: {stats['repository_projection']['projection_nodes']}
- **Projection Edges**: {stats['repository_projection']['projection_edges']:,}
- **Projection Connected Components**: {stats['repository_projection']['projection_connected_components']}

## 3. Saved Graph & Table Files
- **GraphML File**: `{saved_files.get('graphml')}`
- **GEXF File**: `{saved_files.get('gexf')}`
- **Nodes CSV**: `{saved_files.get('nodes_csv')}`
- **Edges CSV**: `{saved_files.get('edges_csv')}`

## 4. Visualizations
- **Static Publication Figure**: `{viz_files.get('static_png')}`
- **Interactive Plotly Graph**: `{viz_files.get('interactive_html')}`
"""
    with open(out_path / "network_construction_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("Bipartite Network Construction pipeline finished successfully.")
    return summary


if __name__ == "__main__":
    summary = run_network_construction_pipeline()
    print(f"Network built: {summary['validation']['total_nodes']} nodes, {summary['validation']['total_edges']} edges. Status: {summary['validation']['validation_status']}")
