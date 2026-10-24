"""Bipartite Network Visualization Module.

Generates:
1. Static publication-grade matplotlib/seaborn figure with distinct representations for
   contributors (circles) and repositories (squares/hexagons), sized by degree and colored by domain.
2. Interactive HTML graph visualization with Plotly for dynamic exploration of subgraphs.
"""

from pathlib import Path
from typing import List, Optional, Set
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def create_subgraph_visualization(
    B: nx.Graph,
    output_dir: Path,
    seed_repos: Optional[List[str]] = None,
    max_repos: int = 12,
    max_contributors: int = 80,
    figure_prefix: str = "collaboration_network_subgraph",
) -> dict:
    """Create a manageable, clean visualization of a core subset of the bipartite network.
    
    Distinguishes:
    - Contributor nodes: Circles (soft coral/orange), sized by degree
    - Repository nodes: Distinct large squares/hexagons (deep navy/teal), labeled with full name
    - Cross-cutting bridge edges: highlighted in bold blue
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    r_nodes = [n for n, d in B.nodes(data=True) if d.get("node_type") == "repository"]
    c_nodes = [n for n, d in B.nodes(data=True) if d.get("node_type") == "contributor"]

    # Select representative repositories (either provided seeds or top-degree repos)
    selected_r_nodes: List[str] = []
    if seed_repos:
        for r in r_nodes:
            full_name = B.nodes[r].get("full_name", "")
            if any(s.lower() in full_name.lower() for s in seed_repos):
                selected_r_nodes.append(r)
                if len(selected_r_nodes) >= max_repos:
                    break

    if len(selected_r_nodes) < 4:
        # Fallback to highest degree repositories
        r_degree_sorted = sorted(r_nodes, key=lambda n: B.degree(n), reverse=True)
        selected_r_nodes = r_degree_sorted[:max_repos]

    # Find neighbor contributors of the selected repos
    candidate_c_nodes: Set[str] = set()
    for r in selected_r_nodes:
        candidate_c_nodes.update(B.neighbors(r))

    # Prioritize multi-repository bridge contributors to show network interconnectivity
    bridge_c = [c for c in candidate_c_nodes if sum(1 for n in B.neighbors(c) if n in selected_r_nodes) > 1]
    single_c = [c for c in candidate_c_nodes if c not in bridge_c]

    # Combine prioritized bridges with a sample of single contributors
    final_c_nodes = bridge_c + single_c[: max(0, max_contributors - len(bridge_c))]

    sub_nodes = set(selected_r_nodes).union(final_c_nodes)
    subgraph = B.subgraph(sub_nodes).copy()

    # 1. Matplotlib Visualization
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    pos = nx.spring_layout(subgraph, k=0.45, iterations=80, seed=42)

    sub_r = [n for n in selected_r_nodes if n in subgraph]
    sub_c = [n for n in final_c_nodes if n in subgraph]

    # Draw edges: bridge edges in royal blue, single edges in light gray
    bridge_edges = []
    normal_edges = []
    for u, v in subgraph.edges():
        c_node = u if subgraph.nodes[u].get("bipartite") == 0 else v
        # Count connections to selected repos
        if sum(1 for n in subgraph.neighbors(c_node) if n in sub_r) > 1:
            bridge_edges.append((u, v))
        else:
            normal_edges.append((u, v))

    nx.draw_networkx_edges(subgraph, pos, edgelist=normal_edges, ax=ax, alpha=0.35, edge_color="#bdc3c7", width=1.0)
    nx.draw_networkx_edges(subgraph, pos, edgelist=bridge_edges, ax=ax, alpha=0.75, edge_color="#2980b9", width=2.2)

    # Draw Contributor Nodes (Circles, Coral/Gold)
    c_sizes = [max(40, subgraph.degree(n) * 35) for n in sub_c]
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=sub_c,
        ax=ax,
        node_shape="o",
        node_size=c_sizes,
        node_color="#e67e22",
        alpha=0.85,
        edgecolors="#d35400",
        linewidths=1.0,
        label="Contributors (Circles)",
    )

    # Draw Repository Nodes (Squares, Deep Navy)
    r_sizes = [max(400, subgraph.degree(n) * 30) for n in sub_r]
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=sub_r,
        ax=ax,
        node_shape="s",
        node_size=r_sizes,
        node_color="#2c3e50",
        alpha=0.95,
        edgecolors="#1a252f",
        linewidths=2.0,
        label="Repositories (Squares)",
    )

    # Labels for repositories
    r_labels = {n: subgraph.nodes[n].get("repo_name", n) for n in sub_r}
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels=r_labels,
        ax=ax,
        font_size=9,
        font_weight="bold",
        font_color="#ffffff",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#2c3e50", edgecolor="#1a252f", alpha=0.9),
    )

    # Labels for key bridge contributors
    top_bridges = sorted(bridge_c, key=lambda n: subgraph.degree(n), reverse=True)[:10]
    bridge_labels = {n: subgraph.nodes[n].get("username", n) for n in top_bridges if n in subgraph}
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels=bridge_labels,
        ax=ax,
        font_size=7,
        font_color="#78281f",
        font_family="sans-serif",
    )

    ax.set_title(
        f"GitHub Bipartite Collaboration Network Subgraph\n"
        f"({len(sub_r)} Repositories, {len(sub_c)} Contributors, {len(bridge_c)} Cross-Project Bridges)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.axis("off")
    plt.tight_layout()

    png_path = output_dir / f"{figure_prefix}.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    results["static_png"] = png_path

    # 2. Interactive Plotly Visualization
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.0, color="#95a5a6"),
        hoverinfo="none",
        mode="lines",
    )

    # Contributor Scatter
    c_x = [pos[n][0] for n in sub_c]
    c_y = [pos[n][1] for n in sub_c]
    c_text = [
        f"<b>Contributor:</b> {subgraph.nodes[n].get('username')}<br>"
        f"ID: {subgraph.nodes[n].get('contributor_id')}<br>"
        f"Degree (Repos): {subgraph.degree(n)}"
        for n in sub_c
    ]
    contributor_trace = go.Scatter(
        x=c_x,
        y=c_y,
        mode="markers",
        hoverinfo="text",
        text=c_text,
        name="Contributors",
        marker=dict(
            symbol="circle",
            size=[max(8, subgraph.degree(n) * 4) for n in sub_c],
            color="#e67e22",
            line_width=1,
            line_color="#d35400",
        ),
    )

    # Repository Scatter
    r_x = [pos[n][0] for n in sub_r]
    r_y = [pos[n][1] for n in sub_r]
    r_text = [
        f"<b>Repository:</b> {subgraph.nodes[n].get('full_name')}<br>"
        f"Stars: {subgraph.nodes[n].get('stars_count')}<br>"
        f"Degree (Contributors): {subgraph.degree(n)}"
        for n in sub_r
    ]
    repo_trace = go.Scatter(
        x=r_x,
        y=r_y,
        mode="markers+text",
        hoverinfo="text",
        text=r_text,
        textposition="top center",
        name="Repositories",
        marker=dict(
            symbol="square",
            size=18,
            color="#2c3e50",
            line_width=2,
            line_color="#1a252f",
        ),
    )

    fig_plotly = go.Figure(
        data=[edge_trace, contributor_trace, repo_trace],
        layout=go.Layout(
            title=dict(
                text="Interactive GitHub Bipartite Collaboration Network (Subset)",
                font=dict(size=16),
            ),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#f8f9fa",
        ),
    )

    html_path = output_dir / f"{figure_prefix}.html"
    fig_plotly.write_html(str(html_path))
    results["interactive_html"] = html_path

    return results
