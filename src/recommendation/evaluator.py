"""Evaluation engine calculating ranking, precision, recall, F1, NDCG, coverage, and diversity."""

import math
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from src.recommendation.recommenders import BaseRecommender


def compute_dcg(relevance: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for i in range(min(k, len(relevance))):
        if relevance[i] > 0:
            dcg += relevance[i] / math.log2(i + 2)
    return dcg


def compute_idcg(num_relevant: int, k: int) -> float:
    """Compute Ideal Discounted Cumulative Gain at rank K."""
    ideal_relevance = [1] * min(num_relevant, k)
    return compute_dcg(ideal_relevance, k)


def compute_ndcg(recommended_repos: List[str], ground_truth: Set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    if not ground_truth:
        return 0.0
    relevance = [1 if repo in ground_truth else 0 for repo in recommended_repos[:k]]
    actual_dcg = compute_dcg(relevance, k)
    ideal_dcg = compute_idcg(len(ground_truth), k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0.0 else 0.0


def evaluate_recommender(
    recommender: BaseRecommender,
    test_interactions: Dict[str, Set[str]],
    all_repo_nodes: List[str],
    community_map: Optional[Dict[str, int]] = None,
    top_k_list: List[int] = [5, 10],
) -> Dict[str, Any]:
    """Evaluate recommender performance on held-out test interactions.
    
    Metrics:
    - Precision@K, Recall@K, F1@K
    - NDCG@K
    - Catalog Coverage (Fraction of total repositories ever recommended)
    - Recommendation Diversity (Intra-list community entropy / distinct communities recommended)
    """
    total_evaluated = len(test_interactions)
    valid_recs_count = 0
    skipped_count = 0

    metrics_by_k = {k: {"precision": [], "recall": [], "f1": [], "ndcg": []} for k in top_k_list}
    unique_recommended_repos: Set[str] = set()
    community_diversity_scores: List[float] = []

    for c_node, ground_truth in test_interactions.items():
        if not ground_truth:
            skipped_count += 1
            continue

        recs = recommender.recommend(c_node, top_k=max(top_k_list), exclude_known=True)
        if not recs:
            # Recommender produced no candidates (e.g. cold-start or disconnected isolate)
            for k in top_k_list:
                metrics_by_k[k]["precision"].append(0.0)
                metrics_by_k[k]["recall"].append(0.0)
                metrics_by_k[k]["f1"].append(0.0)
                metrics_by_k[k]["ndcg"].append(0.0)
            continue

        valid_recs_count += 1
        rec_repo_nodes = [item["repository_node"] for item in recs]
        unique_recommended_repos.update(rec_repo_nodes)

        # Community diversity of the recommended top-10 list
        if community_map:
            comms = [community_map.get(r, -1) for r in rec_repo_nodes[:10]]
            distinct_comms = len(set(comms))
            community_diversity_scores.append(distinct_comms / max(1, len(comms)))

        for k in top_k_list:
            top_k_repos = rec_repo_nodes[:k]
            hits = sum(1 for r in top_k_repos if r in ground_truth)

            precision = hits / k
            recall = hits / len(ground_truth)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            ndcg = compute_ndcg(top_k_repos, ground_truth, k)

            metrics_by_k[k]["precision"].append(precision)
            metrics_by_k[k]["recall"].append(recall)
            metrics_by_k[k]["f1"].append(f1)
            metrics_by_k[k]["ndcg"].append(ndcg)

    catalog_coverage = len(unique_recommended_repos) / max(1, len(all_repo_nodes))
    mean_diversity = float(np.mean(community_diversity_scores)) if community_diversity_scores else 0.0

    results = {
        "method": recommender.name,
        "evaluated_contributors": total_evaluated,
        "valid_recommendations": valid_recs_count,
        "skipped_contributors": skipped_count,
        "catalog_coverage": round(float(catalog_coverage), 4),
        "mean_community_diversity": round(mean_diversity, 4),
    }

    for k in top_k_list:
        results[f"precision@{k}"] = round(float(np.mean(metrics_by_k[k]["precision"])), 4)
        results[f"recall@{k}"] = round(float(np.mean(metrics_by_k[k]["recall"])), 4)
        results[f"f1@{k}"] = round(float(np.mean(metrics_by_k[k]["f1"])), 4)
        results[f"ndcg@{k}"] = round(float(np.mean(metrics_by_k[k]["ndcg"])), 4)

    return results
