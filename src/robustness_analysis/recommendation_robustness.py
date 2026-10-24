"""Recommendation degradation evaluation under contributor removal.

Strict Experimental Protocol:
- Uses the verified Part 5 train/test split.
- When contributors are removed:
  1. The selected contributors and their incident training edges are removed from G_train.
  2. The remaining training network G_train_post is rebuilt.
  3. Leiden communities are recomputed de novo on G_train_post (zero leakage).
  4. Models (Collaborative Filtering, Leiden Community-Aware, Jaccard, Adamic-Adar, Node2Vec)
     are refit strictly on G_train_post.
  5. Evaluated on the FIXED SURVIVING test cohort (surviving multi-repo contributors whose test edges remain unexposed).
  6. Computes Precision@5/10, Recall@5/10, F1@5/10, NDCG@5/10, Coverage, and Diversity.
  7. Computes absolute and relative changes against Part 5 baseline.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd

from src.recommendation.evaluator import evaluate_recommender
from src.recommendation.recommenders import (
    PopularityRecommender,
    CommunityAwareRecommender,
    CollaborativeFilteringRecommender,
    TopologicalSimilarityRecommender,
    Node2VecRecommender,
)


class RecommendationRobustnessEvaluator:
    """Evaluates recommendation performance before and after contributor removal."""

    def __init__(
        self,
        baseline_metrics: Dict[str, Any],
        full_test_interactions: Dict[str, Set[str]],
        all_repo_nodes: List[str],
    ):
        self.baseline_metrics = baseline_metrics
        self.full_test_interactions = full_test_interactions
        self.all_repo_nodes = all_repo_nodes

    def evaluate_post_removal(
        self,
        G_train_post: nx.Graph,
        removed_c_nodes: Set[str],
        train_leiden_comms_post: Dict[str, int],
        experiment_meta: Dict[str, Any],
        eval_node2vec: bool = False,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Fit models on post-removal training graph and evaluate surviving test cohort."""
        # 1. Define surviving evaluation cohort
        surviving_test_interactions = {
            c: ground_truth
            for c, ground_truth in self.full_test_interactions.items()
            if c not in removed_c_nodes
        }

        eval_cohort_size = len(surviving_test_interactions)
        removed_eval_users = len(self.full_test_interactions) - eval_cohort_size

        # 2. Build training repository projection
        r_nodes = [n for n, d in G_train_post.nodes(data=True) if d.get("node_type") == "repository" or d.get("bipartite") == 1]
        train_repo_proj_post = nx.bipartite.weighted_projected_graph(G_train_post, r_nodes)

        # 3. Fit models strictly on post-removal training graph
        models = [
            ("Collaborative Filtering", CollaborativeFilteringRecommender().fit(G_train_post)),
            ("Leiden Community-Aware", CommunityAwareRecommender().fit(G_train_post, community_assignments=train_leiden_comms_post)),
            ("Jaccard Similarity", TopologicalSimilarityRecommender(metric="jaccard").fit(G_train_post, train_repo_proj=train_repo_proj_post)),
            ("Adamic-Adar", TopologicalSimilarityRecommender(metric="adamic_adar").fit(G_train_post, train_repo_proj=train_repo_proj_post)),
            ("Popularity Baseline", PopularityRecommender().fit(G_train_post)),
        ]

        if eval_node2vec:
            models.append((
                "Node2Vec Embeddings",
                Node2VecRecommender(dimensions=64, walk_length=16, num_walks=10, seed=seed).fit(G_train_post),
            ))

        results = []
        for name, model in models:
            eval_dict = evaluate_recommender(
                recommender=model,
                test_interactions=surviving_test_interactions,
                all_repo_nodes=self.all_repo_nodes,
                community_map=train_leiden_comms_post,
                top_k_list=[5, 10],
            )

            # Retrieve baseline for this model to compute deltas
            base = self.baseline_metrics.get(name, {})
            base_rec10 = base.get("recall@10", 0.0)
            base_ndcg10 = base.get("ndcg@10", 0.0)
            base_rec5 = base.get("recall@5", 0.0)
            base_ndcg5 = base.get("ndcg@5", 0.0)

            post_rec10 = eval_dict["recall@10"]
            post_ndcg10 = eval_dict["ndcg@10"]
            post_rec5 = eval_dict["recall@5"]
            post_ndcg5 = eval_dict["ndcg@5"]

            abs_delta_rec10 = round(post_rec10 - base_rec10, 4)
            rel_delta_rec10 = round((post_rec10 - base_rec10) / base_rec10 * 100, 2) if base_rec10 > 0 else 0.0

            abs_delta_ndcg10 = round(post_ndcg10 - base_ndcg10, 4)
            rel_delta_ndcg10 = round((post_ndcg10 - base_ndcg10) / base_ndcg10 * 100, 2) if base_ndcg10 > 0 else 0.0

            row = {
                **experiment_meta,
                "model": name,
                "surviving_evaluated_contributors": eval_cohort_size,
                "removed_evaluated_contributors": removed_eval_users,
                "valid_recommendations": eval_dict["valid_recommendations"],
                "catalog_coverage": eval_dict["catalog_coverage"],
                "community_diversity": eval_dict["mean_community_diversity"],
                "precision@5": eval_dict["precision@5"],
                "recall@5": post_rec5,
                "f1@5": eval_dict["f1@5"],
                "ndcg@5": post_ndcg5,
                "precision@10": eval_dict["precision@10"],
                "recall@10": post_rec10,
                "f1@10": eval_dict["f1@10"],
                "ndcg@10": post_ndcg10,
                "baseline_recall@10": base_rec10,
                "baseline_ndcg@10": base_ndcg10,
                "abs_change_recall@10": abs_delta_rec10,
                "rel_change_recall@10_pct": rel_delta_rec10,
                "abs_change_ndcg@10": abs_delta_ndcg10,
                "rel_change_ndcg@10_pct": rel_delta_ndcg10,
            }
            results.append(row)

        return results
