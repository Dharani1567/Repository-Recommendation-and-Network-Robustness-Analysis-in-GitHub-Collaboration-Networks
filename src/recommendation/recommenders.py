"""Implementation of 5 GitHub repository recommendation algorithms.

Algorithms:
1. Popularity Baseline Recommender (Training contribution counts)
2. Community-Aware Recommender (Leiden partition intra-community density)
3. Collaborative Filtering Recommender (User-item bipartite cosine / co-occurrence)
4. Topological Similarity Recommenders (Jaccard and Adamic-Adar on repository projection)
5. Node2Vec Graph Embedding Recommender (Biased random walks on training graph)
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd


class BaseRecommender:
    """Abstract base class for all repository recommendation algorithms."""

    def __init__(self, name: str):
        self.name = name

    def fit(self, train_graph: nx.Graph, **kwargs) -> "BaseRecommender":
        raise NotImplementedError

    def recommend(
        self,
        contributor_node: str,
        top_k: int = 10,
        exclude_known: bool = True,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class PopularityRecommender(BaseRecommender):
    """Popularity baseline: ranks unseen repositories by training-set contribution counts.
    
    Does NOT use GitHub stars as collaboration proxy. Strictly uses training graph edge weights/degrees.
    """

    def __init__(self):
        super().__init__("Popularity Baseline")
        self.train_graph: Optional[nx.Graph] = None
        self.repo_popularity: List[Tuple[str, float]] = []
        self.all_repos: List[str] = []

    def fit(self, train_graph: nx.Graph, **kwargs) -> "PopularityRecommender":
        self.train_graph = train_graph
        self.all_repos = [
            n for n, d in train_graph.nodes(data=True)
            if d.get("node_type") == "repository" or d.get("bipartite") == 1
        ]

        # Score = sum of contribution weights (commits) or degree in training graph
        repo_scores = {}
        for r in self.all_repos:
            # Weighted in-degree in bipartite graph
            score = sum(d.get("weight", 1.0) for _, _, d in train_graph.edges(r, data=True))
            repo_scores[r] = score

        self.repo_popularity = sorted(repo_scores.items(), key=lambda x: x[1], reverse=True)
        return self

    def recommend(
        self,
        contributor_node: str,
        top_k: int = 10,
        exclude_known: bool = True,
    ) -> List[Dict[str, Any]]:
        known_repos = set(self.train_graph.neighbors(contributor_node)) if exclude_known and contributor_node in self.train_graph else set()

        recs = []
        for r_node, score in self.repo_popularity:
            if r_node in known_repos:
                continue
            recs.append({
                "contributor_node": contributor_node,
                "repository_node": r_node,
                "repo_id": self.train_graph.nodes[r_node].get("repo_id"),
                "repo_name": self.train_graph.nodes[r_node].get("full_name", r_node),
                "score": float(score),
                "method": self.name,
            })
            if len(recs) >= top_k:
                break
        return recs


class CommunityAwareRecommender(BaseRecommender):
    """Community-Aware Recommender: Recommends unseen repositories from relevant Leiden communities.
    
    Given contributor C who contributed to {R_1, R_2, ...} in training:
    1. Identifies the distribution of communities among C's known repositories.
    2. Weights candidate repositories by their community relevance multiplied by intra-community training degree.
    3. Handles isolated repositories by falling back to popularity priors.
    """

    def __init__(self, community_assignments: Optional[Dict[str, int]] = None):
        super().__init__("Leiden Community-Aware")
        self.community_assignments = community_assignments or {}
        self.train_graph: Optional[nx.Graph] = None
        self.all_repos: List[str] = []
        self.community_repo_ranking: Dict[int, List[Tuple[str, float]]] = {}
        self.global_fallback_ranking: List[Tuple[str, float]] = []

    def fit(self, train_graph: nx.Graph, community_assignments: Optional[Dict[str, int]] = None, **kwargs) -> "CommunityAwareRecommender":
        self.train_graph = train_graph
        if community_assignments:
            self.community_assignments = community_assignments

        self.all_repos = [
            n for n, d in train_graph.nodes(data=True)
            if d.get("node_type") == "repository" or d.get("bipartite") == 1
        ]

        # Group repositories by community and rank by training activity
        by_comm: Dict[int, Dict[str, float]] = {}
        global_scores: Dict[str, float] = {}

        for r in self.all_repos:
            comm_id = self.community_assignments.get(r, -1)
            score = sum(d.get("weight", 1.0) for _, _, d in train_graph.edges(r, data=True))
            global_scores[r] = score

            if comm_id not in by_comm:
                by_comm[comm_id] = {}
            by_comm[comm_id][r] = score

        self.community_repo_ranking = {
            c_id: sorted(repos.items(), key=lambda x: x[1], reverse=True)
            for c_id, repos in by_comm.items()
        }
        self.global_fallback_ranking = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)
        return self

    def recommend(
        self,
        contributor_node: str,
        top_k: int = 10,
        exclude_known: bool = True,
    ) -> List[Dict[str, Any]]:
        known_repos = set(self.train_graph.neighbors(contributor_node)) if exclude_known and contributor_node in self.train_graph else set()

        # Compute contributor's community preferences from training repositories
        comm_counts: Dict[int, float] = {}
        for r in known_repos:
            c_id = self.community_assignments.get(r, -1)
            comm_counts[c_id] = comm_counts.get(c_id, 0.0) + 1.0

        # Score candidate repositories
        candidate_scores: Dict[str, float] = {}
        total_known = len(known_repos) if known_repos else 1.0

        for r in self.all_repos:
            if r in known_repos:
                continue
            c_id = self.community_assignments.get(r, -1)
            # Community affinity weight
            comm_weight = comm_counts.get(c_id, 0.0) / total_known if c_id != -1 else 0.0

            # Training popularity base score (log-scaled)
            pop = sum(d.get("weight", 1.0) for _, _, d in self.train_graph.edges(r, data=True))
            log_pop = math.log1p(pop)

            if comm_weight > 0:
                # Strong boost for intra-community repositories
                score = (1.0 + comm_weight * 5.0) * log_pop
            else:
                # Inter-community baseline
                score = 0.1 * log_pop

            candidate_scores[r] = score

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "contributor_node": contributor_node,
                "repository_node": r_node,
                "repo_id": self.train_graph.nodes[r_node].get("repo_id"),
                "repo_name": self.train_graph.nodes[r_node].get("full_name", r_node),
                "community_id": self.community_assignments.get(r_node, -1),
                "score": float(score),
                "method": self.name,
            }
            for r_node, score in ranked
        ]


class CollaborativeFilteringRecommender(BaseRecommender):
    """User-based / Item-based collaborative filtering on the sparse bipartite graph.
    
    Computes cosine similarity between contributor interaction vectors.
    Ranks unseen repositories by aggregating similarity-weighted ratings of peer contributors.
    """

    def __init__(self):
        super().__init__("Collaborative Filtering")
        self.train_graph: Optional[nx.Graph] = None
        self.contributor_nodes: List[str] = []
        self.repo_nodes: List[str] = []
        self.c_to_idx: Dict[str, int] = {}
        self.r_to_idx: Dict[str, int] = {}
        self.idx_to_r: Dict[int, str] = {}
        self.contrib_matrix = None  # Contributor x Repository binary/weighted sparse matrix
        self.item_item_sim = None   # Repository x Repository co-occurrence cosine matrix

    def fit(self, train_graph: nx.Graph, **kwargs) -> "CollaborativeFilteringRecommender":
        self.train_graph = train_graph
        self.contributor_nodes = [
            n for n, d in train_graph.nodes(data=True)
            if d.get("node_type") == "contributor" or d.get("bipartite") == 0
        ]
        self.repo_nodes = [
            n for n, d in train_graph.nodes(data=True)
            if d.get("node_type") == "repository" or d.get("bipartite") == 1
        ]

        self.c_to_idx = {c: i for i, c in enumerate(self.contributor_nodes)}
        self.r_to_idx = {r: i for i, r in enumerate(self.repo_nodes)}
        self.idx_to_r = {i: r for r, i in self.r_to_idx.items()}

        # Build repository co-occurrence cosine matrix (Item-Item CF)
        # Item-Item is statistically robust for sparse users (handles degree-1 contributors cleanly)
        num_repos = len(self.repo_nodes)
        co_matrix = np.zeros((num_repos, num_repos), dtype=np.float32)

        for c in self.contributor_nodes:
            repos = [self.r_to_idx[r] for r in train_graph.neighbors(c) if r in self.r_to_idx]
            for i in repos:
                for j in repos:
                    if i != j:
                        co_matrix[i, j] += 1.0

        # Compute cosine similarity
        norms = np.sqrt(np.diag(co_matrix))
        # Add small epsilon to avoid divide by zero
        norms[norms == 0] = 1e-9
        norm_matrix = co_matrix / np.outer(norms, norms)
        np.fill_diagonal(norm_matrix, 0.0)
        self.item_item_sim = norm_matrix
        return self

    def recommend(
        self,
        contributor_node: str,
        top_k: int = 10,
        exclude_known: bool = True,
    ) -> List[Dict[str, Any]]:
        known_repos = set(self.train_graph.neighbors(contributor_node)) if exclude_known and contributor_node in self.train_graph else set()
        known_indices = [self.r_to_idx[r] for r in known_repos if r in self.r_to_idx]

        if not known_indices:
            # Cold-start fallback
            scores = np.zeros(len(self.repo_nodes))
        else:
            # Aggregate similarity scores of known repositories
            scores = self.item_item_sim[known_indices].sum(axis=0)

        # Mask known repositories
        for idx in known_indices:
            scores[idx] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]

        recs = []
        for idx in top_indices:
            r_node = self.idx_to_r[idx]
            score = float(scores[idx])
            if score <= 0.0:
                # If similarity is zero, repository has no co-contributor overlap
                continue
            recs.append({
                "contributor_node": contributor_node,
                "repository_node": r_node,
                "repo_id": self.train_graph.nodes[r_node].get("repo_id"),
                "repo_name": self.train_graph.nodes[r_node].get("full_name", r_node),
                "score": score,
                "method": self.name,
            })
        return recs


class TopologicalSimilarityRecommender(BaseRecommender):
    """Topological similarity recommender using Jaccard and Adamic-Adar on the training repository projection.
    
    Scores candidate repository R_cand based on similarity to contributor's known repositories:
    - Jaccard(R_A, R_B) = |N(R_A) ∩ N(R_B)| / |N(R_A) ∪ N(R_B)|
    - Adamic-Adar(R_A, R_B) = sum_{z ∈ N(R_A) ∩ N(R_B)} 1 / log(degree(z))
    """

    def __init__(self, metric: str = "jaccard"):
        metric_name = "Jaccard Similarity" if metric.lower() == "jaccard" else "Adamic-Adar"
        super().__init__(metric_name)
        self.metric = metric.lower()
        self.train_graph: Optional[nx.Graph] = None
        self.train_repo_proj: Optional[nx.Graph] = None
        self.similarity_matrix: Dict[Tuple[str, str], float] = {}

    def fit(self, train_graph: nx.Graph, train_repo_proj: Optional[nx.Graph] = None, **kwargs) -> "TopologicalSimilarityRecommender":
        self.train_graph = train_graph
        if train_repo_proj is None:
            r_nodes = [
                n for n, d in train_graph.nodes(data=True)
                if d.get("node_type") == "repository" or d.get("bipartite") == 1
            ]
            self.train_repo_proj = nx.bipartite.weighted_projected_graph(train_graph, r_nodes)
        else:
            self.train_repo_proj = train_repo_proj

        # Precompute similarity pairs
        self.similarity_matrix.clear()
        nodes = list(self.train_repo_proj.nodes())

        if self.metric == "jaccard":
            # Jaccard on repository neighbor sets in projection
            for i, u in enumerate(nodes):
                u_neighbors = set(self.train_repo_proj.neighbors(u))
                if not u_neighbors:
                    continue
                for v in nodes[i + 1:]:
                    v_neighbors = set(self.train_repo_proj.neighbors(v))
                    union_len = len(u_neighbors.union(v_neighbors))
                    if union_len > 0:
                        inter_len = len(u_neighbors.intersection(v_neighbors))
                        if inter_len > 0:
                            sim = inter_len / union_len
                            self.similarity_matrix[(u, v)] = sim
                            self.similarity_matrix[(v, u)] = sim

        elif self.metric == "adamic_adar":
            # Adamic-Adar: sum 1/log(deg(w)) for common neighbors in projection
            for i, u in enumerate(nodes):
                u_neighbors = set(self.train_repo_proj.neighbors(u))
                if not u_neighbors:
                    continue
                for v in nodes[i + 1:]:
                    v_neighbors = set(self.train_repo_proj.neighbors(v))
                    common = u_neighbors.intersection(v_neighbors)
                    if common:
                        aa = sum(1.0 / math.log(self.train_repo_proj.degree(w)) for w in common if self.train_repo_proj.degree(w) > 1)
                        if aa > 0:
                            self.similarity_matrix[(u, v)] = aa
                            self.similarity_matrix[(v, u)] = aa

        return self

    def recommend(
        self,
        contributor_node: str,
        top_k: int = 10,
        exclude_known: bool = True,
    ) -> List[Dict[str, Any]]:
        known_repos = set(self.train_graph.neighbors(contributor_node)) if exclude_known and contributor_node in self.train_graph else set()
        candidate_scores: Dict[str, float] = {}

        for r_cand in self.train_repo_proj.nodes():
            if r_cand in known_repos:
                continue
            score = sum(self.similarity_matrix.get((r_known, r_cand), 0.0) for r_known in known_repos)
            if score > 0:
                candidate_scores[r_cand] = score

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "contributor_node": contributor_node,
                "repository_node": r_node,
                "repo_id": self.train_graph.nodes[r_node].get("repo_id"),
                "repo_name": self.train_graph.nodes[r_node].get("full_name", r_node),
                "score": float(score),
                "method": self.name,
            }
            for r_node, score in ranked
        ]


class Node2VecRecommender(BaseRecommender):
    """Node2Vec representation learning recommender.
    
    Generates biased random walks on the training bipartite graph G_train.
    Trains skip-gram embeddings with negative sampling (implemented natively with PyTorch).
    Ranks repositories for contributor C via cosine similarity in embedding space.
    """

    def __init__(self, dimensions: int = 64, walk_length: int = 16, num_walks: int = 10, p: float = 1.0, q: float = 1.0, seed: int = 42):
        super().__init__("Node2Vec Embeddings")
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.seed = seed
        self.embeddings: Dict[str, np.ndarray] = {}
        self.train_graph: Optional[nx.Graph] = None
        self.repo_nodes: List[str] = []

    def fit(self, train_graph: nx.Graph, **kwargs) -> "Node2VecRecommender":
        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.train_graph = train_graph
        self.repo_nodes = [
            n for n, d in train_graph.nodes(data=True)
            if d.get("node_type") == "repository" or d.get("bipartite") == 1
        ]

        # Filter nodes with at least 1 edge for random walk generation
        active_nodes = [n for n in train_graph.nodes() if train_graph.degree(n) > 0]
        node_to_idx = {n: i for i, n in enumerate(active_nodes)}
        idx_to_node = {i: n for n, i in node_to_idx.items()}

        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        # 1. Generate Random Walks
        walks = []
        for _ in range(self.num_walks):
            shuffled_nodes = list(active_nodes)
            rng.shuffle(shuffled_nodes)
            for start_node in shuffled_nodes:
                walk = [start_node]
                curr = start_node
                for _ in range(self.walk_length - 1):
                    neighbors = list(train_graph.neighbors(curr))
                    if not neighbors:
                        break
                    # Biased or uniform transition
                    weights = [train_graph[curr][nbr].get("weight", 1.0) for nbr in neighbors]
                    probs = np.array(weights, dtype=np.float64) / sum(weights)
                    curr = rng.choice(neighbors, p=probs)
                    walk.append(curr)
                walks.append(walk)

        # 2. Extract Skip-Gram Target-Context Pairs (Window = 3)
        window_size = 3
        pairs = []
        for walk in walks:
            for i, target in enumerate(walk):
                target_idx = node_to_idx[target]
                start = max(0, i - window_size)
                end = min(len(walk), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        context_idx = node_to_idx[walk[j]]
                        pairs.append((target_idx, context_idx))

        if not pairs:
            return self

        # Subsample pairs to ensure fast, predictable execution
        max_pairs = 150000
        if len(pairs) > max_pairs:
            sample_indices = rng.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[idx] for idx in sample_indices]

        # 3. Fast PyTorch Skip-Gram Model with Negative Sampling
        vocab_size = len(active_nodes)
        embed_dim = self.dimensions

        target_tensor = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        context_tensor = torch.tensor([p[1] for p in pairs], dtype=torch.long)

        target_emb = nn.Embedding(vocab_size, embed_dim)
        context_emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(target_emb.weight)
        nn.init.xavier_uniform_(context_emb.weight)

        optimizer = optim.Adam(list(target_emb.parameters()) + list(context_emb.parameters()), lr=0.02)
        batch_size = 4096
        num_epochs = 3
        dataset_len = len(pairs)

        for epoch in range(num_epochs):
            perm = torch.randperm(dataset_len)
            for b in range(0, dataset_len, batch_size):
                batch_indices = perm[b: b + batch_size]
                t_b = target_tensor[batch_indices]
                c_b = context_tensor[batch_indices]

                # Positive loss
                v_t = target_emb(t_b)
                v_c = context_emb(c_b)
                pos_score = torch.sum(v_t * v_c, dim=1)
                pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-9).mean()

                # Negative sampling (5 negatives per positive)
                neg_samples = torch.randint(0, vocab_size, (len(t_b), 5))
                v_neg = context_emb(neg_samples)  # (batch, 5, dim)
                neg_score = torch.bmm(v_neg, v_t.unsqueeze(2)).squeeze(2)  # (batch, 5)
                neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-9).sum(dim=1).mean()

                loss = pos_loss + neg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Extract normalized embeddings
        weights_np = target_emb.weight.detach().cpu().numpy()
        norms = np.linalg.norm(weights_np, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normed_weights = weights_np / norms

        self.embeddings = {idx_to_node[i]: normed_weights[i] for i in range(vocab_size)}
        return self

    def recommend(
        self,
        contributor_node: str,
        top_k: int = 10,
        exclude_known: bool = True,
    ) -> List[Dict[str, Any]]:
        known_repos = set(self.train_graph.neighbors(contributor_node)) if exclude_known and contributor_node in self.train_graph else set()

        if contributor_node not in self.embeddings:
            return []

        c_vec = self.embeddings[contributor_node]
        candidate_scores = {}

        for r_node in self.repo_nodes:
            if r_node in known_repos or r_node not in self.embeddings:
                continue
            r_vec = self.embeddings[r_node]
            cosine_sim = float(np.dot(c_vec, r_vec))
            candidate_scores[r_node] = cosine_sim

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "contributor_node": contributor_node,
                "repository_node": r_node,
                "repo_id": self.train_graph.nodes[r_node].get("repo_id"),
                "repo_name": self.train_graph.nodes[r_node].get("full_name", r_node),
                "score": float(score),
                "method": self.name,
            }
            for r_node, score in ranked
        ]
