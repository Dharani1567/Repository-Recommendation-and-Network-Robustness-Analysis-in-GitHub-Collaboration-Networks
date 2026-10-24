# GitHub Repository Recommendation System Benchmark (Part 5)

- **Evaluation Date**: 2026-10-25
- **Input Graph**: `data/network/collaboration_network.graphml` (4,759 contributors, 206 repositories)
- **Train/Test Protocol**: Controlled Contributor Interaction Split (Seed: 42)
  - **Train Contribution Edges**: 5,181
  - **Held-Out Test Ground-Truth Edges**: 577
  - **Evaluated Multi-Repository Contributors**: 562
  - **Zero Data Leakage Guarantee**: Leiden communities and repository similarity projections were computed strictly on the training graph after edge holdout.

---

## 1. Experimental Benchmark Results

| method                  |   precision@5 |   recall@5 |   f1@5 |   precision@10 |   recall@10 |   f1@10 |   ndcg@5 |   ndcg@10 |   catalog_coverage |   mean_community_diversity |
|:------------------------|--------------:|-----------:|-------:|---------------:|------------:|--------:|---------:|----------:|-------------------:|---------------------------:|
| Popularity Baseline     |        0.0028 |     0.0142 | 0.0047 |         0.003  |      0.0302 |  0.0055 |   0.0061 |    0.011  |             0.0534 |                     0.3    |
| Leiden Community-Aware  |        0.0192 |     0.0939 | 0.0317 |         0.0206 |      0.1993 |  0.0373 |   0.0545 |    0.0885 |             0.5194 |                     0.1283 |
| Collaborative Filtering |        0.0669 |     0.3287 | 0.1109 |         0.039  |      0.379  |  0.0704 |   0.2418 |    0.2589 |             0.733  |                     0.3859 |
| Jaccard Similarity      |        0.0498 |     0.2429 | 0.0823 |         0.0359 |      0.351  |  0.065  |   0.1681 |    0.2024 |             0.7233 |                     0.2704 |
| Adamic-Adar             |        0.0473 |     0.2282 | 0.0779 |         0.0311 |      0.302  |  0.0562 |   0.1625 |    0.1864 |             0.6505 |                     0.2935 |
| Node2Vec Embeddings     |        0.0466 |     0.2251 | 0.0768 |         0.0292 |      0.2825 |  0.0527 |   0.1601 |    0.1791 |             1      |                     0.4984 |

---

## 2. Key Analytical Insights

### A. Popularity Baseline vs. Community-Aware Recommendation
- **Popularity Baseline** (0.014 Recall@5) recommends universally active repositories (`pandas`, `numpy`, `transformers`). While it captures some general developer interest, it has zero personalization and very low catalog coverage (5.3%).
- **Leiden Community-Aware** (0.094 Recall@5, 0.088 NDCG@10) significantly outperforms the popularity baseline by constraining recommendations to the developer's verified technical cluster (e.g. recommending `uvicorn` to a `fastapi` developer rather than a disconnected deep learning library).

### B. Collaborative Filtering & Topological Similarity
- **Collaborative Filtering & Adamic-Adar** achieve competitive accuracy on the connected component because they reward rare, high-affinity co-working edges over generic high-degree hubs.
- **Jaccard Similarity** achieves moderate precision but is penalized when candidate repositories have large total neighbor sets (high degrees).

### C. Node2Vec Representation Learning
- **Node2Vec** (0.225 Recall@5, 100.0% Coverage) provides the broadest catalog exploration, successfully mapping latent topological distance even across sparse multi-hop paths.

### D. Best-Performing Method
- **Highest Accuracy / NDCG**: **Collaborative Filtering** (NDCG@10 = 0.2589, Recall@10 = 0.3790).
- **Highest Catalog Coverage**: **Node2Vec Embeddings** (100.0% of catalog reachable).

---

## 3. Data Leakage Prevention Documentation
1. **Split Isolation**: All 577 test interactions were removed before constructing the training graph.
2. **Community Recomputation**: Leiden communities were **not** imported from the full graph; they were recalculated de novo using only the training graph projection G_R_train.
3. **No Target Leakage**: Models only had access to contributors' training repositories.

---

## 4. Artifacts Generated
- Tabular Example Recommendations: `data/recommendation/recommendation_examples.csv`
- Benchmark Metrics JSON: `data/recommendation/recommendation_metrics.json`
- Precision/Recall Comparison: `data/recommendation/visualizations/recommendation_precision_recall_comparison.png`
- NDCG Ranking Quality: `data/recommendation/visualizations/recommendation_ndcg_ranking_comparison.png`
- Coverage vs. Diversity: `data/recommendation/visualizations/coverage_vs_diversity_tradeoff.png`
