# Recommendation Analysis

**Project**: Community-Aware Repository Recommendation and Network-Robustness Analysis  
**Artifact Source**: `data/recommendation/recommendation_metrics.json`, `data/research_validation/metric_audit.csv`

---

## 1. Validated Recommendation Benchmark

The recommendation benchmark was evaluated under strict zero-leakage conditions on a partitioned training graph. A total of **562 multi-repository contributors** (each having $\ge 2$ observed contribution edges) were evaluated on **577 held-out test interactions**. Single-repository contributors (4,197 nodes) were retained strictly within the training graph to preserve repository degree distributions and bipartite topology without disconnected isolate artifacts.

### Full Comparative Results Table

| Recommendation Model | P@5 | R@5 | F1@5 | NDCG@5 | P@10 | R@10 | F1@10 | NDCG@10 | Catalog Coverage | Community Diversity | Valid Recs / 562 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Collaborative Filtering** | **0.0669** | **0.3287** | **0.1109** | **0.2418** | **0.0390** | **0.3790** | **0.0704** | **0.2589** | 73.30% | 0.3859 | 503 |
| **Jaccard Similarity** | 0.0498 | 0.2429 | 0.0823 | 0.1681 | 0.0359 | 0.3510 | 0.0650 | 0.2024 | 72.33% | 0.2704 | 503 |
| **Adamic-Adar** | 0.0473 | 0.2282 | 0.0779 | 0.1625 | 0.0311 | 0.3020 | 0.0562 | 0.1864 | 65.05% | 0.2935 | 503 |
| **Node2Vec Embeddings** | 0.0466 | 0.2251 | 0.0768 | 0.1601 | 0.0292 | 0.2825 | 0.0527 | 0.1791 | **100.00%** | **0.4984** | **562** |
| **Leiden Community-Aware** | 0.0192 | 0.0939 | 0.0317 | 0.0545 | 0.0206 | 0.1993 | 0.0373 | 0.0885 | 51.94% | 0.1283 | **562** |
| **Popularity Baseline** | 0.0028 | 0.0142 | 0.0047 | 0.0061 | 0.0030 | 0.0302 | 0.0055 | 0.0110 | 5.34% | 0.3000 | **562** |

---

## 2. Evaluation Protocol & Rigor

1. **Evaluated Contributors**: 562 contributors with degree $\ge 2$. (Single-repo developers with degree 1 cannot be evaluated under leave-one-out testing as leaving out their only edge leaves 0 training edges).
2. **Held-out Interactions**: Exactly 577 interactions (1 edge held out for contributors with small degree, 20% for high-degree contributors).
3. **Candidate Selection Policy**: For each contributor $u$, the candidate pool consists of all 206 repositories excluding repositories already present in $u$'s training history ($R_{\text{cand}} = R \setminus N_{\text{train}}(u)$).
4. **Averaging Method**: Macro-averaging across all 562 test contributors. When an algorithm produces no candidates for a user (valid_recommendations < 562), precision, recall, and NDCG are scored as 0.0 for that contributor.
5. **Leakage Controls**:
   - The test interactions were masked before building training graphs.
   - The repository projection $G_{R,\text{train}}$ was built exclusively from training edges.
   - Leiden community detection was executed exclusively on $G_{R,\text{train}}$.
   - Node2Vec random walks were generated exclusively on the training graph.

---

## 3. In-Depth Comparative Findings

### 3.1 Collaborative Filtering Superiority in Accuracy
Collaborative Filtering (CF) achieves the highest accuracy across every cutoff ($R@5 = 32.87\%$, $R@10 = 37.90\%$, $\text{NDCG}@10 = 0.2589$). By computing cosine similarity over contributor co-occurrence vectors ($C_{ij} = \sum_u \mathbb{I}(u \in N(i) \land u \in N(j))$), CF captures direct developer cross-commit habits. It outperforms topological projections (Jaccard: 35.10%, Adamic-Adar: 30.20%) because it preserves contributor-specific weighting without projecting away bipartite structure prematurely.

### 3.2 The Trade-Off Between Accuracy, Coverage, and Diversity
- **Node2Vec Embeddings**: While Node2Vec ranks 4th in Recall@10 (28.25%), it is the only algorithm that achieves **100.0% Catalog Coverage** and the highest **Community Diversity (0.4984)**. Node2Vec maps all 206 repositories into a dense continuous 64-dimensional space via random walks, allowing it to recommend long-tail repositories that have low raw co-occurrence.
- **Popularity Baseline**: Severely suffers from popularity bias ($R@10 = 3.02\%$), covering only 5.34% of the repository catalog (recommending only the top 11 most popular repositories repeatedly).
- **Leiden Community-Aware**: Ranks 5th in Recall@10 (19.93%) and has the lowest community diversity (0.1283). Because it prioritizes repositories within the contributor's primary detected community, it creates an echo-chamber effect, trading catalog exploration for within-cluster precision.

---

## 4. Key Takeaways for System Design
1. For pure offline prediction accuracy, **Collaborative Filtering** is clearly optimal.
2. For discovering novel repositories across different ecosystems, **Node2Vec** is far superior due to 100% catalog coverage.
3. **Community-Aware** methods serve best as re-ranking or diversity filters rather than stand-alone candidate generators.
