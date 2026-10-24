# Final Results Summary

**Project Title**: Community-Aware Repository Recommendation and Network-Robustness Analysis in GitHub Collaboration Networks  
**Dataset**: 4,759 Contributors, 206 Repositories, 5,758 Observed Contribution Edges  
**Synthesized**: 2026-10-25  
**Scope**: Synthesis of Parts 3 through 8 (Validated Results)

---

## 1. Executive Summary

This research investigates the interplay between community structure, contributor influence, recommendation accuracy, and network robustness in open-source software collaboration networks. Integrating empirical evidence across bipartite graph modeling, community detection, collaborative and topological recommendation models, multi-centrality influence rankings, and targeted removal simulations yields five fundamental findings:

1. **Collaborative Filtering is the Strongest Recommender but Highly Fragile**: Contributor-based Collaborative Filtering achieves the highest recommendation accuracy (**Recall@10 = 37.90%**, **NDCG@10 = 0.2589**), substantially outperforming Community-Aware baselines (**Recall@10 = 19.93%**, **NDCG@10 = 0.0885**) and Popularity baselines (**Recall@10 = 3.02%**, **NDCG@10 = 0.0110**). However, CF suffers total co-occurrence collapse under targeted removal of just 5% of degree-central contributors.
2. **Node2Vec Embeddings Maximize Coverage and Diversity**: Node2Vec Skip-Gram embeddings provide the highest catalog coverage (**100.0%** vs 56.8% for CF) and highest intra-list diversity (**0.4984** vs 0.3859 for CF), while maintaining competitive accuracy (**Recall@10 = 28.25%**, **NDCG@10 = 0.1791**).
3. **Leiden Community Detection Provides Stable Topical Modularity**: Leiden partitioning of the repository projection identifies 10 well-defined repository clusters ($Q = 0.4643$, 5-seed ARI = 0.7826), outperforming Louvain in partition stability and connectivity guarantees.
4. **Influence Divergence Confined to Bridging Developers**: While 88.19% of contributors are degree-1 leaf nodes causing near-perfect global rank correlation ($\rho \approx 0.998$), the 562 multi-repository contributors display marked divergence: **Betweenness Centrality correlates at only $\rho = 0.5245$ with Degree and $\rho = 0.3454$ with K-Core**, capturing critical cross-community bridges.
5. **Targeted Attacks Severely Degrade Structure and Recommendations**: Targeted contributor removals cause catastrophic breakdown compared to random node loss. Under 10% Betweenness attack, the Giant Connected Component collapses by **98.3%** (from 4,584 nodes to 78 nodes), while 10% random removal retains **91.3%** of the GCC. Under 5% Degree attack, Collaborative Filtering recommendation drops to **0.00%** because all multi-repository links in the training graph are severed.

---

## 2. Integrated Experiment Summary Table

| Research Stage | Validated Output Artifact | Key Metric / Result | Scientific Interpretation |
| :--- | :--- | :--- | :--- |
| **Part 3: Network Construction** | `collaboration_network.graphml` | 4,759 contributors, 206 repos, 5,758 edges; GCC = 4,584 nodes (92.33%) | Scale-free bipartite structure with 88.19% degree-1 leaves. |
| **Part 4: Community Detection** | `community_detection_metrics.json` | Leiden: 10 communities, $Q = 0.4643$, ARI = 0.7826 | High modularity topical clustering without disconnected artifacts. |
| **Part 5: Recommendation Benchmark**| `recommendation_metrics.json` | CF Recall@10 = 37.90%, Node2Vec = 28.25%, Leiden = 19.93% | Co-occurrence > Topological Similarity > Community Baselines. |
| **Part 6: Influence Analysis** | `influence_metrics.json` | 203 cross-community contributors (4.27%); Multi-repo $\rho(\text{deg}, \text{betw}) = 0.5245$ | Bridging role is concentrated in $<5\%$ of developer population. |
| **Part 7: Network Robustness** | `robustness_metrics.csv` | 10% Betweenness attack collapses GCC to 1.70%; 10% Random retains 91.30% | Extreme vulnerability to targeted bridging attacks. |
| **Part 8: Consistency Audit** | `baseline_comparison.csv` | CF matched baseline drop: -31.69% at 5% degree removal (post-attack = 0.00%) | Mathematically verified co-occurrence collapse and cohort shift control. |

---

## 3. Structural Breakdown vs. Recommendation Breakdown

The research connects structural network integrity with algorithmic utility:
- **Structural Attack Vulnerability**: Betweenness Centrality causes the most severe topological disruption, shattering the network into 171 disconnected components at 10% removal.
- **Algorithmic Attack Vulnerability**: Degree Centrality causes the fastest algorithmic collapse for Collaborative Filtering, completely wiping out item-item co-occurrence by 5% removal.
- **Community Resilience**: While Leiden Community-Aware recommendation has lower initial accuracy (19.93%), it does not suffer complete zero-recommendation collapse under 5% Degree attack (retaining 3.09% Recall@10) because community boundaries remain partially intact even when individual co-occurrences vanish.

---

## 4. Synthesis Index

Detailed thematic analyses are provided in the companion synthesis documents:
- [Recommendation Analysis](file:///c:/Users/darsh/OneDrive/Desktop/hh/data/final_analysis/recommendation_analysis.md)
- [Community Analysis](file:///c:/Users/darsh/OneDrive/Desktop/hh/data/final_analysis/community_analysis.md)
- [Contributor Influence Analysis](file:///c:/Users/darsh/OneDrive/Desktop/hh/data/final_analysis/contributor_influence_analysis.md)
- [Robustness Analysis](file:///c:/Users/darsh/OneDrive/Desktop/hh/data/final_analysis/robustness_analysis.md)
- [Research Questions and Answers](file:///c:/Users/darsh/OneDrive/Desktop/hh/data/final_analysis/research_questions_and_answers.md)
- [Limitations](file:///c:/Users/darsh/OneDrive/Desktop/hh/data/final_analysis/limitations.md)
- [Research Contribution](file:///c:/Users/darsh/OneDrive/Desktop/hh/data/final_analysis/research_contribution.md)
