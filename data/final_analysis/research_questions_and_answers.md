# Research Questions and Evidence-Based Answers

**Project Title**: Community-Aware Repository Recommendation and Network-Robustness Analysis in GitHub Collaboration Networks  
**Scope**: Definitive answers grounded strictly in empirical data from Parts 3 through 8.

---

### Question 1: Which recommendation method performs best?

**Answer**:  
**Collaborative Filtering (CF)** performs best overall in terms of accuracy, achieving a top-ranked **Recall@10 of 37.90%**, **Precision@10 of 0.0390**, and **NDCG@10 of 0.2589**.  
However, when performance is evaluated on **Catalog Coverage** and **Recommendation Diversity**, **Node2Vec Embeddings** is superior, achieving **100.0% catalog coverage** (vs 73.30% for CF and 5.34% for Popularity) and the highest **Community Diversity (0.4984)** while maintaining a strong **Recall@10 of 28.25%**.

*Source Artifact*: `data/recommendation/recommendation_metrics.json`, `data/final_analysis/final_metrics_summary.csv`

---

### Question 2: How does community information contribute to repository recommendation?

**Answer**:  
Community structure (derived via the Leiden algorithm on the repository projection) provides a reliable topical grouping of repositories ($Q = 0.4643$), but direct community-aware recommendation achieves lower overall accuracy (**Recall@10 = 19.93%**, **NDCG@10 = 0.0885**) than co-occurrence methods.  
The contribution of community information is twofold:
1. **Cold-start & Exploration Baseline**: It substantially outperforms the standard Popularity baseline (Recall@10 = 3.02%) by steering developers to domain-relevant repositories.
2. **Resilience under Attack**: Community assignments remain valid during network disruptions. At 5% Degree attack, when CF completely collapses (Recall@10 = 0.00%), Leiden Community-Aware recommendation still yields valid recommendations (Recall@10 = 3.09%).

*Source Artifact*: `data/community_detection/community_detection_metrics.json`, `data/research_validation/baseline_comparison.csv`

---

### Question 3: Which contributor influence measures identify structurally important contributors?

**Answer**:  
**Betweenness Centrality** and **K-Core Centrality** identify the most structurally indispensable contributors for global network cohesion, whereas **Degree Centrality** identifies contributors critical for recommendation co-occurrence.  
Crucially, Part 8 proved that among multi-repository developers (N=562), Betweenness has only a moderate correlation with Degree ($\rho = 0.5245$) and a weak correlation with K-Core ($\rho = 0.3454$). Betweenness uniquely isolates the **203 cross-community bridging contributors (4.27%)** whose presence prevents the network from shattering into disjoint topical silos.

*Source Artifact*: `data/influence_analysis/influence_metrics.json`, `data/research_validation/validation_summary.json`

---

### Question 4: Which targeted removal strategy causes the greatest structural damage?

**Answer**:  
**Targeted Betweenness Removal** causes the greatest structural damage to the global network.  
At 10% contributor removal (476 nodes), Betweenness attack reduces the Giant Connected Component to **1.70% of the network (78 nodes)** and fragments the network into **171 connected components** (up from 14 at baseline). K-Core removal is similarly devastating (GCC collapses to 1.47%, 164 components). In contrast, Degree removal leaves an 18.94% GCC (868 nodes).

*Source Artifact*: `data/robustness_analysis/robustness_metrics.csv`

---

### Question 5: Does targeted removal outperform random removal in network damage?

**Answer**:  
**Yes, targeted removal is vastly more damaging than random failure.**  
At 10% removal, random node loss barely impacts network topology: the GCC retains **91.30% of nodes (4,098 nodes)** with only 15.7 components (vs 14 baseline). In stark contrast, targeted betweenness removal collapses the GCC to **1.70% (78 nodes)** with 171 components. This confirms that the collaboration network displays classical scale-free robustness: highly robust to random failures, but catastrophically fragile under targeted topological attack.

*Source Artifact*: `data/robustness_analysis/robustness_metrics.csv`, `data/robustness_analysis/random_removal_results.csv`

---

### Question 6: How does contributor removal affect recommendation quality?

**Answer**:  
Contributor removal severely degrades recommendation quality, but the rate of degradation depends directly on whether the attack disrupts the **item-item co-occurrence matrix**.  
Under Degree Centrality attack, CF Recall@10 drops from **37.90% to 29.57% at 1% removal**, and completely collapses to **0.00% at 5% and 10% removal**. Even when evaluated against **Matched Baselines** (surviving users on unattacked graph), the true degradation of CF at 5% degree removal is **-31.69 percentage points** (falling from a matched baseline of 31.69% to 0.00%). Under random removal, CF Recall@10 remains stable at **37.13% at 5% removal** (true degradation of only -1.00%).

*Source Artifact*: `data/research_validation/baseline_comparison.csv`, `data/research_validation/metric_audit.csv`

---

### Question 7: Does community-aware recommendation behave differently from Collaborative Filtering under network disruption?

**Answer**:  
**Yes, community-aware recommendation exhibits superior structural fault tolerance.**  
While Collaborative Filtering suffers a total wipeout (0 valid recommendations, Recall@10 = 0.00%) at 5% degree removal, Leiden Community-Aware recommendation continues functioning, producing valid recommendations for surviving users (Recall@10 = 3.09% at 5% degree removal, and 4.65% at 10% betweenness removal). Because community partitions capture broader modular boundaries rather than individual co-occurrence edges, they degrade gracefully under targeted attacks.

*Source Artifact*: `data/robustness_analysis/recommendation_robustness.csv`, `data/research_validation/baseline_comparison.csv`

---

### Question 8: What is the relationship between network fragmentation and recommendation degradation?

**Answer**:  
The relationship is **asymmetric and model-dependent**:
1. **Co-occurrence models (CF, Jaccard, Adamic-Adar)** degrade via **bipartite path severance**, not global GCC collapse. Even before the global network fragments into hundreds of pieces (which occurs at 10% betweenness removal), removing just 5% of degree-central contributors eliminates all multi-repository connections in the training graph, completely collapsing co-occurrence.
2. **Global structural fragmentation (shattering the GCC)** heavily impacts topological and path-based routing, while community-aware models retain utility as long as individual repository communities maintain internal edges.

*Source Artifact*: `data/research_validation/validation_report.md`, `data/robustness_analysis/recommendation_robustness.csv`

---

### Question 9: What limitations prevent generalizing the findings?

**Answer**:  
Generalization is constrained by:
1. **Dataset Scope**: Sampled across 206 curated popular Python/ML/Web repositories (4,759 contributors), which may not represent low-activity or non-Python open-source ecosystems.
2. **Extreme Degree Imbalance**: 88.19% of contributors have degree 1, meaning multi-repository collaboration is concentrated in a small minority.
3. **Synthetic Removal Protocol**: Contributor departure was simulated offline without modeling dynamic compensation (e.g., remaining contributors taking over unmaintained repositories).
4. **Offline Evaluation Metric**: Leave-one-out interaction holdout measures historical link recreation rather than user satisfaction in a live production environment.

*Source Artifact*: `data/final_analysis/limitations.md`
