# Limitations

**Project**: Community-Aware Repository Recommendation and Network-Robustness Analysis  
**Scope**: Methodological, Empirical, and Theoretical Limitations

---

## 1. Dataset and Sampling Limitations

1. **Ecosystem and Language Bias**:
   - The collaboration network was constructed from 206 curated popular repositories, heavily skewed toward the Python open-source data science, machine learning, and web backend ecosystems (e.g., PyTorch, Transformers, FastAPI, Polars).
   - Findings may not generalize directly to monorepo-dominated ecosystems, languages with different package managers (e.g., JavaScript/npm, Rust/crates.io), or enterprise repositories with closed contribution models.
2. **Catalog Scale**:
   - While 4,759 contributors and 206 repositories represent a substantial collaboration network, GitHub in its entirety contains millions of repositories. Scaling community detection and Node2Vec embeddings to millions of nodes introduces algorithmic and computational bottlenecks not evaluated here.

---

## 2. Contributor Topology and Degree Distribution

1. **Extreme Degree-1 Imbalance**:
   - 4,197 out of 4,759 contributors (88.19%) contributed to exactly one repository in the catalog.
   - While representative of open-source "drive-by" contributions, this heavy tail limits the pool of evaluable contributors in leave-one-out recommendation benchmarks to 562 developers (11.81%).
2. **Bipartite Density**:
   - Network density is 0.00587. Bipartite sparsity makes item-item co-occurrence matrices susceptible to disconnection when high-degree nodes are removed.

---

## 3. Recommendation Evaluation Protocols

1. **Offline Re-Creation vs. Real-World Discovery**:
   - Evaluation relied on offline leave-one-out historical interaction prediction. While standard in recommender research, this assumes that recommending a repository a developer previously contributed to is "relevant."
   - In practice, developers may benefit more from discovering novel cross-domain repositories (where Node2Vec excels with 100% catalog coverage) rather than re-discovering tightly coupled sister repositories.
2. **Cold-Start Developers**:
   - Single-repository developers cannot be evaluated under leave-one-out testing without entirely removing their training presence. While community and popularity recommenders can serve them, collaborative filtering cannot generate candidate rankings without prior multi-repo co-occurrences.

---

## 4. Synthetic Robustness Simulation Limitations

1. **Static Removal vs. Dynamic Adaptation**:
   - The simulation assumes static instantaneous removal of top-central contributors without compensatory developer behavior.
   - In actual open-source projects, the departure of a lead maintainer often triggers governance handoffs, onboarding of new maintainers, or repository forks.
2. **Simultaneous Attack Strategy**:
   - Targeted removals were executed simultaneously at fixed cohorts (1%, 5%, 10%) rather than iteratively recalculating centralities after each single node removal. While computationally standard for large bipartite graphs, iterative recalculation might alter the exact ranking order of subsequent targets.
3. **Random Removal Repetitions**:
   - Random removal was benchmarked across 3 independent random seeds per fraction (9 runs total). While the standard error remained under $\pm 1.0\%$, running hundreds of Monte Carlo iterations would provide narrower confidence intervals.

---

## 5. Causal Claims vs. Observational Associations

- All reported relationships between network fragmentation and recommendation degradation are **empirical experimental observations** within the simulated pipeline. They demonstrate mathematical and topological vulnerabilities of specific algorithms, but should not be interpreted as causal laws of open-source human developer sociology without longitudinal, real-world deployment data.
