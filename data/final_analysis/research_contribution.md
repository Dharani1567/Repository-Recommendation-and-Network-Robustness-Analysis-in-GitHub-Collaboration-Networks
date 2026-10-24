# Research Contribution

**Project**: Community-Aware Repository Recommendation and Network-Robustness Analysis in GitHub Collaboration Networks  
**Scope**: Experimental and Empirical Contribution (Not a Claim of Proven Global Literature Gap)

---

## 1. Experimental Research Contributions

This study provides a unified empirical framework connecting **contributor influence analysis**, **community detection**, **repository recommendation algorithms**, and **targeted network robustness simulations** on an empirical bipartite GitHub collaboration network (4,759 contributors, 206 repositories, 5,758 edges).

The core experimental contributions are:

### Contribution 1: Identification of the Structural Co-Occurrence Collapse Threshold
We empirically demonstrate and mathematically verify the **co-occurrence collapse phenomenon** in bipartite recommendation systems. While Collaborative Filtering is the most accurate offline model (Recall@10 = 37.90%), its functionality depends entirely on an extremely fragile structural core of multi-repository contributors ($< 5\%$ of the developer population). Removing the top 5% (238) degree-central contributors completely eliminates all multi-repository developers in the training graph, causing the item-item co-occurrence matrix to vanish identically ($C \equiv 0$) and producing a total failure mode (Recall@10 = 0.00%).

### Contribution 2: Multi-Centrality Disentanglement in Scale-Free Collaboration Graphs
We resolve an empirical pitfall common to collaboration network research: the artificial near-perfect correlation ($\rho \approx 0.998$) observed between Degree, Betweenness, and K-Core centralities. We show that this is an artifact of the scale-free power law, where 88.19% of contributors are degree-1 leaf nodes with identical tied minimum ranks. By filtering out leaf nodes, we demonstrate that **Betweenness Centrality diverges markedly from Degree ($\rho = 0.5245$) and K-Core ($\rho = 0.3454$)**, isolating a distinct cadre of 203 cross-community bridging contributors (4.27%) who are invisible to commit-volume metrics.

### Contribution 3: The Asymmetry Between Structural Fragmentation and Recommendation Breakdown
We establish an empirical distinction between **topological network damage** and **recommendation utility degradation**:
- **Targeted Betweenness Attack** is the most destructive structural strategy, shattering the Giant Connected Component by **98.3%** (reducing GCC to 1.70% of nodes and generating 171 disconnected components).
- **Targeted Degree Attack** is the most destructive algorithmic strategy against Collaborative Filtering, inducing total co-occurrence collapse at 5% removal before the GCC has even collapsed (GCC retains 82.02% at 5% degree removal).
- **Random Node Loss** exerts negligible structural or algorithmic impact even at 10% removal (GCC retains 91.30%, CF Recall@10 retains 34.69%), confirming scale-free resilience to random failures coupled with extreme vulnerability to targeted developer churn.

### Contribution 4: Community-Aware Fault Tolerance
We show that while Leiden Community-Aware recommendation achieves lower peak accuracy than CF (Recall@10 = 19.93% vs 37.90%), it exhibits **graceful degradation under attack**. When CF collapses completely (0 valid recommendations), Leiden continues to recommend valid within-community repositories (Recall@10 = 3.09% under 5% degree attack and 4.65% under 10% betweenness attack) because community partitions remain topologically informative even when specific pairwise co-occurrences are severed.

### Contribution 5: Matched-Baseline Evaluation Methodology for Robustness Simulation
We introduce and validate a **Matched Baseline Evaluation Protocol** for network perturbation experiments. When nodes are removed from a network, the surviving test cohort naturally changes. Evaluating degraded models against an unattacked baseline computed over a different population introduces cohort bias. By evaluating the unattacked model on the exact surviving user cohort, researchers can isolate the true degradation caused by structural loss from changes in user propensity.
