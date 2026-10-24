# Network Robustness Analysis

**Project**: Community-Aware Repository Recommendation and Network-Robustness Analysis  
**Artifact Source**: `data/robustness_analysis/robustness_metrics.csv`, `data/robustness_analysis/random_removal_results.csv`, `data/research_validation/validation_summary.json`

---

## 1. Experimental Protocol for Contributor Removal

Network robustness was evaluated by simulating synthetic contributor removal on the full bipartite network (4,759 contributors, 206 repositories, 4,965 total nodes, 5,758 edges) across six distinct removal strategies at three fractional cohorts: **1% (47 nodes)**, **5% (238 nodes)**, and **10% (476 nodes)**.

Strategies tested:
1. **Targeted Degree Removal** (removing contributors with the highest repository degree)
2. **Targeted Betweenness Removal** (removing contributors with the highest shortest-path transit)
3. **Targeted PageRank Removal** (removing contributors with the highest random-walk authority)
4. **Targeted K-Core Removal** (removing contributors in the deepest core shells)
5. **Targeted Weighted Degree Removal** (removing contributors with the highest total commits)
6. **Random Removal Baseline** (repeated across multiple independent Monte Carlo seeds)

---

## 2. Structural Degradation Across Attack Strategies

### Comprehensive Structural Breakdown Table

| Strategy | Fraction Removed | Remaining Nodes | Edges Remaining | GCC Nodes | GCC Fraction | Connected Components | Avg Path Length (GCC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (No Removal)** | 0.00 | 4,965 | 5,758 | 4,584 | **92.33%** | 14 | 7.7485 |
| **Random Removal (Mean)** | 0.01 | 4,918 | 5,700 | 4,540 | 92.34% | 14.0 | 7.7487 |
| **Random Removal (Mean)** | 0.05 | 4,727 | 5,470 | 4,346 | 91.93% | 14.7 | 7.8261 |
| **Random Removal (Mean)** | 0.10 | 4,489 | 5,182 | 4,098 | **91.30%** | 15.7 | 7.8953 |
| **Degree Centrality** | 0.01 | 4,918 | 5,349 | 4,504 | 91.68% | 16 | 8.9618 |
| **Degree Centrality** | 0.05 | 4,727 | 4,372 | 3,887 | 82.02% | 29 | 15.0354 |
| **Degree Centrality** | 0.10 | 4,489 | 3,790 | 868 | **18.94%** | 132 | 13.7762 |
| **Betweenness Centrality** | 0.01 | 4,918 | 5,502 | 4,118 | 83.63% | 25 | 9.3101 |
| **Betweenness Centrality** | 0.05 | 4,727 | 4,874 | 2,504 | 52.87% | 71 | 10.8567 |
| **Betweenness Centrality** | 0.10 | 4,489 | 4,379 | 78 | **1.70%** | **171** | 7.8840 |
| **K-Core Centrality** | 0.01 | 4,918 | 5,364 | 4,525 | 92.05% | 15 | 8.9551 |
| **K-Core Centrality** | 0.05 | 4,727 | 4,455 | 2,817 | 59.40% | 53 | 11.2341 |
| **K-Core Centrality** | 0.10 | 4,489 | 3,923 | 68 | **1.47%** | 164 | 7.7007 |
| **PageRank** | 0.01 | 4,918 | 5,541 | 4,529 | 92.15% | 16 | 7.8319 |
| **PageRank** | 0.05 | 4,727 | 5,020 | 4,196 | 88.51% | 20 | 7.9968 |
| **PageRank** | 0.10 | 4,489 | 4,557 | 3,858 | 85.52% | 25 | 8.6749 |
| **Weighted Degree** | 0.01 | 4,918 | 5,617 | 4,547 | 92.30% | 15 | 7.8733 |
| **Weighted Degree** | 0.05 | 4,727 | 5,263 | 4,185 | 88.28% | 20 | 8.0360 |
| **Weighted Degree** | 0.10 | 4,489 | 4,960 | 3,991 | 88.47% | 22 | 8.8329 |

---

## 3. Targeted Attacks vs. Random Failure

The empirical comparison unequivocally demonstrates the scale-free vulnerability of open-source collaboration networks:
- **Random Removal Resilience**: Removing 10% (476) randomly chosen contributors barely perturbs network topology. The Giant Connected Component retains **91.30%** of all remaining nodes (4,098 nodes), and connected components only rise slightly from 14 to 15.7.
- **Targeted Attack Destructiveness**: Removing 10% of high-betweenness or high-k-core contributors leads to **total topological shattering**:
  - Betweenness attack reduces the GCC to **1.70% (78 nodes)** and fragments the network into **171 disconnected components**.
  - K-Core attack reduces the GCC to **1.47% (68 nodes)** with **164 components**.
  - Degree attack reduces the GCC to **18.94% (868 nodes)** with **132 components**.

---

## 4. Resolution of the Average Path Length (APL) Paradox

A subtle and important phenomenon occurs in the APL trajectory:
- In unattacked baseline: $\text{APL} = 7.7485$.
- At 1% and 5% Degree attack: APL lengthens from $8.96 \rightarrow 15.04$ because bridging nodes are severed, forcing paths through circuitous detours.
- At 10% Betweenness and K-Core attacks: APL suddenly drops back to $7.88$ and $7.70$.

### Why Does APL Decrease Under Severe Attack?
It is a grave scientific error to interpret this decrease as "improved network efficiency".
Average Shortest Path Length is calculated strictly within the surviving Giant Connected Component. When the GCC collapses from 4,584 nodes down to 78 nodes (a 98.3% collapse), the GCC is no longer a global network; it is a tiny, dense, local clique of tightly coupled repositories. In a 78-node local fragment, paths are naturally short. Thus, the APL reduction represents **severe topological condensation and fragmentation**, not network health.
