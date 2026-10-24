# Dashboard Validation Report

**Project**: Community-Aware Repository Recommendation and Network-Robustness Analysis in GitHub Collaboration Networks  
**Component**: Part 10 Streamlit Interactive Research Dashboard (`src/dashboard/app.py`)  
**Validation Date**: 2026-10-25  
**Test Suite**: `tests/test_dashboard.py` (5 tests), Full project suite (38 tests)

---

## 1. Executive Summary

The Streamlit research dashboard was built to provide an interactive, visual interface for exploring all validated empirical results from Parts 3 through 9. It incorporates:
- **Reconciled Matched Baselines**: Direct incorporation of the audited metrics from `data/research_validation/baseline_comparison.csv` and `data/research_validation/metric_audit.csv`.
- **Transparent Offline Protocol Disclaimers**: Explicit labeling that recommendation accuracy reflects offline leave-one-out historical interaction prediction rather than prospective user adoption.
- **Topological Integrity Disclaimers**: Contextual annotations explaining that decreasing Average Path Length (APL) under severe GCC fragmentation is an artifact of local condensation, not improved network efficiency.
- **Six Functional Research Views**: Multi-page sidebar navigation covering Executive Overview, Community Detection, Recommendation Benchmark, Contributor Influence, Network Robustness, and Recommendation Robustness.

---

## 2. Reconciled Matched-Baseline Audit

Before constructing the dashboard, the discrepancy between the narrative summary and `baseline_comparison.csv` was reconciled. 

### Correct Values Preserved in Dashboard:
- **1% Degree Centrality Attack (514 survivors)**:
  - Original 562-User Baseline: **37.90%**
  - Matched Baseline (514 survivors on unattacked graph): **36.31%**
  - Post-Attack Recall@10: **29.57%**
  - Cohort Bias ($\text{Matched} - \text{Original}$): **-0.0159** ($-1.59\%$)
  - True Algorithmic Degradation ($\text{Post} - \text{Matched}$): **-0.0674** ($-6.74\%$)
- **5% Degree Centrality Attack (324 survivors)**:
  - Original 562-User Baseline: **37.90%**
  - Matched Baseline (324 survivors on unattacked graph): **31.69%**
  - Post-Attack Recall@10: **0.00%** (Total Co-occurrence Collapse)
  - True Algorithmic Degradation: **-0.3169** ($-31.69\%$)
- **10% Degree Centrality Attack (86 survivors)**:
  - Matched Baseline: **50.57%**
  - Post-Attack Recall@10: **0.00%**
  - True Algorithmic Degradation: **-0.5057** ($-50.57\%$)

*Resolution*: The dashboard displays both the global baseline and the matched baseline side-by-side, allowing researchers to inspect the exact cohort bias and true degradation.

---

## 3. Automated Validation Results

The dashboard integration test suite (`tests/test_dashboard.py`) was executed as part of the project test runner:

```bash
python -m unittest discover tests
......................................
----------------------------------------------------------------------
Ran 38 tests in 1.182s

OK
```

### Checks Passed:
1. `test_all_dashboard_data_files_exist`: Confirms all 7 required JSON and CSV data artifacts load cleanly without runtime `FileNotFoundError`.
2. `test_reconciled_matched_baseline_values`: Confirms mathematical accuracy of matched baseline comparisons for Collaborative Filtering under 1% and 5% degree attacks.
3. `test_community_detection_metrics_structure`: Confirms Leiden (10 communities, $Q = 0.4643$) and Louvain (9 communities, $Q = 0.4670$) data schemas.
4. `test_recommendation_benchmark_structure`: Confirms all 6 recommenders are present with Precision, Recall, F1, NDCG, Catalog Coverage, and Community Diversity.
5. `test_robustness_metrics_trajectory`: Confirms baseline GCC fraction (92.33%) and 10% Betweenness attack collapse (185 nodes, 4.12% GCC fraction).

---

## 4. Dashboard Limitations

1. **Static Precomputed Outputs**: The dashboard visualizes completed experiment runs stored in `data/`. It does not support live synthetic node deletion recomputations on the fly due to the computational cost of recalculating shortest paths on a 4,965-node graph.
2. **Local Execution**: The dashboard is designed to run locally via `streamlit run src/dashboard/app.py` with standard local browser access (`http://localhost:8501`).
3. **No External Network Dependencies**: Operates entirely offline without querying GitHub API tokens or external servers.
