# Research Project Plan

**Title**: Community-Aware Repository Recommendation and Network-Robustness Analysis in GitHub Collaboration Networks  
**Domain**: Social Network Analysis (SNA), Recommender Systems, Complex Network Resilience  

---

## 🎯 Research Hypotheses & Objectives

1. **H1 (Community Structure)**: Open-source GitHub collaboration forms statistically significant modular community structures reflecting shared technological domains or organizational affinities.
2. **H2 (Recommendation Superiority)**: Community-aware collaborative filtering and representation learning (Node2Vec) significantly outperform baseline degree/popularity-based recommendations in recommending relevant repositories to contributors.
3. **H3 (Network Robustness Vulnerability)**: Targeted removal of high-influence contributors (e.g., top betweenness or PageRank nodes) accelerates network fragmentation (drop in Giant Connected Component size and global efficiency) much faster than random developer churn.
4. **H4 (Downstream Task Degradation)**: Disruption of influential central bridge contributors causes a pronounced decline in recommendation quality (Precision@K, Recall@K, NDCG@K) for peripheral developers.

---

## 📅 Development Roadmap & Stages

### Part 1: Project Setup & Architecture (Completed ✅)
- Inspect environment and detect installed vs. missing dependencies.
- Establish clean modular source hierarchy:
  - `config`, `data_collection`, `preprocessing`, `network_construction`, `community_detection`, `recommendation`, `influence`, `robustness`, `evaluation`, `dashboard`, `tests`.
- Configure environment protection (`.env.example`, `.gitignore`).
- Prepare project documentation (`README.md`, `requirements.txt`, `PROJECT_PLAN.md`).

---

### Part 2: GitHub Data Collection & Ingestion
- **Target Target Selection**: Curate seed topics or target open-source ecosystems (e.g., Python machine learning, web tooling, or systems software).
- **GitHub REST API Client**:
  - Authenticated queries with exponential backoff on `403 rate limit exceeded`.
  - Fetch repository metadata (stars, forks, topics, language).
  - Fetch contributor profiles, commit frequencies, and contribution counts.
- **Local Caching Layer**: Cache JSON responses to prevent redundant network roundtrips.
- **Bot & CI Filtering**: Filter automated accounts (e.g., `dependabot`, `actions-user`, `greenkeeper`).

---

### Part 3: Graph Construction & Topological Preprocessing
- **Bipartite Graph ($G_{CR}$)**:
  - Disjoint node sets: Contributors ($C$) and Repositories ($R$).
  - Undirected or directed edges weighted by contribution count / log-normalized commits.
- **Unipartite Projections**:
  - Contributor Collaboration Network ($G_C$): Nodes are developers; edge weights represent shared repository co-contributions (Jaccard, Newman, or simple overlap).
  - Repository Proximity Network ($G_R$): Nodes are repositories; edge weights reflect shared contributor pools.
- **Topological Statistics**: Density, diameter, degree distribution (power-law check), clustering coefficient.

---

### Part 4: Community Detection Analysis
- **Leiden Algorithm**: Guarantee well-connected community partitions with higher modularity and no disconnected communities.
- **Louvain Algorithm Baseline**: Benchmark modularity score ($Q$) and execution speed.
- **Analysis**:
  - Cross-community topic distribution (do communities map to specific tech stacks?).
  - Modularity density and inter- vs. intra-cluster edge ratio.

---

### Part 5: Community-Aware Repository Recommendation
- **Baseline Models**:
  - Global Popularity (most starred / highest contributor repositories).
  - Classical Bipartite Collaborative Filtering (User-Item similarity).
- **Proposed Community-Aware Model**:
  - Intra-community preference weighting: Bias recommendations towards repositories within the contributor's detected collaboration community.
- **Graph Representation Learning (Node2Vec)**:
  - Biased random walks on $G_{CR}$ or projected graphs to learn dense node embeddings.
  - Cosine similarity ranking in embedding space.
- **Evaluation Framework**:
  - Temporal or masked train/test edge split (80/20).
  - Metrics: Precision@K, Recall@K, Mean Average Precision (MAP), Normalized Discounted Cumulative Gain (NDCG@K).

---

### Part 6: Contributor Influence & Centrality Profiling
- **Centrality Metrics**:
  - Degree Centrality: Direct activity volume.
  - Betweenness Centrality: Bridge roles across repositories and sub-communities.
  - Closeness Centrality: Proximity to entire network.
  - PageRank & HITS: Prestige and authority in collaboration flow.
  - K-Core Decomposition: Coreness levels separating peripheral vs. core contributors.
- **Correlation Matrix**: Compare how different centrality rankings align.

---

### Part 7: Network Robustness & Attack Simulation
- **Attack Strategies**:
  1. Targeted removal by **Betweenness Centrality** (critical bridges).
  2. Targeted removal by **PageRank** (authoritative hubs).
  3. Targeted removal by **Degree Centrality** (highest connectivity).
  4. Random node removal (Simulated natural churn / baseline).
- **Resilience Metrics**:
  - Relative size of the Giant Connected Component ($S = |GCC| / |V|$).
  - Average Shortest Path Length ($L$).
  - Global Network Efficiency ($E_{glob}$).
  - Number of disconnected components.

---

### Part 8: Downstream Recommendation Degradation
- **Joint Analysis**:
  - Re-evaluate recommendation performance (Precision@K, NDCG@K) after removing varying fractions ($f \in [0.01, 0.20]$) of top influential contributors.
  - Compare recommendation robustness under targeted vs. random removal.
  - Identify whether community-aware recommendations buffer against peripheral disengagement.

---

### Part 9: Streamlit Dashboard & Reporting
- Visual interactive application:
  - Interactive network graph explorer (using Plotly/PyVis).
  - Community modularity and partition inspector.
  - Contributor influence leaderboards.
  - "What-If" interactive robustness simulation slider.
  - Recommendation engine test sandbox.
- Final research paper synthesis & visual figures.
