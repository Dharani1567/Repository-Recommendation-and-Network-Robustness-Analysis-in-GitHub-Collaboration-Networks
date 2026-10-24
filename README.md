# Community-Aware Repository Recommendation and Network-Robustness Analysis in GitHub Collaboration Networks

## 📖 Overview

This research project conducts an empirical Social Network Analysis (SNA) on GitHub collaboration ecosystems. The study focuses on:
1. **Bipartite Collaboration Graphs**: Modeling contributor–repository interactions.
2. **Community Detection**: Segmenting developers and repositories using Leiden modularity optimization.
3. **Personalized Recommendation**: Recommending repositories to contributors through community-aware collaborative filtering and Node2Vec graph embeddings.
4. **Influential Contributor Identification**: Ranking key contributors via centrality and PageRank metrics.
5. **Network Robustness & Breakdown Analysis**: Simulating targeted removals of key developers vs. random churn.
6. **Downstream Recommendation Degradation**: Measuring how structural network degradation impacts repository discovery quality.

---

## 🏗️ Project Architecture

```
hh/
├── .env.example                # Safe environment variable template
├── .gitignore                  # Git ignore rules (protects .env and raw data)
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview and setup instructions
├── PROJECT_PLAN.md             # Multi-stage research development roadmap
├── data/                       # Data storage directory
│   ├── raw/                    # Raw JSON responses from GitHub REST API
│   ├── processed/              # Cleaned tabular edges and node attributes
│   ├── cache/                  # HTTP cache for GitHub API requests
│   └── outputs/                # Experiment results, graphs, and metric plots
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── config/                 # Settings and environment loader
│   ├── data_collection/        # GitHub REST API client, rate-limit handling, pagination
│   ├── preprocessing/          # Bot filtering, edge extraction, attribute parsing
│   ├── network_construction/   # Bipartite and unipartite projected graph generation
│   ├── community_detection/    # Leiden and Louvain modularity algorithms
│   ├── recommendation/         # Community-aware CF and Node2Vec embeddings
│   ├── influence/              # Degree, betweenness, PageRank, coreness metrics
│   ├── robustness/             # Targeted and random node attack simulations
│   ├── evaluation/             # Precision@K, Recall@K, MAP, NDCG, link prediction
│   └── dashboard/              # Streamlit interactive exploratory dashboard
└── tests/                      # Automated test suite
```

---

## ⚙️ Technology Stack & Dependencies

- **Language**: Python 3.14+
- **Graph & Network Theory**: `networkx`, `cdlib` / `leidenalg` / `igraph`, `node2vec`, `python-louvain`
- **Data & Numerical Computation**: `pandas`, `numpy`, `scipy`, `scikit-learn`
- **API & Networking**: `requests`, `urllib3`, `tqdm`
- **Visualization & UI**: `streamlit`, `plotly`, `matplotlib`, `seaborn`
- **Configuration & Testing**: `python-dotenv`, `pydantic`, `pytest`

---

## 🚀 Setup & Installation

### 1. Configure Python Environment
Ensure Python 3.10+ is available. Check installation:
```bash
python --version
```

### 2. Configure Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```
Open `.env` and set your personal GitHub access token:
```ini
GITHUB_TOKEN=ghp_yourActualTokenHere
```
> **Security Notice**: Never commit `.env` or hardcode tokens in script files.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Tests
```bash
pytest tests/
```

### 5. Launch Interactive Dashboard
```bash
streamlit run src/dashboard/app.py
```
