# Community Detection Analysis

**Project**: Community-Aware Repository Recommendation and Network-Robustness Analysis  
**Artifact Source**: `data/community_detection/community_detection_metrics.json`, `data/community_detection/repository_communities.csv`

---

## 1. Algorithmic Comparison: Leiden vs. Louvain

The repository collaboration network was constructed by projecting the bipartite graph onto repository nodes, where edge weights represent the number of shared unique contributors. The unattacked repository projection contains **206 repositories** and **1,353 weighted collaboration edges**, including **12 isolated repositories** that share zero contributors with any other repository in the catalog.

### Comparative Clustering Metrics Table

| Metric | Louvain Algorithm | Leiden Algorithm | Scientific Difference |
| :--- | :---: | :---: | :--- |
| **Number of Communities** | 9 | **10** | Leiden detects finer modular structure. |
| **Modularity ($Q$)** | **0.4670** | 0.4643 | Louvain is marginally higher ($\Delta Q = 0.0027$). |
| **Stability across 5 Seeds (ARI)** | 0.7677 | **0.7826** | Leiden is significantly more stable (+1.94%). |
| **Stability across 5 Seeds (NMI)** | 0.8448 | **0.8616** | Leiden preserves mutual information across runs. |
| **Cross-Algorithm Agreement (ARI)** | 0.6531 | 0.6531 | Moderate-high consensus between algorithms. |
| **Cross-Algorithm Agreement (NMI)** | 0.7985 | 0.7985 | Strong mutual information between partitions. |
| **Isolated Repositories Handled** | 12 | 12 | Retained as distinct isolated components. |
| **Disconnected Communities Found**| Potential risk | **Guaranteed 0** | Leiden guarantees well-connected communities. |

---

## 2. Community Size Distributions

- **Louvain (9 Communities)**: `[49, 45, 26, 23, 23, 14, 10, 2, 2]`
- **Leiden (10 Communities)**: `[52, 37, 30, 24, 21, 14, 8, 4, 2, 2]`

Both algorithms identify two large dominant super-communities comprising approximately 45–50 repositories each, three medium-sized communities (20–35 repositories), and several specialized sub-clusters (2–8 repositories).

---

## 3. Topical and Semantic Interpretation of Leiden Communities

Qualitative inspection of `data/community_detection/repository_communities.csv` demonstrates clear thematic separation across software domains:
1. **Community 0 (Core Machine Learning & Deep Learning, 52 repos)**: Contains PyTorch ecosystem libraries, Hugging Face transformers, acceleration runtimes (ONNX, TensorRT), and computer vision models.
2. **Community 1 (Web Frameworks & API Tooling, 37 repos)**: Contains FastAPI, Starlette, Pydantic, HTTP clients, and ASGI servers.
3. **Community 2 (Data Science, ETL & Tabular Analytics, 30 repos)**: Encompasses Polars, DuckDB, Pandas tooling, Arrow, and data ingestion pipelines.
4. **Community 3 (LLM Agents & Vector Search, 24 repos)**: Groups LangChain, LlamaIndex, Chroma, and Qdrant integration packages.
5. **Community 4 (DevOps, Infrastructure & CLI, 21 repos)**: Encompasses Docker tools, Kubernetes operators, Rich, Typer, and Click utilities.

---

## 4. How Community Structure Informs Recommendation

In Part 5, the **Leiden Community-Aware Recommender** assigned each contributor to their dominant community based on training commit history:
$$\text{dominant\_comm}(u) = \arg\max_c |N_{\text{train}}(u) \cap C_c|$$
Candidates within that community were ranked by within-community in-degree.

### Scientific Evaluation of Community Utility
- **Does high modularity ($Q = 0.4643$) guarantee superior recommendation accuracy?**  
  **No.** While Leiden communities capture genuine architectural affinities, restricting candidates to the detected community limits Recall@10 to **19.93%**, far below Collaborative Filtering (**37.90%**).
- **Why does community-aware recommendation underperform CF?**
  1. **Coarse Granularity**: A 52-repository ML cluster contains many repositories that a developer does not need or want.
  2. **Inter-Community Bridging**: Modern developers frequently bridge domains (e.g., integrating a Web API framework with a Vector Database). Pure community recommendation penalizes cross-domain exploration.
- **The True Value of Communities in Recommendation**:  
  Communities provide an invaluable **diversity filter** and **cold-start fallback**. When a user has few historical commits, knowing their primary community enables a much better recommendation (19.93%) than global popularity (3.02%). Furthermore, community partitions remain valid during network disruptions when fine-grained co-occurrences disappear.
