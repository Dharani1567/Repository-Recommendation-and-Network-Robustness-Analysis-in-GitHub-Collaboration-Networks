# Community Detection Evaluation: Leiden vs. Louvain

- **Evaluation Date**: 2026-10-25
- **Input Graph**: `data/network/collaboration_network.graphml`
- **Total Repositories**: 206
- **Repository Co-Contribution Edges**: 1,353
- **Connected Repositories Evaluated**: 194
- **Isolated Repositories (Degree 0 in Projection)**: 12 (Assigned Community ID = -1)

---

## 1. Quantitative Algorithmic Comparison

| Evaluation Metric | Louvain Algorithm | Leiden Algorithm | Winner / Advantage |
| :--- | :--- | :--- | :--- |
| **Modularity Score ($Q$)** | **0.4670** | **0.4643** | **Tie / Competitive** (both $> 0.46$) |
| **Number of Communities** | **9** | **10** | Consistent granularity |
| **Stability across Seeds (ARI)** | **0.7677** | **0.7826** | **Leiden** (superior determinism & stability) |
| **Stability across Seeds (NMI)** | **0.8448** | **0.8616** | **Leiden** |
| **Cross-Algorithm Agreement (ARI)**| 0.6531 | 0.6531 | High structural convergence |
| **Cross-Algorithm Agreement (NMI)**| 0.7985 | 0.7985 | Strong mutual agreement |
| **Disconnected Communities** | Potential risk | **Guaranteed 0 disconnected** | **Leiden** |
| **Singletons (size = 1)** | 0 | 0 | No isolated singletons inside GCC |
| **Isolated Repositories ($k=0$)** | 12 | 12 | Handled cleanly in cluster `-1` |

---

## 2. Community Size Distributions

- **Louvain Community Sizes**: `[49, 45, 26, 23, 23, 14, 10, 2, 2]`
- **Leiden Community Sizes**: `[52, 37, 30, 24, 21, 14, 8, 4, 2, 2]`

---

## 3. Discovered Qualitative Communities (Leiden Partition)

### Community 1 (52 repos)
- `scikit-learn/scikit-learn`
- `pytorch/pytorch`
- `vllm-project/vllm`
- `pytorch/vision`
- `pytorch/audio`
- `dmlc/xgboost`
- `lightgbm-org/LightGBM`
- `ray-project/ray`
- `streamlit/streamlit`
- `Dao-AILab/flash-attention`

### Community 2 (37 repos)
- `stan-dev/pystan`
- `sqlfluff/sqlfluff`
- `pallets/flask`
- `pallets/jinja`
- `pallets/werkzeug`
- `pallets/click`
- `django/django`
- `encode/django-rest-framework`
- `psf/requests`
- `urllib3/urllib3`

### Community 3 (30 repos)
- `tensorflow/tensorflow`
- `keras-team/keras`
- `catboost/catboost`
- `optuna/optuna`
- `wandb/wandb`
- `mlflow/mlflow`
- `fastapi/fastapi`
- `explosion/spaCy`
- `PrefectHQ/prefect`
- `alembic/alembic`

### Community 4 (24 repos)
- `huggingface/transformers`
- `huggingface/datasets`
- `huggingface/diffusers`
- `huggingface/accelerate`
- `langchain-ai/langchain`
- `run-llama/llama_index`
- `openai/openai-python`
- `ollama/ollama`
- `deepspeedai/DeepSpeed`
- `fastai/fastai`

### Community 5 (21 repos)
- `apache/airflow`
- `aio-libs/aiohttp`
- `cherrypy/cherrypy`
- `falconry/falcon`
- `eventlet/eventlet`
- `graphql-python/graphene`
- `strawberry-graphql/strawberry`
- `psycopg/psycopg`
- `tox-dev/tox`
- `kubernetes-client/python`

### Community 6 (14 repos)
- `stanfordnlp/stanza`
- `pymc-devs/pymc`
- `cython/cython`
- `jupyter/notebook`
- `jupyterlab/jupyterlab`
- `ipython/ipython`
- `jupyter/jupyter_core`
- `jupyter-server/jupyter_server`
- `voila-dashboards/voila`
- `tornadoweb/tornado`

### Community 7 (8 repos)
- `Lightning-AI/pytorch-lightning`
- `karpathy/nanoGPT`
- `karpathy/micrograd`
- `ultralytics/ultralytics`
- `ultralytics/yolov5`
- `facebookresearch/detectron2`
- `facebookresearch/faiss`
- `allenai/allennlp`

### Community 8 (4 repos)
- `xorbitsai/inference`
- `xorbitsai/xorbits`
- `hiyouga/LlamaFactory`
- `zai-org/ChatGLM-6B`

### Community 9 (2 repos)
- `grpc/grpc`
- `protocolbuffers/protobuf`

### Community 10 (2 repos)
- `boto/boto3`
- `aws/aws-cli`

---

## 4. Isolated Repositories Handling (Cluster -1)

The following 12 repositories share zero contributors with the rest of the 206 repositories:
- `capitalone/datacompy` (ID: 126487536)
- `dagster-io/dagster` (ID: 131619646)
- `cleanlab/cleanlab` (ID: 132975485)
- `milvus-io/milvus` (ID: 208728772)
- `qdrant/qdrant` (ID: 268163609)
- `frictionlessdata/frictionless-py` (ID: 28409905)
- `whylabs/whylogs` (ID: 287642401)
- `deepchecks/deepchecks` (ID: 415969774)
- `mosaicml/composer` (ID: 416157128)
- `chroma-core/chroma` (ID: 546206616)
- `meta-llama/llama` (ID: 601538369)
- `weasyprint/weasyprint` (ID: 620549478)

### Recommendation Strategy for Isolated Repositories:
In the downstream recommendation engine (Part 5), isolated repositories cannot be reached via bipartite random walks or co-contribution graph hops. They will be handled via:
1. **Fallback to Global Popularity**: Recommending top-starred repositories in the matching domain/topic.
2. **Cold-Start Category Priors**: Using repository topic metadata as a content fallback.

---

## 5. Final Recommendation & Selected Output

- **Selected Partition**: **Leiden Community Detection**
- **Justification**:
  1. Leiden fixes Louvain's well-documented theoretical flaw of aggregating disconnected subgraphs within communities.
  2. Leiden exhibits strictly higher partition stability across random seeds (ARI = 0.7826).
  3. The resulting technological clusters map directly onto the 4 real-world target domains (Machine Learning, Data Science, Web Frameworks, Developer Tools).

---

## 6. Output Artifacts Generated

- Tabular Assignments: `data/community_detection/repository_communities.csv`
- Metrics Record: `data/community_detection/community_detection_metrics.json`
- Annotated GraphML: `data/community_detection/repository_communities.graphml`
- Visualizations:
  - `data/community_detection/visualizations/algorithm_comparison_metrics.png`
  - `data/community_detection/visualizations/community_size_distribution.png`
  - `data/community_detection/visualizations/repository_communities_network.png`
