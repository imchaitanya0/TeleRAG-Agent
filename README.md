# TeleRAG-Agent

- **Problem Statement Number** - 10
- **Problem Statement Title** - RAG based Future-Ready Telecom RAN Assistant
- **Team name** - under_served
- **Team members (Names)** - Chaitanya Kadupukutla
- **Institute/College Name** - IIIT Hyderabad, Gachibowli, Hyderabad 500032
- **Final Presentation Google Drive Link** - *[To be added after recording]*
- **Full Submission Demo Video Link** - *[To be added after recording]*
- **Setup & Result Reproducibility Video Link** - *[To be added after recording]*

---

## About TeleRAG-Agent

TeleRAG-Agent is an **agentic RAG system** for telecom Radio Access Networks that automates 3GPP spec question-answering, O-RAN alarm analysis, and KPI anomaly detection. It uses a **LangGraph 4-node agentic loop** (PLAN → RETRIEVE → GENERATE → REFLECT) with confidence-based re-retrieval, hybrid search (dense + sparse + knowledge graph), cross-encoder re-ranking, and a QLoRA fine-tuned LLaMA-3-8B backbone.

### Key Features

- **Hybrid Retrieval:** Dense (BGE-large) + Sparse (BM25) + KG heading graph, fused with Reciprocal Rank Fusion
- **Cross-Encoder Reranking:** BGE-reranker-v2-m3 for precision over the top-k candidates
- **Agentic Loop:** LangGraph state machine with PLAN → RETRIEVE → GENERATE → REFLECT nodes
- **Confidence-based Re-retrieval:** If the REFLECT node scores below threshold, the agent re-plans and re-retrieves
- **QLoRA Fine-tuning:** LoRA adapter trained on TeleQnA for telecom domain specialization
- **O-RAN Alarm Analysis:** Storm detection, severity breakdown, root cause analysis
- **KPI Anomaly Detection:** Z-score statistical analysis aligned with 3GPP TS 28.552 thresholds
- **Security:** Prompt injection detection, PII masking, audit logging
- **Explainability:** Source citations with spec clause numbers, full agent thinking trace

---

### Project Artefacts

- **Technical Documentation** - See the [`docs/`](docs/) folder:
  - [`docs/architecture.md`](docs/architecture.md) — System architecture with Mermaid diagrams
  - [`docs/ax.md`](docs/ax.md) — Agentic AI deep-dive: workflows, tool use, what worked and what didn't
  - [`docs/installation.md`](docs/installation.md) — Installation and setup instructions
- **Source Code** - See the [`src/`](src/) folder containing all modules:
  - `src/agent/` — LangGraph agent with 4 nodes and 3 specialized tools
  - `src/retrieval/` — Hybrid search, fusion, reranker, context assembler
  - `src/models/` — Singleton model loader, inference wrapper
  - `src/evaluation/` — Retrieval metrics, answer accuracy, ablation study
  - `src/security/` — Input sanitizer, PII detector, audit logger
  - `src/ui/` — Gradio web interface with 4 tabs
  - `src/data/` — Data preparation, synthetic generators
  - `src/ingestion/` — PDF/HTML parsers, hierarchical chunker, embedder, indexer

### Models Used

| Model                         | Purpose                        | Link                                                             |
| ----------------------------- | ------------------------------ | ---------------------------------------------------------------- |
| AliMaatouk/LLama-3-8B-Tele-it | Base LLM (telecom-specialized) | [HuggingFace](https://huggingface.co/AliMaatouk/LLama-3-8B-Tele-it) |
| BAAI/bge-large-en-v1.5        | Dense embedding model          | [HuggingFace](https://huggingface.co/BAAI/bge-large-en-v1.5)        |
| BAAI/bge-reranker-v2-m3       | Cross-encoder reranker         | [HuggingFace](https://huggingface.co/BAAI/bge-reranker-v2-m3)       |
| Qdrant/bm25                   | Sparse BM25 embedding          | [HuggingFace](https://huggingface.co/Qdrant/bm25)                   |

### Models Published

| Model                    | Description                              | Link                                                        |
| ------------------------ | ---------------------------------------- | ----------------------------------------------------------- |
| Imchaitanya/TeleRAG_LoRA | QLoRA fine-tuned adapter for telecom Q&A | [HuggingFace](https://huggingface.co/Imchaitanya/TeleRAG_LoRA) |

### Datasets Used

| Dataset                  | Description                                 | Link                                         |
| ------------------------ | ------------------------------------------- | -------------------------------------------- |
| TeleQnA                  | 10K telecom Q&A from 3GPP standards         | [GitHub](https://github.com/netop-team/TeleQnA) |
| 3GPP Release 16/18 Specs | Technical specifications (TS 38.xxx series) | [3GPP](https://www.3gpp.org/specifications)     |
| O-RAN Alliance Data      | RAN alarm logs and KPI metrics              | [netop-team](https://github.com/netop-team)     |

### Datasets Published

| Dataset                    | Description                                        | Link                                                                   |
| -------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| Imchaitanya/TeleRAG-Chunks | 35K processed spec chunks + KG graph + eval splits | [HuggingFace](https://huggingface.co/datasets/Imchaitanya/TeleRAG-Chunks) |

---

### Target KPIs

| Metric                     | Target    | Achieved                              | Notes |
| -------------------------- | --------- | ------------------------------------- | ----- |
| Mean Reciprocal Rank (MRR) | Above 75% | ✅ **100% (MRR@10 = 1.0)**            | Perfect retrieval — relevant doc always at rank 1 |
| Top-k Accuracy (Recall@5)  | Above 85% | ✅ **100% (Recall@5 = 1.0)**          | Every relevant chunk found in top 5 |
| MCQ Accuracy               | Above 80% | ⚠️ **70%** (50-question ablation set) | Full dataset mixes 3GPP specs + research papers |
| Recall@1                   | Above 85% | ✅ **100% (Recall@1 = 1.0)**          | Top-1 result is always relevant |
| Faithfulness               | Above 90% | ✅ Source citations in every response | Clause-level citations, no hallucinated sources |
| LoRA Fine-tuning Value     | >0% improvement | ✅ **+64 pp** improvement       | Base model: 6% (86% abstentions) → With LoRA: 70% |

### Ablation Study Results (June 21, 2026 — 50 questions)

| Experiment          | Accuracy | MRR@10 | Recall@5 | Latency  | Notes |
|---------------------|----------|--------|----------|----------|-------|
| **Full System**     | **70%**  | 1.0000 | 100%     | 10.2s    | Dense + Sparse + KG + Reranker + LoRA |
| No Re-ranker        | 70%      | 1.0000 | 100%     | 8.6s     | Reranker adds precision, not accuracy here |
| Sparse (BM25) Only  | 70%      | 0.9800 | 98%      | 8.1s     | Slight retrieval degradation (-2% recall) |
| **No Fine-tuning**  | **6%**   | 1.0000 | 100%     | 8.6s     | 86% abstentions — base model refuses MCQ format |

> **Key insight:** LoRA fine-tuning is the critical component. Without it, the base model abstains on 86% of questions. The +64 percentage point improvement is the core contribution of this work.

---

#### Final Presentation

The final presentation covers: system architecture, agentic workflow design, hybrid retrieval pipeline, QLoRA fine-tuning results, ablation study with the key LoRA vs no-LoRA comparison, security features, and live demo walkthrough.

#### Full Submission Demo Video

Demonstrates: 3GPP spec Q&A with citations, prompt injection blocking, O-RAN alarm storm detection, KPI anomaly analysis, and agent thinking trace visualization.

#### Setup & Result Reproducibility Video

Shows complete reproduction from scratch: `git pull` on Kaggle, automatic dataset download from HuggingFace via `kaggle_setup.py`, Qdrant database loading, full ablation evaluation script execution, and KPI metric reproduction.

---

### Kaggle Notebook Setup (2 cells)

**Cell 1 — Setup & install:**
```python
# Clone / pull latest code
import subprocess
if not __import__('os').path.exists('/kaggle/working/TeleRAG-Agent'):
    subprocess.run(['git', 'clone', 'https://github.com/imchaitanya0/TeleRAG-Agent.git',
                    '/kaggle/working/TeleRAG-Agent'], check=True)
else:
    subprocess.run(['git', '-C', '/kaggle/working/TeleRAG-Agent', 'pull', 'origin', 'main'], check=True)

# Run setup: installs deps, downloads models & data from HuggingFace
import sys
sys.path.insert(0, '/kaggle/working/TeleRAG-Agent')
%run /kaggle/working/TeleRAG-Agent/scripts/kaggle_setup.py \
    --qdrant-path /kaggle/input/telerag-qdrant-db/qdrant_storage \
    --no-launch
```

**Cell 2 — Run full ablation evaluation:**
```python
import subprocess
result = subprocess.run(
    ['python', '/kaggle/working/TeleRAG-Agent/scripts/run_eval.py',
     '--mode', 'ablation', '--n', '50', '--verbose'],
    env={**__import__('os').environ, 'QDRANT_URL': ''},
    capture_output=False
)
```

**Expected output:**
```
Full System:    70%  MRR@10=1.0000  Recall@5=100%  ~10s/query
No Re-ranker:   70%  MRR@10=1.0000  Recall@5=100%  ~8.6s/query
Sparse Only:    70%  MRR@10=0.9800  Recall@5=98%   ~8.1s/query
No Fine-tuning:  6%  MRR@10=1.0000  Recall@5=100%  ~8.6s/query
```

### Attribution

This project builds upon the following open-source projects and models:

- **[TeleQnA](https://github.com/netop-team/TeleQnA)** — Telecom Q&A dataset by netop-team. Used for training data and evaluation benchmarks.
- **[Tele-LLMs](https://github.com/Ali-maatouk/Tele-LLMs)** — Telecom-specialized LLaMA models by Ali Maatouk et al. We use `LLama-3-8B-Tele-it` as our base model.
- **[LangGraph](https://github.com/langchain-ai/langgraph)** — Agentic state machine framework by LangChain. Used to build our 4-node PLAN/RETRIEVE/GENERATE/REFLECT loop.
- **[Qdrant](https://github.com/qdrant/qdrant)** — Vector database for hybrid dense+sparse search.
- **[Gradio](https://github.com/gradio-app/gradio)** — Web UI framework for the interactive demo.
- **[PEFT](https://github.com/huggingface/peft)** — Parameter-Efficient Fine-Tuning (QLoRA) by HuggingFace.
- **[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)** — 4-bit quantization for efficient GPU inference.

**New features developed by Me:**

- Complete agentic RAG pipeline with confidence-based re-retrieval
- 3-tier hierarchical chunking with parent-child expansion
- Knowledge Graph heading index for structural retrieval
- O-RAN alarm storm detection and root cause analysis
- KPI anomaly detection with 3GPP TS 28.552 thresholds
- Security module: prompt injection detection, PII masking, audit logging
- Full evaluation framework with 4-experiment ablation study
