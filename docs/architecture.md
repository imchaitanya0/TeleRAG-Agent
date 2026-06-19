# TeleRAG-Agent: System Architecture

## Overview

TeleRAG-Agent is an **agentic Retrieval-Augmented Generation (RAG) system** designed for the telecom domain. It answers natural-language questions about 3GPP specifications, diagnoses O-RAN network faults, and detects KPI anomalies — all powered by a domain-specialized LLM and a hybrid retrieval pipeline running on a single GPU.

---

## High-Level Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────────────────────┐
│              Security Layer                           │
│  • Input sanitizer (prompt injection detection)       │
│  • PII redactor (output scanning)                     │
│  • JSON audit logger                                  │
└───────────────────────┬───────────────────────────────┘
                        │ clean query
                        ▼
┌───────────────────────────────────────────────────────┐
│           LangGraph Agentic Loop                      │
│                                                       │
│   PLAN ──▶ RETRIEVE ──▶ GENERATE ──▶ REFLECT         │
│              ▲                           │             │
│              └─────── (mid-conf) ────────┘             │
│                                                       │
│   Max 3 iterations. Exits on: high confidence,        │
│   max iterations hit, or clarification needed.        │
└───────────────────────┬───────────────────────────────┘
                        │ final answer + sources + confidence
                        ▼
                    Gradio UI
```

---

## Component Breakdown

### 1. Security Layer (`src/security/`)

| Module | Purpose |
|---|---|
| `input_sanitizer.py` | Regex + keyword pattern matching to block prompt injection, OS commands, SQL injection |
| `pii_detector.py` | Regex-based scanner that redacts IPv4, MAC addresses, IMSI, cell IDs from model output |
| `audit_logger.py` | Writes every query, security event, and response to a JSON audit trail |

### 2. LangGraph Agent (`src/agent/`)

The agent uses a **4-node state machine** compiled with LangGraph:

#### PLAN node (`nodes/plan.py`)
- Classifies the query into one of 5 types: `spec_lookup`, `fault_diagnosis`, `kpi_analysis`, `comparison`, `open_ended`
- Decomposes complex queries into sub-queries
- Selects which tools to use

#### RETRIEVE node (`nodes/retrieve.py`)
Executes the hybrid retrieval pipeline:
1. **Dense search** — BGE-large-en-v1.5 embeddings → Qdrant vector search
2. **Sparse search** — BM25 via Qdrant sparse vectors
3. **KG search** — NetworkX heading index for section-level boost
4. **RRF fusion** — Reciprocal Rank Fusion (k=60) merges all 3 sources
5. **Cross-encoder reranking** — BGE-reranker-v2-m3 rescores top-20 → top-5

#### GENERATE node (`nodes/generate.py`)
- Builds a structured prompt with retrieved context
- Generates answer using LLaMA-3-8B-Tele-it (4-bit quantized via bitsandbytes)
- Appends inline citations `[Source: TS 38.331, §5.3.3]`

#### REFLECT node (`nodes/reflect.py`)
- Scores confidence (0.0–1.0) based on: source count, rerank scores, answer length, spec citations
- **≥ 0.8**: route to END (high confidence)
- **0.5–0.8**: route back to RETRIEVE with expanded query (re-plan)
- **< 0.5**: route to END with clarification request

### 3. Retrieval Pipeline (`src/retrieval/`)

```
query
  │
  ├─▶ DenseSearcher.search()     →  top-40 results (cosine similarity)
  ├─▶ SparseSearcher.search()    →  top-40 results (BM25 term overlap)
  └─▶ KGSearcher.search()        →  section IDs (heading graph traversal)
            │
            ▼
       HybridRetriever.rrf_fusion()   →  top-20 merged by RRF score
            │
            ▼
       Reranker.rerank()              →  top-5 by cross-encoder score
            │
            ▼
       ContextAssembler.assemble()    →  3,500-token context string
```

### 4. Models (`src/models/`)

| Model | Purpose | Size | Quantization |
|---|---|---|---|
| `AliMaatouk/LLama-3-8B-Tele-it` | Answer generation | 8B params | 4-bit NF4 (bitsandbytes) |
| `BAAI/bge-large-en-v1.5` | Dense embeddings | 335M params | FP32 |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder reranking | 568M params | FP32 |
| `Imchaitanya/TeleRAG-LoRA` | LoRA fine-tuned adapter | 20M params | merged at inference |


The model loader (`src/models/loader.py`) uses a singleton pattern — the LLM is loaded once and cached for the lifetime of the process.

### 5. Knowledge Base (`data/qdrant_storage/`)

- **Vector DB**: Qdrant (local path mode, no server required on Kaggle)
- **Collection**: `telecom_specs`
- **Chunks**: ~30,000 leaf chunks from 15+ 3GPP specifications
- **Chunk schema**: `{spec_number, clause_string, clause_title, content, chunk_type, parent_id}`
- **Vectors**: 1024-dim dense (BGE-large) + sparse BM25 indices

### 6. Agent Tools (`src/agent/tools/`)

| Tool | Data Source | What It Does |
|---|---|---|
| `spec_retriever.py` | Qdrant vector DB | Hybrid search for spec content |
| `alarm_analyzer.py` | `data/raw/oran/` + `data/synthetic/alarms.json` | Storm detection, frequency analysis, RCA |
| `kpi_calculator.py` | `data/synthetic/kpis.csv` | Z-score anomaly detection, 3GPP TS 28.552 thresholds |

### 7. Gradio UI (`src/ui/app.py`)

Four tabs:
1. **💬 Ask a Question** — Chat with full RAG pipeline
2. **🚨 O-RAN Alarm Analysis** — Alarm storm detection + RCA
3. **📊 KPI Anomaly Detection** — Z-score anomaly reporting
4. **ℹ️ About** — Architecture, model cards, datasets

---

## Data Flow: Complete End-to-End

```
User: "What is RRC Connection Reconfiguration?"
  │
  1. InputSanitizer → passes (no injection patterns)
  2. PLAN → type=spec_lookup, sub_queries=["RRC reconfiguration 5G NR procedure"]
  3. RETRIEVE:
       dense_results: [TS 38.331 §5.3.3, TS 38.331 §5.3.4, TS 38.321 §5.4.1 ...]
       sparse_results: [TS 38.331 §5.3.3, TS 38.300 §9.2.3 ...]
       kg_results: [section "RRC Connection Reconfiguration"]
       RRF fusion → top-20
       reranker → top-5, TS 38.331 §5.3.3 score=0.92
  4. GENERATE:
       context: "5.3.3 RRC Connection Reconfiguration\nThe RRC connection reconfiguration..."
       prompt: "### Question:\n[context]\nWhat is RRC...\n### Answer:\n"
       answer: "RRC Connection Reconfiguration is the procedure by which..."
       citation: "[Source: TS 38.331, §5.3.3]"
  5. REFLECT:
       confidence = 0.87 (high source count, top-1 score 0.92)
       → done (route to END)
  6. PIIDetector → no PII found
  7. AuditLogger → logs session_id, query, confidence, latency
  8. UI → renders answer + sources table + thinking trace
```

---

## Infrastructure

| Component | Technology |
|---|---|
| Compute | Kaggle T4 GPU (16GB VRAM) |
| Framework | Python 3.12, PyTorch 2.10, LangGraph 1.x |
| Vector DB | Qdrant (local disk, no server) |
| UI | Gradio 5.50 / 6.x (version-aware) |
| Embeddings | fastembed (sparse) + sentence-transformers (dense) |
