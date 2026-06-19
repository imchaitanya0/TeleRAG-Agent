# Installation Guide

## Prerequisites

- Python 3.10+
- Git
- 16GB GPU VRAM (for inference) or CPU-only mode (slower)
- HuggingFace account + access token

## Local Development Setup (Mac/Linux, CPU-only)

```bash
# Clone the repo
git clone https://github.com/imchaitanya0/TeleRAG-Agent.git
cd TeleRAG-Agent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: set HF_TOKEN=your_token, QDRANT_URL= (empty = local disk)
```

## Set Up the Knowledge Base (Qdrant)

The pre-built Qdrant database is available as a Kaggle dataset:
`chaitanyakadupukutla/telerag-qdrant-db`

```bash
# Download and extract to data/qdrant_storage/
# Option 1: Kaggle CLI
kaggle datasets download chaitanyakadupukutla/telerag-qdrant-db
unzip telerag-qdrant-db.zip -d data/qdrant_storage/

# Option 2: Manual download from Kaggle and extract
```

## Generate Synthetic Data (for Alarm + KPI tabs)

```bash
python src/data/synthetic_alarms.py
python src/data/synthetic_kpis.py
```

## Run the App Locally

```bash
# CPU mode (no GPU — model loads in float32, ~10min per query)
QDRANT_URL="" python src/ui/app.py

# With GPU (much faster — requires CUDA)
QDRANT_URL="" python src/ui/app.py
```

## Kaggle Deployment (Recommended — Free T4 GPU)

See the main [README](../README.md) for the 2-cell Kaggle notebook setup.

## Run Evaluation

```bash
# Retrieval metrics (no LLM needed — fast)
QDRANT_URL="" python scripts/run_eval.py --mode retrieval --n 50

# MCQ accuracy (needs GPU)
QDRANT_URL="" python scripts/run_eval.py --mode accuracy --n 50 --lora

# Full ablation (needs GPU, ~1-2 hours for 50 questions × 4 experiments)
QDRANT_URL="" python scripts/run_eval.py --mode ablation --n 50
```

Results are saved to `eval_results/`.
