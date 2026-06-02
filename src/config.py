import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_SYNTHETIC_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model IDs
LLM_MODEL_ID = "AliMaatouk/LLama-3-8B-Tele-it"
EMBED_MODEL_ID = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"

# Qdrant configuration
QDRANT_PATH = str(DATA_DIR / "qdrant_storage")
COLLECTION_NAME = "telecom_specs"

# Chunking configuration
LEAF_MIN = 200
LEAF_MAX = 500
SECTION_MIN = 500
SECTION_MAX = 1500
SUMMARY_MAX = 400

# Retrieval configuration
TOP_K = 20
RRF_K = 60
RERANK_TOP_K = 5
RELEVANCE_THRESHOLD = 0.5
CONTEXT_TOKEN_BUDGET = 3500

# Agent configuration
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.8
CLARIFY_THRESHOLD = 0.5

# QLoRA Fine-tuning configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4

# Credentials
HF_TOKEN = os.getenv("HF_TOKEN")
