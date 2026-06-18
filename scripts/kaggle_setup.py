#!/usr/bin/env python3
"""
scripts/kaggle_setup.py

BULLETPROOF Kaggle launcher for TeleRAG-Agent.
Tested against Kaggle's June 2026 environment:
  - Python 3.12, PyTorch 2.10.0+cu128, Gradio 5.50, Tesla T4

Usage (paste in a Kaggle notebook cell):
    !python /kaggle/working/TeleRAG-Agent/scripts/kaggle_setup.py \
        --qdrant-path /kaggle/input/telerag-qdrant-db/qdrant_storage \
        --share
"""

import os
import sys
import subprocess
import time
import argparse
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 0. ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="TeleRAG-Agent Kaggle Setup")
parser.add_argument(
    "--qdrant-path", type=str,
    default="/kaggle/input/telerag-qdrant-db/qdrant_storage",
)
parser.add_argument("--share", action="store_true")
parser.add_argument("--no-launch", action="store_true")
args = parser.parse_args()

REPO_DIR = Path(__file__).resolve().parent.parent
QDRANT_DEST = REPO_DIR / "data" / "qdrant_storage"

def banner(msg):
    print(f"\n{'═'*60}\n  {msg}\n{'═'*60}", flush=True)

def run(cmd, check=True):
    ret = os.system(cmd)
    if check and ret != 0:
        print(f"  ⚠ exit {ret}: {cmd[:80]}")
    return ret

# ─────────────────────────────────────────────────────────────────────────────
# 1. DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 1/6 — Environment Check")
import torch

print(f"  Python:  {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU RAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

# ─────────────────────────────────────────────────────────────────────────────
# 2. INSTALL DEPENDENCIES
#    KEY INSIGHT: We DO NOT pin sentence-transformers to 2.7.0 anymore.
#    That old version is INCOMPATIBLE with PyTorch 2.10 (missing torch._utils
#    internal functions). Since v3.0+, torchcodec is OPTIONAL — only needed
#    for audio/video embedding, which we don't use.
#    We use Kaggle's pre-installed Gradio 5.50 (our UI is now compatible).
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 2/6 — Install Dependencies")

print("  [2a] Installing sentence-transformers (latest, PyTorch 2.10 compatible) ...")
# Do NOT pin to 2.7.0 — it uses removed torch._utils internals.
# Latest versions (3.x+) are compatible with PyTorch 2.10 and
# torchcodec is OPTIONAL (only for audio/video, which we don't use).
run("pip install -U 'sentence-transformers' -q 2>&1 | tail -5", check=False)

print("  [2b] Installing remaining deps ...")
run("""pip install \
    'langgraph>=0.2.0' 'langchain-core>=0.2.0' \
    'qdrant-client>=1.9.0' 'fastembed>=0.3.6' \
    'FlagEmbedding>=1.2.0' \
    'bitsandbytes>=0.46.1' \
    'python-dotenv' 'pyyaml' 'networkx' 'tqdm' \
    -q 2>&1 | tail -5""", check=False)

# Verify imports
print("\n  Verifying critical imports ...")
import importlib
import importlib.metadata
errors = []

def get_version(pkg_name: str, module=None) -> str:
    """Get version via importlib.metadata (works even if __version__ is absent)."""
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:
        if module and hasattr(module, "__version__"):
            return module.__version__
        return "installed"

try:
    import sentence_transformers
    v = get_version("sentence-transformers", sentence_transformers)
    print(f"  ✅ sentence-transformers: {v}")
except Exception as e:
    errors.append(f"sentence-transformers: {e}")
    print(f"  ❌ sentence-transformers: {e}")

try:
    import gradio
    v = get_version("gradio", gradio)
    print(f"  ✅ gradio: {v}")
except Exception as e:
    errors.append(f"gradio: {e}")
    print(f"  ❌ gradio: {e}")

try:
    import langgraph  # noqa — may not have __version__
    v = get_version("langgraph", langgraph)
    print(f"  ✅ langgraph: {v}")
except Exception as e:
    errors.append(f"langgraph: {e}")
    print(f"  ❌ langgraph: {e}")

try:
    import langchain_core
    v = get_version("langchain-core", langchain_core)
    print(f"  ✅ langchain-core: {v}")
except Exception as e:
    errors.append(f"langchain-core: {e}")
    print(f"  ❌ langchain-core: {e}")

try:
    from qdrant_client import QdrantClient
    v = get_version("qdrant-client")
    print(f"  ✅ qdrant-client: {v}")
except Exception as e:
    errors.append(f"qdrant-client: {e}")
    print(f"  ❌ qdrant-client: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    v = get_version("transformers")
    print(f"  ✅ transformers: {v}")
except Exception as e:
    errors.append(f"transformers: {e}")
    print(f"  ❌ transformers: {e}")

try:
    import fastembed  # noqa
    v = get_version("fastembed", fastembed)
    print(f"  ✅ fastembed: {v}")
except Exception as e:
    errors.append(f"fastembed: {e}")
    print(f"  ❌ fastembed: {e}")

try:
    import networkx  # noqa
    v = get_version("networkx", networkx)
    print(f"  ✅ networkx: {v}")
except Exception as e:
    errors.append(f"networkx: {e}")
    print(f"  ❌ networkx: {e}")

if errors:
    print(f"\n  ❌ {len(errors)} import error(s)! Cannot proceed.")
    for e in errors:
        print(f"     - {e}")
    sys.exit(1)
else:
    print(f"\n  ✅ All imports verified!")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ENVIRONMENT VARIABLES — CRITICAL FOR KAGGLE
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 3/6 — Configure Environment for Kaggle")

# The .env file sets QDRANT_URL=http://localhost:6333 which tries to connect
# to a Docker server that doesn't exist on Kaggle. We clear it.
os.environ.pop("QDRANT_URL", None)
os.environ["QDRANT_URL"] = ""
print("  ✅ Cleared QDRANT_URL (forces local disk mode)")

# Set HF token
hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        from huggingface_hub import login
        login(token=hf_token)
        print(f"  ✅ HuggingFace authenticated (token length: {len(hf_token)})")
    else:
        print("  ⚠ HF_TOKEN secret is empty")
except Exception as e:
    print(f"  ⚠ HF_TOKEN not available: {e}")

sys.path.insert(0, str(REPO_DIR))
os.chdir(str(REPO_DIR))
print(f"  ✅ Working dir: {REPO_DIR}")

# Write a clean .env for Kaggle
kaggle_env = REPO_DIR / ".env"
kaggle_env.write_text(
    "# Auto-generated for Kaggle — no Docker, use local Qdrant\n"
    "QDRANT_URL=\n"
    f"HF_TOKEN={hf_token or ''}\n"
)
print("  ✅ Wrote Kaggle-compatible .env")

# ─────────────────────────────────────────────────────────────────────────────
# 4. QDRANT DATABASE
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 4/6 — Qdrant Database")
QDRANT_DEST.parent.mkdir(parents=True, exist_ok=True)
qdrant_src = Path(args.qdrant_path)

if QDRANT_DEST.exists() and any(QDRANT_DEST.iterdir()):
    print(f"  ✅ Already at {QDRANT_DEST}")
elif qdrant_src.exists():
    print(f"  Copying {qdrant_src} → {QDRANT_DEST} ...")
    shutil.copytree(str(qdrant_src), str(QDRANT_DEST))
    size = subprocess.check_output(f"du -sh {QDRANT_DEST}", shell=True, text=True).strip()
    print(f"  ✅ Ready. {size}")
else:
    print(f"  ⚠ {qdrant_src} not found!")
    avail = list(Path("/kaggle/input").iterdir()) if Path("/kaggle/input").exists() else []
    print(f"  Available: {[p.name for p in avail]}")
    print("  Fix --qdrant-path and re-run.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PRE-DOWNLOAD ALL MODELS WITH PROGRESS
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 5/6 — Download & Verify Models")

LLM_MODEL_ID    = "AliMaatouk/LLama-3-8B-Tele-it"
EMBED_MODEL_ID  = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"

print(f"  Models:")
print(f"    1. {LLM_MODEL_ID}")
print(f"    2. {EMBED_MODEL_ID}")
print(f"    3. {RERANK_MODEL_ID}")

# ── Embedder ──
print(f"\n  [5a] Embedding model ({EMBED_MODEL_ID}) ...")
t0 = time.time()
from sentence_transformers import SentenceTransformer, CrossEncoder
embedder = SentenceTransformer(EMBED_MODEL_ID)
vec = embedder.encode("test")
print(f"  ✅ Done ({time.time()-t0:.0f}s) dim={len(vec)}")
del embedder

# ── Reranker ──
print(f"\n  [5b] Reranker model ({RERANK_MODEL_ID}) ...")
t0 = time.time()
reranker = CrossEncoder(RERANK_MODEL_ID)
score = reranker.predict([["query", "doc"]])
print(f"  ✅ Done ({time.time()-t0:.0f}s) score={float(score[0]):.4f}")
del reranker

# ── LLM ──
print(f"\n  [5c] LLM ({LLM_MODEL_ID}) ...")
print("  ⏳ Downloading ~16GB — expect 10-20 min on first run")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  ✅ Tokenizer ({time.time()-t0:.0f}s)")

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID, quantization_config=bnb, device_map="auto",
)
model.eval()
gpu_gb = torch.cuda.memory_allocated() / 1e9
print(f"  ✅ LLM loaded ({time.time()-t0:.0f}s) GPU={gpu_gb:.1f}GB")

# Sanity inference
prompt = "### Question:\nQuestion: What is 5G NR?\n\n### Answer:\n"
inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"  ✅ Inference test: {resp[:120]!r}")

del model, tokenizer
torch.cuda.empty_cache()
print("\n  ✅ All models downloaded and verified!")

if args.no_launch:
    print("\n--no-launch: setup complete, UI not started.")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# 6. LAUNCH GRADIO
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 6/6 — Launch Gradio UI")
print("  Building UI and launching server in this process ...")
print("  The gradio.live URL will appear below. Ctrl+C to stop.\n" + "-"*60)

# Run in-process: avoids subprocess buffering and hidden crashes.
# All errors and the gradio.live URL will be visible directly here.
sys.argv = ["app.py"]
if args.share:
    sys.argv.append("--share")

from src.ui.app import main as launch_ui
launch_ui()

