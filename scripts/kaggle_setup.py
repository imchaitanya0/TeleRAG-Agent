#!/usr/bin/env python3
"""
scripts/kaggle_setup.py

One-shot TeleRAG-Agent setup and launch for Kaggle.

Usage (in a Kaggle notebook cell):
    !python /kaggle/working/TeleRAG-Agent/scripts/kaggle_setup.py \\
        --qdrant-path /kaggle/input/telerag-qdrant-db/qdrant_storage \\
        --share

Arguments:
    --qdrant-path  Path to the qdrant_storage folder in your attached Kaggle dataset.
    --share        Add this flag to get a public gradio.live URL.
    --no-launch    Run setup only, don't start the UI (useful for debugging).
"""

import os
import sys
import subprocess
import time
import argparse
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 0. ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="TeleRAG-Agent Kaggle Setup")
parser.add_argument(
    "--qdrant-path",
    type=str,
    default="/kaggle/input/telerag-qdrant-db/qdrant_storage",
    help="Path to qdrant_storage in your Kaggle input dataset",
)
parser.add_argument("--share", action="store_true", help="Launch with public gradio.live URL")
parser.add_argument("--no-launch", action="store_true", help="Run setup only, skip UI launch")
args = parser.parse_args()

REPO_DIR = Path(__file__).resolve().parent.parent
QDRANT_DEST = REPO_DIR / "data" / "qdrant_storage"


def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def run(cmd: str, check: bool = True) -> int:
    ret = os.system(cmd)
    if check and ret != 0:
        print(f"  ❌ Command failed (exit {ret}): {cmd[:80]}")
    return ret


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENVIRONMENT DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 1 / 5 — Environment Diagnostics")

import torch

print(f"  Python:  {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU RAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

# Disk check
disk_info = subprocess.check_output("df -h / | tail -1", shell=True, text=True).split()
print(f"  Disk:    {disk_info[2]} used / {disk_info[1]} total ({disk_info[4]} full)")
print(f"  Repo:    {REPO_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. INSTALL DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 2 / 5 — Install / Fix Dependencies")

print("  [2a] Forcing sentence-transformers==2.7.0 (avoids torchcodec/FFmpeg crash)...")
# sentence-transformers >= 3.x imports torchcodec -> needs FFmpeg .so files
# that are NOT present in Kaggle. 2.7.0 is the last clean version.
# Note: CrossEncoder.__init__ in 2.7.0 does NOT accept 'token' kwarg.
# Authentication is done globally via huggingface_hub.login() instead.
run("pip install 'sentence-transformers==2.7.0' --force-reinstall -q 2>&1 | tail -3", check=False)

print("  [2b] Installing remaining dependencies...")
run("""pip install \
    'langgraph>=0.2.0' \
    'langchain-core>=0.2.0' \
    'qdrant-client>=1.9.0' \
    'fastembed>=0.3.6' \
    'FlagEmbedding>=1.2.0' \
    'python-dotenv' 'pyyaml' 'networkx' 'tqdm' \
    'gradio>=4.0.0,<5.0.0' \
    -q 2>&1 | tail -3""", check=False)

# Verify
import importlib
try:
    import sentence_transformers as st
    importlib.reload(st)
    ver = st.__version__
    print(f"  ✅ sentence-transformers: {ver}")
    if not ver.startswith("2."):
        print(f"  ⚠️  Warning: still on {ver}. Restarting kernel may be needed.")
except Exception as e:
    print(f"  ❌ sentence_transformers import failed: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. QDRANT DATABASE SETUP
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 3 / 5 — Qdrant Database Setup")

QDRANT_DEST.parent.mkdir(parents=True, exist_ok=True)
qdrant_src = Path(args.qdrant_path)

if QDRANT_DEST.exists() and any(QDRANT_DEST.iterdir()):
    print(f"  ✅ Qdrant storage already at {QDRANT_DEST}")
elif qdrant_src.exists():
    print(f"  Copying from {qdrant_src} → {QDRANT_DEST} ...")
    shutil.copytree(str(qdrant_src), str(QDRANT_DEST))
    size = subprocess.check_output(f"du -sh {QDRANT_DEST}", shell=True, text=True).strip()
    print(f"  ✅ Qdrant storage ready. Size: {size}")
else:
    print(f"  ⚠️  WARNING: {qdrant_src} not found!")
    available = list(Path("/kaggle/input").iterdir()) if Path("/kaggle/input").exists() else []
    print(f"  Available datasets: {[p.name for p in available]}")
    print("  Chat retrieval will fail. Fix --qdrant-path and re-run.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PRE-DOWNLOAD ALL MODELS WITH PROGRESS
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 4 / 5 — Download Models")

# Authenticate globally — works for ALL huggingface downloads without
# needing to pass token= kwarg to individual classes (which is version-specific).
hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print(f"  ✅ Logged into HuggingFace Hub (token length: {len(hf_token)})")
    else:
        print("  ⚠️  HF_TOKEN secret is empty. Public models will download fine.")
except Exception as e:
    print(f"  ⚠️  Could not get HF_TOKEN from Kaggle Secrets: {e}")
    print("  Add it via: Add-ons → Secrets → New Secret → Name: HF_TOKEN")

LLM_MODEL_ID    = "AliMaatouk/LLama-3-8B-Tele-it"
EMBED_MODEL_ID  = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"

print(f"\n  Models to download:")
print(f"    1. {LLM_MODEL_ID}")
print(f"    2. {EMBED_MODEL_ID}")
print(f"    3. {RERANK_MODEL_ID}")

# ── Embedding model ──────────────────────────────────────────────────────────
print(f"\n  [4a] Downloading embedding model (~1.3 GB) ...")
t0 = time.time()
from sentence_transformers import SentenceTransformer, CrossEncoder
# Do NOT pass token= here. sentence-transformers 2.7.0 doesn't support it.
# Authentication is inherited from huggingface_hub.login() above.
embedder = SentenceTransformer(EMBED_MODEL_ID)
vec = embedder.encode("test")
print(f"  ✅ Embedding model ready! ({time.time()-t0:.0f}s) | dim={len(vec)}")
del embedder

# ── Reranker model ───────────────────────────────────────────────────────────
print(f"\n  [4b] Downloading reranker model (~1.1 GB) ...")
t0 = time.time()
# Do NOT pass token= here. CrossEncoder in 2.7.0 doesn't support it.
reranker = CrossEncoder(RERANK_MODEL_ID)
score = reranker.predict([["test query", "test document"]])
print(f"  ✅ Reranker ready! ({time.time()-t0:.0f}s) | sample_score={float(score[0]):.4f}")
del reranker

# ── LLM ─────────────────────────────────────────────────────────────────────
print(f"\n  [4c] Downloading LLM ({LLM_MODEL_ID}) ...")
print("  Expect 10-20 min on first run. 'Loading checkpoint shards' will appear below.")
t0 = time.time()
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  ✅ Tokenizer ready ({time.time()-t0:.0f}s)")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()
elapsed = time.time() - t0
gpu_used = torch.cuda.memory_allocated() / 1e9
print(f"  ✅ LLM loaded! ({elapsed:.0f}s) | GPU used: {gpu_used:.1f} GB")

# Quick sanity test
print("\n  Running sanity inference ...")
prompt = "### Question:\nQuestion: What is 5G NR?\n\n### Answer:\n"
inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"  LLM output: {response[:200]!r}")
print("  ✅ Sanity test passed!")

del model, tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n  ✅ All models downloaded and verified!")
print("  Models cached in ~/.cache/huggingface/")

if args.no_launch:
    print("\n--no-launch flag set. Skipping UI. Setup complete.")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. LAUNCH GRADIO UI (streaming output)
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 5 / 5 — Launch Gradio UI")
print("  Look for the 'gradio.live' link in the output below.")
print("  The LLM loads from cache — expect ~30 sec before UI is ready.")
print("-" * 60)

launch_cmd = [sys.executable, str(REPO_DIR / "src" / "ui" / "app.py")]
if args.share:
    launch_cmd.append("--share")

env = os.environ.copy()
if hf_token:
    env["HF_TOKEN"] = hf_token

proc = subprocess.Popen(
    launch_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env,
    cwd=str(REPO_DIR),
)

for line in proc.stdout:
    print(line, end="", flush=True)

proc.wait()
print(f"\nProcess ended with exit code: {proc.returncode}")
