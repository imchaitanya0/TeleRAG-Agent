#!/usr/bin/env python3
"""
scripts/kaggle_setup.py

BULLETPROOF Kaggle launcher for TeleRAG-Agent.
Handles EVERY known compatibility issue between our code and Kaggle's environment.

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
    print(f"\n{'═'*60}\n  {msg}\n{'═'*60}")

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
# 2. MONKEY-PATCH torch._utils FOR COMPATIBILITY
#    PyTorch 2.10+ removed _maybe_view_chunk_cat but sentence-transformers
#    2.7.0 still references it internally. We polyfill it.
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 2/6 — Patch PyTorch + Install Dependencies")

print("  [2a] Patching torch._utils._maybe_view_chunk_cat ...")
import torch._utils
if not hasattr(torch._utils, '_maybe_view_chunk_cat'):
    def _maybe_view_chunk_cat(tensors, dim):
        return torch.cat(tensors, dim)
    torch._utils._maybe_view_chunk_cat = _maybe_view_chunk_cat
    print("  ✅ Patch applied (torch._utils._maybe_view_chunk_cat polyfilled)")
else:
    print("  ✅ Already present, no patch needed")

print("  [2b] Pinning sentence-transformers==2.7.0 ...")
run("pip install 'sentence-transformers==2.7.0' --force-reinstall -q 2>&1 | tail -3", check=False)

print("  [2c] Installing remaining deps (NOT touching Gradio — using Kaggle's built-in) ...")
# IMPORTANT: We do NOT install gradio here. Kaggle ships with Gradio 5.50.
# Our UI code is now compatible with both Gradio 4.x and 5.x.
run("""pip install \
    'langgraph>=0.2.0' 'langchain-core>=0.2.0' \
    'qdrant-client>=1.9.0' 'fastembed>=0.3.6' \
    'FlagEmbedding>=1.2.0' \
    'python-dotenv' 'pyyaml' 'networkx' 'tqdm' \
    -q 2>&1 | tail -3""", check=False)

# Verify
import importlib
try:
    import sentence_transformers as st
    importlib.reload(st)
    print(f"  ✅ sentence-transformers: {st.__version__}")
except Exception as e:
    print(f"  ❌ import failed: {e}")
    sys.exit(1)

try:
    import gradio
    print(f"  ✅ gradio: {gradio.__version__} (Kaggle built-in, not downgraded)")
except Exception as e:
    print(f"  ⚠ gradio not found, installing...")
    run("pip install gradio -q 2>&1 | tail -3", check=False)

# ─────────────────────────────────────────────────────────────────────────────
# 3. ENVIRONMENT VARIABLES — CRITICAL FOR KAGGLE
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 3/6 — Configure Environment for Kaggle")

# FIX: The .env file sets QDRANT_URL=http://localhost:6333 which tries
# to connect to a Docker server that doesn't exist on Kaggle.
os.environ.pop("QDRANT_URL", None)
os.environ["QDRANT_URL"] = ""
print("  ✅ Cleared QDRANT_URL (forces local disk mode, no Docker needed)")

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

# ─────────────────────────────────────────────────────────────────────────────
# 5. PRE-DOWNLOAD ALL MODELS WITH PROGRESS
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 5/6 — Download & Verify Models")

LLM_MODEL_ID    = "AliMaatouk/LLama-3-8B-Tele-it"
EMBED_MODEL_ID  = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"

print(f"  Models: {LLM_MODEL_ID}, {EMBED_MODEL_ID}, {RERANK_MODEL_ID}")

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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
print("\n  ✅ All models verified!")

if args.no_launch:
    print("\n--no-launch: setup complete, UI not started.")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# 6. LAUNCH GRADIO
#    We launch as a subprocess but MUST apply the torch patch there too.
#    We do this by writing a tiny wrapper that patches + runs app.py.
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 6/6 — Launch Gradio UI")
print("  Models are cached — UI should start in ~30s")
print("  Look for the gradio.live URL below\n" + "-"*60)

# Write a wrapper script so the subprocess also has the torch._utils patch
wrapper = REPO_DIR / "scripts" / "_launch_patched.py"
wrapper.write_text(f'''#!/usr/bin/env python3
"""Auto-generated wrapper that patches torch._utils before launching app.py."""
import sys, os
sys.path.insert(0, "{REPO_DIR}")
os.chdir("{REPO_DIR}")

# Clear QDRANT_URL to force local disk mode
os.environ["QDRANT_URL"] = ""

# Patch torch._utils for PyTorch 2.10+ compatibility
import torch._utils
if not hasattr(torch._utils, "_maybe_view_chunk_cat"):
    def _maybe_view_chunk_cat(tensors, dim):
        import torch
        return torch.cat(tensors, dim)
    torch._utils._maybe_view_chunk_cat = _maybe_view_chunk_cat

# Now launch the actual app
from src.ui.app import main
main()
''')

env = os.environ.copy()
env["QDRANT_URL"] = ""

cmd = [sys.executable, str(wrapper)]
if args.share:
    cmd.extend(["--share"])

# We need to pass --share via sys.argv since the wrapper calls main()
# Actually, let's just pass it through the wrapper directly
wrapper.write_text(f'''#!/usr/bin/env python3
"""Auto-generated wrapper that patches torch._utils before launching app.py."""
import sys, os
sys.path.insert(0, "{REPO_DIR}")
os.chdir("{REPO_DIR}")

# Clear QDRANT_URL to force local disk mode
os.environ["QDRANT_URL"] = ""

# Patch torch._utils for PyTorch 2.10+ compatibility
import torch._utils
if not hasattr(torch._utils, "_maybe_view_chunk_cat"):
    def _maybe_view_chunk_cat(tensors, dim):
        import torch
        return torch.cat(tensors, dim)
    torch._utils._maybe_view_chunk_cat = _maybe_view_chunk_cat

# Forward CLI args (e.g. --share)
sys.argv = ["app.py"] + sys.argv[1:]

# Now launch the actual app
from src.ui.app import main
main()
''')

cmd = [sys.executable, str(wrapper)]
if args.share:
    cmd.append("--share")

proc = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, env=env, cwd=str(REPO_DIR),
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
print(f"\nExit code: {proc.returncode}")
