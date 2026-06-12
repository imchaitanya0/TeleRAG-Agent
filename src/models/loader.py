"""
src/models/loader.py

Singleton model loader for TeleRAG-Agent.

- On GPU (Kaggle/production): loads in 4-bit with device_map="auto" via bitsandbytes.
- On CPU (local Mac dev): loads in float32 with explicit cpu device, no quantization.
- LoRA adapter is merged when a lora_repo is supplied.
- Singleton pattern: loads once, reuses on every subsequent call.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import LLM_MODEL_ID, HF_TOKEN

# ─────────────────────────────────────────────
# Singleton state
# ─────────────────────────────────────────────
_model = None
_tokenizer = None

HAS_CUDA = torch.cuda.is_available()
HAS_BNB = False
try:
    import bitsandbytes  # noqa: F401
    HAS_BNB = True
except ImportError:
    pass


def _load_model_and_tokenizer(
    base_model_id: str,
    lora_repo: Optional[str],
    load_in_4bit: bool,
    hf_token: Optional[str],
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Internal loader — call via get_model()."""

    # ── Tokenizer ──────────────────────────────────────────────
    print(f"[Loader] Loading tokenizer from {base_model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        token=hf_token or None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Decide load strategy ───────────────────────────────────
    use_4bit = load_in_4bit and HAS_CUDA and HAS_BNB
    if load_in_4bit and not HAS_CUDA:
        print("[Loader] No CUDA GPU detected — loading in float32 on CPU (slower, but works).")
    if load_in_4bit and not HAS_BNB:
        print("[Loader] bitsandbytes not installed — skipping 4-bit quantization.")

    # ── Model kwargs ───────────────────────────────────────────
    model_kwargs: dict = dict(
        token=hf_token or None,
        trust_remote_code=True,
    )

    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    else:
        # CPU / no-bitsandbytes path — needs accelerate for device_map too, so
        # just load to CPU explicitly with no device_map.
        model_kwargs["dtype"] = torch.float32

    print(f"[Loader] Loading model from {base_model_id} "
          f"({'4-bit GPU' if use_4bit else 'float32 CPU'}) ...")
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    if not use_4bit:
        model = model.to("cpu")

    model.config.use_cache = True

    # ── LoRA adapter ───────────────────────────────────────────
    if lora_repo:
        try:
            from peft import PeftModel
            print(f"[Loader] Merging LoRA adapter from {lora_repo} ...")
            model = PeftModel.from_pretrained(model, lora_repo, token=hf_token or None)
            print("[Loader] LoRA adapter merged successfully.")
        except Exception as e:
            print(f"[Loader] Warning: could not load LoRA adapter ({e}). "
                  "Continuing with base model.")

    model.eval()
    print("[Loader] Model ready.")
    return model, tokenizer


def get_model(
    base_model_id: Optional[str] = None,
    lora_repo: Optional[str] = None,
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Return the singleton (model, tokenizer).  Loads on first call, caches thereafter.

    Args:
        base_model_id: HuggingFace model ID.  Defaults to config.LLM_MODEL_ID.
        lora_repo:     HuggingFace LoRA repo ID, e.g. "chaitanya-k/TeleRAG-LoRA".
                       Pass None to use the base model only.
        load_in_4bit:  Enable 4-bit quantization on GPU.  Auto-disabled on CPU.
        hf_token:      HuggingFace access token.  Defaults to config.HF_TOKEN.
    """
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    _model, _tokenizer = _load_model_and_tokenizer(
        base_model_id=base_model_id or LLM_MODEL_ID,
        lora_repo=lora_repo,
        load_in_4bit=load_in_4bit,
        hf_token=hf_token or HF_TOKEN,
    )
    return _model, _tokenizer


def reset_model():
    """Force-unload the cached model (useful for adapter switching or tests)."""
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    if HAS_CUDA:
        torch.cuda.empty_cache()
    print("[Loader] Model cache cleared.")


# ─────────────────────────────────────────────
# Smoke test: python src/models/loader.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model, tokenizer = get_model(load_in_4bit=False, lora_repo=None)
    prompt = "### Question:\nQuestion: What is 5G NR?\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nModel output: {response}")
    print("\n[Smoke test passed]")
