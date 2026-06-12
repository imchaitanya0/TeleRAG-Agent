"""
src/models/inference.py

Clean inference wrapper for TeleRAG-Agent.

Wraps the raw model into a simple generate(prompt) -> str interface
that the rest of the pipeline can call without worrying about tokenization,
device placement, or decoding.
"""

import re
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.models.loader import get_model


# ──────────────────────────────────────────────────────────────
# Prompt helpers — MUST match the training format exactly
# ──────────────────────────────────────────────────────────────

def build_mcq_prompt(question: str, options: list[str], context: str = "") -> str:
    """
    Build an MCQ prompt in the exact format used during fine-tuning:

        ### Question:
        [optional context block]
        Question: {question}

        Options:
        A) ...
        B) ...

        ### Answer:
    """
    opts_str = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(options))
    q_block = f"Question: {question}\n\nOptions:\n{opts_str}"

    if context and len(context.strip()) > 100:
        # Truncate to avoid exceeding max_seq_length
        ctx_trimmed = context[:3000]
        q_block = f"Relevant information:\n{ctx_trimmed}\n\n" + q_block

    return f"### Question:\n{q_block}\n\n### Answer:\n"


def build_open_prompt(question: str, context: str = "") -> str:
    """
    Build an open-ended prompt for non-MCQ queries (used by the agent).

        ### Question:
        [optional context]
        {question}

        ### Answer:
    """
    q_block = question
    if context and len(context.strip()) > 100:
        ctx_trimmed = context[:3000]
        q_block = f"Relevant information:\n{ctx_trimmed}\n\n{question}"

    return f"### Question:\n{q_block}\n\n### Answer:\n"


# ──────────────────────────────────────────────────────────────
# Core generation function
# ──────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    do_sample: bool = False,
    repetition_penalty: float = 1.1,
    lora_repo: Optional[str] = None,
) -> str:
    """
    Generate a text response for the given prompt.

    Args:
        prompt:            Full text prompt (use build_*_prompt helpers above).
        max_new_tokens:    Maximum tokens to generate.
        temperature:       Sampling temperature. Ignored when do_sample=False.
        do_sample:         Greedy decoding by default (deterministic, faster).
        repetition_penalty: Penalises repeated tokens.
        lora_repo:         Optional HuggingFace LoRA repo to load. If already
                           loaded by the singleton, this is ignored.

    Returns:
        The model's generated text (only the new tokens, decoded).
    """
    model, tokenizer = get_model(lora_repo=lora_repo)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample and temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated tokens (not the prompt)
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────
# Answer extraction helpers
# ──────────────────────────────────────────────────────────────

def extract_letter(response: str, num_options: int = 5) -> Optional[str]:
    """
    Extract the answer letter (A–E) from model output.
    Tries in order:
      1. 'The answer is: X'
      2. First standalone letter A–E in the first 50 chars
    Returns None if nothing found.
    """
    max_letter = chr(64 + num_options)
    text = response.strip()

    # Pattern 1
    m = re.search(r"The answer is:\s*([A-Z])", text, re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        if "A" <= letter <= max_letter:
            return letter

    # Pattern 2
    m = re.search(rf"\b([A-{max_letter}])\b", text[:50], re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


# ──────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prompt = build_mcq_prompt(
        question="What is the maximum number of HARQ processes in NR?",
        options=["8", "16", "32", "4"],
    )
    print("=== Prompt ===")
    print(prompt)

    response = generate(prompt, max_new_tokens=20, lora_repo=None)
    print("\n=== Response ===")
    print(response)

    letter = extract_letter(response, num_options=4)
    print(f"\nExtracted letter: {letter}")
