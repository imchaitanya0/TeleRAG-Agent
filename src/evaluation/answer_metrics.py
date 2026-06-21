"""
src/evaluation/answer_metrics.py

Answer quality metrics for the full RAG pipeline:
  - Exact Match (EM): does the answer letter match?
  - Accuracy: % of questions answered correctly (excl. abstentions)
  - Abstention Rate: % of questions where model says "I don't know"

Works on MCQ questions from TeleQnA golden dataset.

EVALUATION NOTE:
  This module is used purely for benchmarking the model's knowledge.
  The Gradio app uses the full RAG pipeline with context for real users.
  For MCQ evaluation the model is tested without injected context so we
  measure what it actually knows, not what the retriever can inject.

Usage:
    from src.evaluation.answer_metrics import evaluate_answers
    results = evaluate_answers(questions)
"""

import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def _extract_letter(text: str, num_options: int = 5) -> Optional[str]:
    """
    Extract the answer letter (A-E) from model output.

    Priority order (from most to least reliable):
      1. 'The answer is: X'  or  'The answer is X'
      2. Line starts with 'X)' or 'X.'
      3. Single standalone letter on its own line
    Deliberately avoids matching letters inside explanatory text.
    """
    max_letter = chr(64 + num_options)
    text = text.strip()

    # Pattern 1 (highest confidence): explicit "The answer is: X"
    m = re.search(r"[Tt]he\s+answer\s+is[:\s]+([A-E])", text)
    if m:
        letter = m.group(1).upper()
        if "A" <= letter <= max_letter:
            return letter

    # Pattern 2: line starting with "X)" or "X." (option-style)
    m = re.search(r"(?:^|\n)\s*([A-E])[)\.]", text)
    if m:
        letter = m.group(1).upper()
        if "A" <= letter <= max_letter:
            return letter

    # Pattern 3: single uppercase letter alone on a line (e.g. just "D\n")
    m = re.search(r"(?:^|\n)\s*([A-E])\s*(?:\n|$)", text)
    if m:
        letter = m.group(1).upper()
        if "A" <= letter <= max_letter:
            return letter

    return None


def evaluate_answers(
    questions: list[dict],
    use_rag: bool = False,   # DEFAULT FALSE: evaluate model knowledge, not retrieval
    lora_repo: Optional[str] = None,
    max_new_tokens: int = 30,   # MCQ answers are ~5 tokens; 30 is generous
    verbose: bool = False,
    reranker_threshold: float = 0.15,  # Only inject context if relevant enough
) -> dict:
    """
    Evaluate answer accuracy on MCQ questions.

    Args:
        questions: list of dicts with keys:
            - "question": str
            - "options": list[str]
            - "answer": str  (correct letter, e.g. "A")
            - "input_text": str  (raw training-format input for exact prompt match)
            - "gold_specs": list[str]  (optional)
        use_rag:    If True, retrieve context and inject if score > threshold.
                    Default FALSE — evaluate base model knowledge directly.
        lora_repo:  HuggingFace LoRA adapter repo ID. None = base model.
        max_new_tokens: Max tokens to generate (30 is enough for "The answer is: X")
        verbose:    Print per-question results
        reranker_threshold: Minimum reranker score to inject context (0.0-1.0)

    Returns:
        dict with accuracy, exact_match, abstention_rate, avg_latency_ms, etc.
    """
    from src.models.inference import generate

    if use_rag:
        from src.retrieval.fusion import HybridRetriever
        from src.retrieval.reranker import Reranker
        from src.retrieval.context_assembler import ContextAssembler
        retriever = HybridRetriever()
        reranker = Reranker()
        assembler = ContextAssembler()

    n_correct = 0
    n_abstained = 0
    n_context_used = 0
    latencies = []

    for i, q in enumerate(questions):
        question = q["question"]
        options = q.get("options", [])
        gold = q.get("answer", "").strip().upper()
        input_text = q.get("input_text", "")
        if not gold:
            continue

        t0 = time.perf_counter()

        # ── Context injection (conditional on RAG + reranker score) ──
        context = ""
        if use_rag:
            try:
                candidates = retriever.search(question, top_k=10)
                reranked = reranker.rerank(question, candidates, top_k=3, threshold=0.0)
                # Only use context if the top result is actually relevant
                if reranked and reranked[0].get("rerank_score", 0) >= reranker_threshold:
                    # Keep context short — 800 chars is ~160 tokens, very manageable
                    raw_context = assembler.assemble(reranked)
                    context = raw_context[:800]
                    n_context_used += 1
            except Exception:
                context = ""

        # ── Build prompt exactly matching training format ──
        # Training format: "### Question:\n{input}\n\n### Answer:\n{output}"
        # input_text already contains "Question: ...\n\nOptions:\nA) ..."
        if input_text:
            if context:
                # Prepend retrieved context before the question block
                prompt = f"### Question:\nRelevant context:\n{context}\n\n{input_text}\n\n### Answer:\n"
            else:
                # Clean format — matches training exactly
                prompt = f"### Question:\n{input_text}\n\n### Answer:\n"
        else:
            # Fallback for questions without input_text (synthetic only)
            from src.models.inference import build_mcq_prompt
            prompt = build_mcq_prompt(question=question, options=options, context=context)

        raw = generate(
            prompt,
            max_new_tokens=max_new_tokens,
            lora_repo=lora_repo,
            repetition_penalty=1.0,  # Disabled — penalizing repeated option text causes wrong answers
        )

        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        predicted = _extract_letter(raw, num_options=len(options) if options else 5)

        if predicted is None:
            n_abstained += 1
            correct = False
        else:
            correct = (predicted == gold)
            if correct:
                n_correct += 1

        if verbose:
            status = "✅" if correct else ("⬜" if predicted is None else "❌")
            ctx_marker = "📄" if context else "  "
            print(f"  Q{i+1:3d}: {status}{ctx_marker} gold={gold} pred={predicted or '?'} "
                  f"({latency:.0f}ms) | {question[:50]}")

    n = len(questions)
    answered = n - n_abstained
    return {
        "accuracy":        round(n_correct / answered, 4) if answered else 0.0,
        "exact_match":     round(n_correct / n, 4) if n else 0.0,
        "abstention_rate": round(n_abstained / n, 4) if n else 0.0,
        "avg_latency_ms":  round(sum(latencies) / n, 1) if n else 0.0,
        "n_questions":     n,
        "n_correct":       n_correct,
        "n_abstained":     n_abstained,
        "n_context_used":  n_context_used,
    }
