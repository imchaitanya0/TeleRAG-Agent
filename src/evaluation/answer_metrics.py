"""
src/evaluation/answer_metrics.py

Answer quality metrics for the full RAG pipeline:
  - Exact Match (EM): does the answer letter match?
  - Accuracy: % of questions answered correctly
  - Abstention Rate: % of questions where model says "I don't know"

Works on MCQ questions from TeleQnA golden dataset.

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
    Handles formats: 'The answer is: A', 'A)', 'A.', 'A'
    """
    max_letter = chr(64 + num_options)
    text = text.strip()

    patterns = [
        r"[Tt]he\s+answer\s+is[:\s]+([A-E])",
        r"^([A-E])[)\.\s]",
        r"\b([A-E])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text[:100])
        if m:
            letter = m.group(1).upper()
            if "A" <= letter <= max_letter:
                return letter
    return None


def evaluate_answers(
    questions: list[dict],
    use_rag: bool = True,
    lora_repo: Optional[str] = None,
    max_new_tokens: int = 50,
    verbose: bool = False,
) -> dict:
    """
    Evaluate answer accuracy on MCQ questions.

    Args:
        questions: list of dicts with keys:
            - "question": str
            - "options": list[str]  (A, B, C, D, E)
            - "answer": str  (correct letter, e.g. "A")
            - "gold_specs": list[str]  (optional, for context)
        use_rag:     If True, retrieve context before generating
        lora_repo:   HuggingFace LoRA adapter repo ID. None = base model.
        max_new_tokens: Max tokens to generate per answer
        verbose:     Print per-question results

    Returns:
        {
            "accuracy":        float,
            "exact_match":     float,
            "abstention_rate": float,  # % where model output no letter
            "avg_latency_ms":  float,
            "n_questions":     int,
            "n_correct":       int,
            "n_abstained":     int,
        }
    """
    from src.models.inference import build_mcq_prompt, generate

    if use_rag:
        from src.retrieval.fusion import HybridRetriever
        from src.retrieval.reranker import Reranker
        from src.retrieval.context_assembler import ContextAssembler
        retriever = HybridRetriever()
        reranker = Reranker()
        assembler = ContextAssembler()

    n_correct = 0
    n_abstained = 0
    latencies = []

    for i, q in enumerate(questions):
        question = q["question"]
        options = q.get("options", [])
        gold = q.get("answer", "").strip().upper()
        if not gold:
            continue

        t0 = time.perf_counter()

        # Get context if RAG enabled
        context = ""
        if use_rag:
            try:
                candidates = retriever.search(question, top_k=10)
                reranked = reranker.rerank(question, candidates, top_k=3, threshold=0.0)
                context = assembler.assemble(reranked)
            except Exception:
                context = ""

        # Build prompt and generate
        prompt = build_mcq_prompt(question=question, options=options, context=context)
        raw = generate(prompt, max_new_tokens=max_new_tokens, lora_repo=lora_repo)

        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        predicted = _extract_letter(raw, num_options=len(options) or 5)

        if predicted is None:
            n_abstained += 1
            correct = False
        else:
            correct = (predicted == gold)
            if correct:
                n_correct += 1

        if verbose:
            status = "✅" if correct else ("⬜" if predicted is None else "❌")
            print(f"  Q{i+1:3d}: {status} gold={gold} pred={predicted or '?'} "
                  f"({latency:.0f}ms) | {question[:55]}")

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
    }
