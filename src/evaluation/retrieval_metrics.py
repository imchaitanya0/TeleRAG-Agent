"""
src/evaluation/retrieval_metrics.py

Retrieval quality metrics: MRR@k, Recall@k, Precision@k.

These run WITHOUT the LLM — only the retrieval + reranking pipeline
is exercised. This lets us measure retrieval quality independently.

Usage:
    from src.evaluation.retrieval_metrics import evaluate_retrieval
    results = evaluate_retrieval(questions, k=10)
"""

import sys
import time
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def _hit_in_sources(sources: list[dict], gold_specs: list[str]) -> bool:
    """Check if any retrieved source matches one of the gold spec numbers."""
    retrieved_specs = {s.get("spec", "").upper() for s in sources}
    for gs in gold_specs:
        if any(gs.upper() in r or r in gs.upper() for r in retrieved_specs):
            return True
    return False


def _reciprocal_rank(sources: list[dict], gold_specs: list[str]) -> float:
    """MRR contribution: 1/rank of first relevant source, 0 if not found."""
    for rank, s in enumerate(sources, 1):
        spec = s.get("spec", "").upper()
        for gs in gold_specs:
            if gs.upper() in spec or spec in gs.upper():
                return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    questions: list[dict],
    top_k: int = 10,
    rerank_k: int = 5,
    use_reranker: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Evaluate retrieval quality on a list of questions.

    Args:
        questions: list of dicts with keys:
            - "question": str
            - "gold_specs": list[str]  (e.g. ["TS 38.331", "38.300"])
            - "answer": str (optional, used for logging)
        top_k:       How many chunks to retrieve
        rerank_k:    How many to keep after reranking
        use_reranker: Whether to run the cross-encoder
        verbose:     Print per-question results

    Returns:
        {
            "mrr_at_k":    float,  # Mean Reciprocal Rank
            "recall_at_1": float,  # % where top-1 is relevant
            "recall_at_5": float,  # % where top-5 contains relevant
            "recall_at_k": float,  # % where top-k contains relevant
            "avg_latency_ms": float,
            "n_questions": int,
        }
    """
    from src.retrieval.fusion import HybridRetriever
    from src.retrieval.reranker import Reranker
    from src.retrieval.context_assembler import ContextAssembler
    from src.pipeline.rag_pipeline import _extract_sources

    retriever = HybridRetriever()
    reranker = Reranker() if use_reranker else None

    mrr_scores = []
    recall_1 = []
    recall_5 = []
    recall_k = []
    latencies = []

    for i, q in enumerate(questions):
        query = q["question"]
        gold_specs = q.get("gold_specs", [])
        t0 = time.perf_counter()

        candidates = retriever.search(query, top_k=top_k)

        if use_reranker and reranker:
            ranked = reranker.rerank(query, candidates, top_k=rerank_k, threshold=0.0)
        else:
            ranked = candidates[:rerank_k]

        sources = _extract_sources(ranked)
        latencies.append((time.perf_counter() - t0) * 1000)

        if gold_specs:
            mrr = _reciprocal_rank(sources, gold_specs)
            hit_1 = _hit_in_sources(sources[:1], gold_specs)
            hit_5 = _hit_in_sources(sources[:5], gold_specs)
            hit_k = _hit_in_sources(sources[:top_k], gold_specs)
        else:
            # No gold labels — just check if any source was retrieved
            mrr = 1.0 if sources else 0.0
            hit_1 = hit_5 = hit_k = bool(sources)

        mrr_scores.append(mrr)
        recall_1.append(float(hit_1))
        recall_5.append(float(hit_5))
        recall_k.append(float(hit_k))

        if verbose:
            print(f"  Q{i+1:3d}: MRR={mrr:.3f} R@1={int(hit_1)} R@5={int(hit_5)} "
                  f"| {query[:60]}")

    n = len(questions)
    return {
        "mrr_at_k":       round(sum(mrr_scores) / n, 4) if n else 0.0,
        "recall_at_1":    round(sum(recall_1) / n, 4) if n else 0.0,
        "recall_at_5":    round(sum(recall_5) / n, 4) if n else 0.0,
        "recall_at_k":    round(sum(recall_k) / n, 4) if n else 0.0,
        "avg_latency_ms": round(sum(latencies) / n, 1) if n else 0.0,
        "n_questions":    n,
    }
