"""
src/evaluation/ablation.py

4-experiment ablation study to quantify the contribution of each component.

Experiments:
  1. Full System       — dense + sparse + KG + reranker + LoRA fine-tuning
  2. No Re-ranker      — skip cross-encoder, use raw RRF ranking
  3. Sparse (BM25) Only — disable dense + KG retrieval
  4. No Fine-tuning    — base model only (no LoRA adapter)

Each experiment runs on the same set of questions and reports:
  - Accuracy (MCQ exact match)
  - MRR@10 (retrieval quality)
  - Avg latency

Usage:
    from src.evaluation.ablation import run_ablation
    results = run_ablation(questions, n=50)
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import LORA_REPO
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.evaluation.answer_metrics import evaluate_answers


EXPERIMENTS = [
    {
        "name":        "Full System",
        "description": "Dense + Sparse + KG + Reranker + LoRA fine-tuning",
        "use_reranker": True,
        "lora_repo":    LORA_REPO,
        "sparse_only":  False,
    },
    {
        "name":        "No Re-ranker",
        "description": "Dense + Sparse + KG, skip cross-encoder reranking",
        "use_reranker": False,
        "lora_repo":    LORA_REPO,
        "sparse_only":  False,
    },
    {
        "name":        "Sparse (BM25) Only",
        "description": "BM25 sparse retrieval only, no dense or KG",
        "use_reranker": True,
        "lora_repo":    LORA_REPO,
        "sparse_only":  True,
    },
    {
        "name":        "No Fine-tuning",
        "description": "Full retrieval pipeline but base LLM (no LoRA adapter)",
        "use_reranker": True,
        "lora_repo":    None,       # base model only
        "sparse_only":  False,
    },
]


def _patch_retriever_sparse_only():
    """Monkey-patch HybridRetriever.search to use only BM25."""
    from src.retrieval import fusion as fusion_mod
    OrigRetriever = fusion_mod.HybridRetriever

    class SparseOnlyRetriever(OrigRetriever):
        def search(self, query, top_k=20):
            _, sparse_vec = self.embedder.embed_query(query)
            sparse_results = self.sparse_searcher.search(sparse_vec, limit=top_k)
            payload_map = {r["chunk_id"]: r["payload"] for r in sparse_results}
            sparse_ids = [r["chunk_id"] for r in sparse_results]
            return [
                {"chunk_id": cid, "payload": payload_map[cid]}
                for cid in sparse_ids[:top_k]
            ]

    fusion_mod.HybridRetriever = SparseOnlyRetriever
    return OrigRetriever, fusion_mod


def run_ablation(
    questions: list[dict],
    n: int = 50,
    verbose: bool = True,
    save_path: Optional[Path] = None,
) -> list[dict]:
    """
    Run all 4 ablation experiments and return a results table.

    Args:
        questions:  List of MCQ dicts (from golden dataset)
        n:          Number of questions to use (first n)
        verbose:    Print progress
        save_path:  If given, save results JSON here

    Returns:
        List of dicts, one per experiment.
    """
    subset = questions[:n]
    results = []

    # ── Pre-load the LLM ONCE ──────────────────────────────────────────
    # The 8B model takes ~90s to load. Loading it once and reusing
    # across all 4 experiments saves ~270 seconds total.
    # We pre-warm with the LoRA adapter; the "No Fine-tuning" experiment
    # passes lora_repo=None so the inference wrapper skips the adapter.
    if verbose:
        print("\n  Pre-loading LLM (done once for all experiments)...")
    try:
        from src.models.loader import get_model_and_tokenizer
        _model, _tokenizer = get_model_and_tokenizer(lora_repo=None)  # base model
        if verbose:
            print("  ✅ LLM pre-loaded (base model, no LoRA)")
    except Exception as e:
        if verbose:
            print(f"  ⚠ Could not pre-load LLM: {e}")

    import src.retrieval.fusion as fusion_mod

    for exp in EXPERIMENTS:
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Running: {exp['name']}")
            print(f"  {exp['description']}")
            print(f"  n={len(subset)} questions")

        import src.retrieval.fusion as fusion_mod
        orig_retriever = None

        try:
            # Patch retriever for sparse-only experiment
            if exp["sparse_only"]:
                orig_retriever, fusion_mod = _patch_retriever_sparse_only()

            t0 = time.time()

            # Retrieval metrics (no LLM needed)
            ret_metrics = evaluate_retrieval(
                subset,
                top_k=10,
                rerank_k=5,
                use_reranker=exp["use_reranker"],
                verbose=False,
            )

            # Answer accuracy metrics (uses pre-loaded LLM)
            ans_metrics = evaluate_answers(
                subset,
                use_rag=True,
                lora_repo=exp["lora_repo"],
                max_new_tokens=100,
                verbose=verbose,
            )

            elapsed = time.time() - t0

            row = {
                "name":            exp["name"],
                "description":     exp["description"],
                # Use exact_match as PRIMARY metric: denominator=n (includes abstentions)
                # This prevents inflated accuracy when model abstains on hard questions.
                "accuracy":        ans_metrics["exact_match"],   # primary for table
                "accuracy_answered": ans_metrics["accuracy"],    # secondary (excl. abstentions)
                "exact_match":     ans_metrics["exact_match"],
                "abstention_rate": ans_metrics["abstention_rate"],
                "mrr_at_10":       ret_metrics["mrr_at_k"],
                "recall_at_5":     ret_metrics["recall_at_5"],
                "recall_at_1":     ret_metrics["recall_at_1"],
                "avg_latency_ms":  ans_metrics["avg_latency_ms"],
                "total_time_min":  round(elapsed / 60, 1),
                "n_questions":     len(subset),
                "n_correct":       ans_metrics["n_correct"],
                "n_abstained":     ans_metrics["n_abstained"],
            }
            results.append(row)

            if verbose:
                print(f"\n  ✅ Results:")
                print(f"     Exact Match:  {row['exact_match']*100:.1f}%  "
                      f"({row['n_correct']}/{len(subset)} total)")
                print(f"     Accuracy*:    {row['accuracy_answered']*100:.1f}%  "
                      f"({row['n_correct']}/{len(subset)-row['n_abstained']} answered)")
                print(f"     Abstentions:  {row['abstention_rate']*100:.1f}%")
                print(f"     MRR@10:       {row['mrr_at_10']:.4f}")
                print(f"     Recall@5:     {row['recall_at_5']*100:.1f}%")
                print(f"     Latency:      {row['avg_latency_ms']:.0f}ms/query")

        finally:
            # Restore original retriever
            if orig_retriever is not None:
                fusion_mod.HybridRetriever = orig_retriever

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\n  💾 Results saved to {save_path}")

    return results


def print_ablation_table(results: list[dict]):
    """Print a nicely formatted ablation table."""
    print(f"\n{'═'*82}")
    print(f"  {'Experiment':<22} {'Accuracy':>9} {'MRR@10':>8} {'Recall@5':>10} {'Latency':>10}")
    print(f"{'─'*82}")
    for r in results:
        print(f"  {r['name']:<22} {r['accuracy']*100:>8.1f}% {r['mrr_at_10']:>8.4f} "
              f"{r['recall_at_5']*100:>9.1f}% {r['avg_latency_ms']:>8.0f}ms")
    print(f"{'═'*82}")
