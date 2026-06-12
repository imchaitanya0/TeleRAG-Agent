"""
src/pipeline/rag_pipeline.py

The end-to-end linear RAG pipeline.

Flow:
  query
    -> HybridRetriever (dense + sparse + KG, RRF fusion)
    -> Reranker (cross-encoder top-5)
    -> ContextAssembler (parent expansion, token budget)
    -> Inference (build prompt, generate, extract answer)
    -> dict { answer, raw_response, context, sources, latency_ms }

Usage:
    from src.pipeline.rag_pipeline import answer

    result = answer("What is RRC Connection Reconfiguration?")
    print(result["answer"])
    print(result["sources"])
"""

import sys
import time
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.context_assembler import ContextAssembler
from src.models.inference import generate, build_open_prompt
from src.config import RERANK_TOP_K, TOP_K


# ──────────────────────────────────────────────────────────────
# Lazy singletons — loaded on first use
# ──────────────────────────────────────────────────────────────
_retriever: Optional[HybridRetriever] = None
_reranker: Optional[Reranker] = None
_assembler: Optional[ContextAssembler] = None


def _get_components():
    global _retriever, _reranker, _assembler
    if _retriever is None:
        print("[Pipeline] Initializing retrieval components...")
        _retriever = HybridRetriever()
        _reranker = Reranker()
        _assembler = ContextAssembler()
        print("[Pipeline] Components ready.")
    return _retriever, _reranker, _assembler


# ──────────────────────────────────────────────────────────────
# Source formatter
# ──────────────────────────────────────────────────────────────

def _extract_sources(candidates: list) -> list[dict]:
    """Pull spec + clause citation from each reranked candidate."""
    sources = []
    for c in candidates:
        payload = c.get("payload", {})
        sources.append({
            "spec": payload.get("spec_number", "Unknown"),
            "clause": payload.get("clause_string", ""),
            "title": payload.get("clause_title", ""),
            "score": round(c.get("rerank_score", 0.0), 4),
        })
    return sources


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Object-oriented interface to the RAG pipeline.
    Useful when you need to keep the pipeline alive across multiple calls
    (e.g., inside the LangGraph agent).
    """

    def __init__(self, lora_repo: Optional[str] = None):
        self.lora_repo = lora_repo
        self.retriever, self.reranker, self.assembler = _get_components()

    def run(
        self,
        query: str,
        top_k: int = TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        max_new_tokens: int = 300,
    ) -> dict:
        """
        Run the full RAG pipeline for an open-ended query.

        Returns:
            {
                "answer":       str,   # model's generated response
                "raw_response": str,   # unprocessed model output
                "context":      str,   # assembled context string
                "sources":      list,  # [{spec, clause, title, score}, ...]
                "latency_ms":   float, # total wall-clock time in ms
            }
        """
        t0 = time.perf_counter()

        # 1. Retrieve
        candidates = self.retriever.search(query, top_k=top_k)

        # 2. Rerank
        reranked = self.reranker.rerank(
            query, candidates, top_k=rerank_top_k, threshold=0.0
        )

        # 3. Assemble context
        context = self.assembler.assemble(reranked)

        # 4. Build prompt and generate
        prompt = build_open_prompt(question=query, context=context)
        raw_response = generate(
            prompt,
            max_new_tokens=max_new_tokens,
            lora_repo=self.lora_repo,
        )

        # 5. Extract sources
        sources = _extract_sources(reranked)

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "answer": raw_response,
            "raw_response": raw_response,
            "context": context,
            "sources": sources,
            "latency_ms": round(latency_ms, 1),
        }


# ──────────────────────────────────────────────────────────────
# Module-level convenience function
# ──────────────────────────────────────────────────────────────

def answer(
    query: str,
    lora_repo: Optional[str] = None,
    max_new_tokens: int = 300,
) -> dict:
    """
    One-shot convenience wrapper around RAGPipeline.

    Example:
        from src.pipeline.rag_pipeline import answer
        result = answer("What is the DRX inactivity timer?")
        print(result["answer"])
    """
    pipeline = RAGPipeline(lora_repo=lora_repo)
    return pipeline.run(query, max_new_tokens=max_new_tokens)


# ──────────────────────────────────────────────────────────────
# Quick smoke test — run: python src/pipeline/rag_pipeline.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    queries = [
        "What is RRC Connection Reconfiguration in 5G NR?",
        "Explain DRX operation in LTE.",
        "What is the role of AMF in 5G core network?",
    ]

    pipeline = RAGPipeline(lora_repo=None)  # base model, no LoRA

    for q in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        result = pipeline.run(q, max_new_tokens=150)
        print(f"ANSWER ({result['latency_ms']:.0f}ms):\n{result['answer']}")
        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  - {s['spec']} §{s['clause']}  [{s['score']}]  {s['title']}")
