"""
src/agent/nodes/retrieve.py

RETRIEVE node: executes multi-source retrieval for all sub-queries,
fuses the results, and assembles the final context window.

Uses HybridRetriever (dense + sparse + KG) → Reranker → ContextAssembler.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.agent.state import AgentState
from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.context_assembler import ContextAssembler
from src.config import TOP_K, RERANK_TOP_K

# ──────────────────────────────────────────────────────────────
# Lazy singletons — loaded once, reused across all retrieve calls
# ──────────────────────────────────────────────────────────────
_retriever: Optional[HybridRetriever] = None
_reranker: Optional[Reranker] = None
_assembler: Optional[ContextAssembler] = None


def _get_retrieval_components():
    global _retriever, _reranker, _assembler
    if _retriever is None:
        _retriever = HybridRetriever()
        _reranker = Reranker()
        _assembler = ContextAssembler()
    return _retriever, _reranker, _assembler


def retrieve_node(state: AgentState) -> AgentState:
    """
    RETRIEVE node: runs retrieval for each sub-query and fuses results.

    Strategy:
    - Run hybrid search (dense + sparse + KG RRF) for each sub-query.
    - Pool all candidates and re-rank jointly using the original query.
    - Assemble context using parent-child expansion within token budget.
    """
    retriever, reranker, assembler = _get_retrieval_components()

    original_query = state["query"]
    sub_queries = state.get("sub_queries", [original_query])

    # ── Multi-query retrieval ──────────────────────────────────
    all_candidates: dict[str, dict] = {}  # chunk_id → candidate (dedup by chunk_id)

    for sq in sub_queries:
        candidates = retriever.search(sq, top_k=TOP_K)
        for cand in candidates:
            cid = cand["chunk_id"]
            if cid not in all_candidates:
                all_candidates[cid] = cand

    pooled = list(all_candidates.values())

    # ── Rerank on the original query ───────────────────────────
    reranked = reranker.rerank(
        original_query,
        pooled,
        top_k=RERANK_TOP_K,
        threshold=0.0,
    )

    # ── Assemble context ───────────────────────────────────────
    context = assembler.assemble(reranked)

    # ── Build citation list ────────────────────────────────────
    sources = []
    citations = []
    for r in reranked:
        payload = r.get("payload", {})
        spec = payload.get("spec_number", "Unknown")
        clause = payload.get("clause_string", "")
        title = payload.get("clause_title", "")
        score = round(r.get("rerank_score", 0.0), 4)
        sources.append({"spec": spec, "clause": clause, "title": title, "score": score})
        cite = f"{spec} §{clause}".strip()
        if cite not in citations:
            citations.append(cite)

    return {
        **state,
        "retrieved_context": context,
        "sources": sources,
        "citations": citations,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
