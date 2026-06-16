"""
src/agent/tools/spec_retriever.py

Tool: Specification Retriever
Wraps the full hybrid RAG pipeline as a callable agent tool.

The agent invokes this tool when query_type is "spec_qa", "troubleshoot",
"optimization", or "kpi". It runs multi-query retrieval over the Qdrant
vector database (3GPP specs) and returns structured results.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.context_assembler import ContextAssembler
from src.config import TOP_K, RERANK_TOP_K

# ─── lazy singletons ───────────────────────────────────────────
_retriever: Optional[HybridRetriever] = None
_reranker: Optional[Reranker] = None
_assembler: Optional[ContextAssembler] = None


def _init():
    global _retriever, _reranker, _assembler
    if _retriever is None:
        _retriever = HybridRetriever()
        _reranker = Reranker()
        _assembler = ContextAssembler()


def spec_retriever_tool(
    query: str,
    sub_queries: Optional[list[str]] = None,
    top_k: int = TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
) -> dict:
    """
    Retrieve relevant passages from the 3GPP specification knowledge base.

    Args:
        query:       The primary query (used for reranking).
        sub_queries: Optional list of sub-queries for multi-hop retrieval.
                     If None, only the primary query is used.
        top_k:       Candidates to retrieve per sub-query.
        rerank_top_k: Final number of passages after reranking.

    Returns:
        {
            "context":  str,   assembled context string
            "sources":  list,  [{spec, clause, title, score}, ...]
            "citations": list, ["TS 38.331 §5.3", ...]
            "num_passages": int
        }
    """
    _init()

    queries = sub_queries if sub_queries else [query]

    # Multi-query retrieval + deduplication
    seen: dict[str, dict] = {}
    for q in queries:
        for cand in _retriever.search(q, top_k=top_k):
            cid = cand["chunk_id"]
            if cid not in seen:
                seen[cid] = cand

    pooled = list(seen.values())
    if not pooled:
        return {
            "context": "",
            "sources": [],
            "citations": [],
            "num_passages": 0,
        }

    # Rerank on original query
    reranked = _reranker.rerank(query, pooled, top_k=rerank_top_k, threshold=0.0)

    # Assemble context
    context = _assembler.assemble(reranked)

    # Build sources + citations
    sources = []
    citations = []
    for r in reranked:
        payload = r.get("payload", {})
        spec = payload.get("spec_number", "Unknown")
        clause = payload.get("clause_string", "")
        title = payload.get("clause_title", "")
        score = round(r.get("rerank_score", 0.0), 4)
        sources.append({"spec": spec, "clause": clause, "title": title, "score": score})
        cite = f"{spec} §{clause}".strip(" §")
        if cite and cite not in citations:
            citations.append(cite)

    return {
        "context": context,
        "sources": sources,
        "citations": citations,
        "num_passages": len(reranked),
    }


# ── smoke test ────────────────────────────────────────────────
if __name__ == "__main__":
    result = spec_retriever_tool(
        query="What is RRC Connection Reconfiguration?",
        sub_queries=[
            "RRC Connection Reconfiguration 5G NR",
            "TS 38.331 RRC reconfiguration procedure",
        ],
    )
    print(f"Passages: {result['num_passages']}")
    print(f"Citations: {result['citations']}")
    print(f"Context snippet:\n{result['context'][:300]}")
