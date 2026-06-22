import sys
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sentence_transformers import CrossEncoder

_cross_encoder_model = None

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        global _cross_encoder_model
        if _cross_encoder_model is None:
            print(f"Loading CrossEncoder re-ranker: {model_name}")
            _cross_encoder_model = CrossEncoder(model_name)
        self.model = _cross_encoder_model

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """
        Re-ranks a list of candidate chunks based on the query using a CrossEncoder.
        Candidates is a list of dicts containing 'chunk_id' and 'payload'.
        Returns the top_k candidates that score above the threshold.
        """
        if not candidates:
            return []
            
        # Format input for cross encoder: [(query, chunk_text), (query, chunk_text), ...]
        # Truncate document text to ~500 tokens (2000 chars) to prevent OOM
        # bge-reranker-v2-m3 has a max sequence length of 512 tokens
        MAX_DOC_CHARS = 2000
        pairs = []
        for cand in candidates:
            text = cand["payload"].get("content", "")
            if len(text) > MAX_DOC_CHARS:
                text = text[:MAX_DOC_CHARS]
            pairs.append([query, text])
            
        # Get scores
        scores = self.model.predict(pairs)
        
        # Attach scores to candidates
        scored_candidates = []
        for i, cand in enumerate(candidates):
            score = float(scores[i])
            if score >= threshold:
                cand_copy = cand.copy()
                cand_copy["rerank_score"] = score
                scored_candidates.append(cand_copy)
                
        # Sort descending by rerank score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return scored_candidates[:top_k]

if __name__ == "__main__":
    # Simple test
    from src.retrieval.fusion import HybridRetriever
    retriever = HybridRetriever()
    query = "What is RRC connection reconfiguration?"
    
    print(f"Query: {query}")
    print("Running Hybrid Search...")
    candidates = retriever.search(query, top_k=10)
    
    print(f"Found {len(candidates)} candidates via Hybrid Search.")
    
    reranker = Reranker()
    print("Re-ranking...")
    final_results = reranker.rerank(query, candidates, top_k=3)
    
    print("\n=== Re-ranked Results ===")
    for i, r in enumerate(final_results):
        print(f"Rank {i+1} [Score: {r.get('rerank_score', 0):.4f}]: {r['payload'].get('spec_number')} §{r['payload'].get('clause_string')}")
        print(f"Preview: {r['payload'].get('content')[:100]}...\n")
