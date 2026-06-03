import sys
from pathlib import Path
from typing import List, Dict

# Add root to python path if run directly
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.retrieval.dense_search import DenseSearcher
from src.retrieval.sparse_search import SparseSearcher
from src.retrieval.kg_search import KGSearcher
from src.ingestion.embedder import TelecomEmbedder

from qdrant_client import QdrantClient
from src.config import QDRANT_PATH

class HybridRetriever:
    def __init__(self):
        self.qdrant_client = QdrantClient(path=str(QDRANT_PATH))
        self.dense_searcher = DenseSearcher(client=self.qdrant_client)
        self.sparse_searcher = SparseSearcher(client=self.qdrant_client)
        self.kg_searcher = KGSearcher()
        self.embedder = TelecomEmbedder()

    def rrf_fusion(self, ranked_lists: List[List[str]], k: int = 60) -> List[str]:
        """Reciprocal Rank Fusion."""
        rrf_scores = {}
        for ranked_list in ranked_lists:
            for rank, item_id in enumerate(ranked_list):
                if item_id not in rrf_scores:
                    rrf_scores[item_id] = 0.0
                rrf_scores[item_id] += 1.0 / (k + rank + 1)
                
        # Sort by rrf score descending
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items]

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Performs hybrid search and RRF fusion, returning full chunk payloads."""
        
        # 1. Embed query
        dense_vec, sparse_vec = self.embedder.embed_query(query)
        
        # 2. Get dense results (fetch more than we need for better fusion)
        dense_results = self.dense_searcher.search(dense_vec, limit=top_k * 2)
        dense_ids = [res["chunk_id"] for res in dense_results]
        
        # 3. Get sparse results
        sparse_results = self.sparse_searcher.search(sparse_vec, limit=top_k * 2)
        sparse_ids = [res["chunk_id"] for res in sparse_results]
        
        # 4. Get KG results
        kg_results = self.kg_searcher.search(query, max_hops=1)
        # KG search returns section node ids (e.g. "TS 38.331_5.3"). We need to boost chunks belonging to these sections.
        # This is a bit complex without reverse lookup, so for now we just use dense + sparse for RRF, 
        # and we can use KG for context expansion later. For true RRF, we need chunk IDs.
        # But wait, Qdrant payload contains 'spec_number' and 'clause_string'. We can match them!
        kg_boosted_ids = []
        kg_section_keys = {f"{r['data'].get('spec_number')}_{r['data'].get('clause_string')}" for r in kg_results}
        
        # Build a mapping of chunk_id -> payload from dense and sparse results
        payload_map = {}
        for r in dense_results + sparse_results:
            payload_map[r["chunk_id"]] = r["payload"]
            
        for chunk_id, payload in payload_map.items():
            section_key = f"{payload.get('spec_number')}_{payload.get('clause_string')}"
            if section_key in kg_section_keys:
                kg_boosted_ids.append(chunk_id)

        # 5. RRF Fusion
        ranked_lists = [dense_ids, sparse_ids]
        if kg_boosted_ids:
            ranked_lists.append(kg_boosted_ids)
            
        fused_ids = self.rrf_fusion(ranked_lists)
        
        # 6. Return top_k payloads
        final_results = []
        for cid in fused_ids[:top_k]:
            final_results.append({
                "chunk_id": cid,
                "payload": payload_map[cid] # It's guaranteed to be in payload_map since we only fused IDs from dense/sparse
            })
            
        return final_results

if __name__ == "__main__":
    retriever = HybridRetriever()
    res = retriever.search("DRX operation")
    
    print("\n=== Hybrid Search (RRF) Results ===")
    for i, r in enumerate(res[:3]):
        print(f"Rank {i+1}: {r['payload'].get('spec_number')} §{r['payload'].get('clause_string')}")
        print(f"Preview: {r['payload'].get('content')[:100]}...\n")
