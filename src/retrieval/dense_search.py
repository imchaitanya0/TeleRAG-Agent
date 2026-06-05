import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Add root to python path if run directly
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import COLLECTION_NAME
from src.qdrant_utils import get_qdrant_client

class DenseSearcher:
    def __init__(self, client: QdrantClient = None):
        if client:
            self.client = client
        else:
            self.client = get_qdrant_client()
        
    def search(self, dense_vector: list, limit: int = 20, spec_filter: str = None) -> list:
        """Searches Qdrant using the dense vector."""
        query_filter = None
        if spec_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="spec_number",
                        match=MatchValue(value=spec_filter)
                    )
                ]
            )
            
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vector,
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )
        
        results = []
        for hit in response.points:
            results.append({
                "chunk_id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
            
        return results

if __name__ == "__main__":
    from src.ingestion.embedder import TelecomEmbedder
    embedder = TelecomEmbedder()
    dense_vec, _ = embedder.embed_query("What is RRC connection reconfiguration?")
    
    searcher = DenseSearcher()
    res = searcher.search(dense_vec, limit=3)
    
    print("\n=== Dense Search Results ===")
    for r in res:
        print(f"[{r['score']:.4f}] {r['payload'].get('spec_number')} §{r['payload'].get('clause_string')}")
        print(f"Preview: {r['payload'].get('content')[:100]}...\n")
