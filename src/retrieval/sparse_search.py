import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Add root to python path if run directly
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import QDRANT_PATH, COLLECTION_NAME

class SparseSearcher:
    def __init__(self, client: QdrantClient = None):
        if client:
            self.client = client
        else:
            self.client = QdrantClient(path=str(QDRANT_PATH))
        
    def search(self, sparse_vector: dict, limit: int = 20, spec_filter: str = None) -> list:
        """Searches Qdrant using the sparse BM25 vector."""
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
            query=qdrant_sparse_vector(sparse_vector),
            using="sparse",
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

def qdrant_sparse_vector(sparse_dict: dict):
    from qdrant_client.models import SparseVector
    return SparseVector(
        indices=sparse_dict["indices"],
        values=sparse_dict["values"]
    )

if __name__ == "__main__":
    from src.ingestion.embedder import TelecomEmbedder
    embedder = TelecomEmbedder()
    _, sparse_vec = embedder.embed_query("HARQ process MAC layer")
    
    searcher = SparseSearcher()
    res = searcher.search(sparse_vec, limit=3)
    
    print("\n=== Sparse Search Results ===")
    for r in res:
        print(f"[{r['score']:.4f}] {r['payload'].get('spec_number')} §{r['payload'].get('clause_string')}")
        print(f"Preview: {r['payload'].get('content')[:100]}...\n")
