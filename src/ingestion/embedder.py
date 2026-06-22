import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from src.config import EMBED_MODEL_ID

_dense_model = None
_sparse_model = None

class TelecomEmbedder:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        
        global _dense_model, _sparse_model
        
        if _dense_model is None:
            print(f"Loading dense embedding model: {EMBED_MODEL_ID}")
            _dense_model = SentenceTransformer(EMBED_MODEL_ID)
        self.dense_model = _dense_model
        
        if _sparse_model is None:
            print("Loading sparse embedding model: Qdrant/bm25")
            _sparse_model = SparseTextEmbedding("Qdrant/bm25")
        self.sparse_model = _sparse_model
        
        # Instruction prefix required by BGE models for retrieval queries
        # Note: We do NOT use this prefix for indexing documents, only for queries
        self.query_prefix = "Represent this telecom question for retrieval: "

    def embed_documents(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embeds a list of chunk dictionaries with both dense and sparse vectors."""
        if not chunks:
            return []
            
        texts = [chunk["content"] for chunk in chunks]
        
        print(f"Generating dense embeddings for {len(texts)} chunks...")
        dense_embeddings = self.dense_model.encode(
            texts, 
            batch_size=self.batch_size, 
            show_progress_bar=True
        )
        
        print(f"Generating sparse embeddings for {len(texts)} chunks...")
        # fastembed returns a generator of SparseEmbedding objects
        sparse_embeddings = list(self.sparse_model.embed(texts, batch_size=self.batch_size))
        
        # Attach embeddings back to chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy["dense_vector"] = dense_embeddings[i].tolist()
            
            # Sparse embedding object has .indices and .values
            sparse = sparse_embeddings[i]
            chunk_copy["sparse_vector"] = {
                "indices": sparse.indices.tolist(),
                "values": sparse.values.tolist()
            }
            embedded_chunks.append(chunk_copy)
            
        return embedded_chunks

    def embed_query(self, query: str) -> Tuple[List[float], Dict[str, List]]:
        """Embeds a single query string (uses instruction prefix for dense)."""
        prefixed_query = f"{self.query_prefix}{query}"
        
        dense = self.dense_model.encode(prefixed_query).tolist()
        
        sparse_obj = list(self.sparse_model.embed([query]))[0]
        sparse = {
            "indices": sparse_obj.indices.tolist(),
            "values": sparse_obj.values.tolist()
        }
        
        return dense, sparse
