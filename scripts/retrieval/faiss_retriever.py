import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

class FAISSRetriever:
    def __init__(self, chunks_path: str, embeddings_path: str, embedder_model: str = "BAAI/bge-large-en-v1.5"):
        # Load chunks (each line is a dict with keys: text, source, section, ...)
        with open(chunks_path, "r") as f:
            self.chunks = [json.loads(line) for line in f]
        
        # Load pre‑computed embeddings (numpy array)
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        
        # Build FAISS index (Inner Product = cosine if vectors are normalized)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        faiss.normalize_L2(self.embeddings)          # normalize for cosine similarity
        self.index.add(self.embeddings)
        
        # Load the same embedder to encode queries
        self.embedder = SentenceTransformer(embedder_model)
        
    def search(self, query: str, top_k: int = 10):
        # Encode query and normalize
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        query_vec = np.array(query_emb).astype(np.float32).reshape(1, -1)
        
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                chunk = self.chunks[idx]
                results.append({
                    "text": chunk["text"],
                    "source": chunk.get("source", "Unknown"),
                    "section": chunk.get("section", ""),
                    "score": float(score),
                })
        return results