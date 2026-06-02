from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.config import QDRANT_PATH, COLLECTION_NAME

class QdrantIndexer:
    def __init__(self):
        # Using local persistent storage instead of Docker
        print(f"Connecting to local Qdrant at: {QDRANT_PATH}")
        self.client = QdrantClient(path=QDRANT_PATH)
        self.collection_name = COLLECTION_NAME
        self.setup_collection()

    def setup_collection(self):
        """Creates the collection with hybrid vector configuration if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            print(f"Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": rest.VectorParams(
                        size=1024, # bge-large-en-v1.5 dim
                        distance=rest.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": rest.SparseVectorParams()
                }
            )
            
            # Create payload indexes for faster filtering
            print("Creating payload indexes...")
            for field in ["metadata.spec_number", "metadata.clause_string", "chunk_tier", "metadata.content_type"]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=rest.PayloadSchemaType.KEYWORD
                )
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def index_chunks(self, embedded_chunks: List[Dict[str, Any]]):
        """Upserts embedded chunks into Qdrant."""
        if not embedded_chunks:
            print("No chunks to index.")
            return

        print(f"Indexing {len(embedded_chunks)} chunks to Qdrant...")
        
        points = []
        for i, chunk in enumerate(embedded_chunks):
            # Qdrant requires UUIDs or integers for point IDs
            # We'll hash the chunk_id to create a deterministic integer ID
            point_id = hash(chunk["chunk_id"]) % ((1 << 63) - 1)
            
            payload = chunk.copy()
            # Remove vectors from payload
            dense_vec = payload.pop("dense_vector")
            sparse_vec = payload.pop("sparse_vector")
            
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": rest.SparseVector(
                            indices=sparse_vec["indices"],
                            values=sparse_vec["values"]
                        )
                    },
                    payload=payload
                )
            )
            
        # Bulk upsert (batching handled internally by qdrant_client for smaller lists, 
        # but we can explicitly batch for very large datasets)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            
        print(f"Successfully indexed {len(embedded_chunks)} chunks.")
        
    def get_stats(self):
        return self.client.get_collection(self.collection_name)
