"""
Local Qdrant Indexer — imports Kaggle ingestion outputs.

Usage:
    python scripts/import_kaggle_ingestion.py \
        --chunks data/processed/chunks_with_embeddings.jsonl \
        --graph data/processed/section_graph.pkl
"""
import json
import argparse
import shutil
import numpy as np
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    PayloadSchemaType, SparseVector, SparseVectorParams,
)
from fastembed import SparseTextEmbedding


def import_to_qdrant(
    chunks_path: str,
    qdrant_path: str = "data/qdrant_storage",
    qdrant_url: str | None = None,
    use_sparse: bool = True,
    batch_size: int = 100,
    timeout: int = 120,
):
    """Load chunks with pre-computed embeddings into local Qdrant."""

    if qdrant_url:
        print(f"Using Qdrant server at {qdrant_url}")
    else:
        # Clear old embedded storage.
        qdrant_dir = Path(qdrant_path)
        if qdrant_dir.exists():
            shutil.rmtree(qdrant_dir)
            print(f"Cleared old Qdrant storage at {qdrant_dir}")

    # Read chunks
    print(f"Reading chunks from {chunks_path}...")
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks")

    # Separate embeddings
    embeddings = []
    payloads = []
    for chunk in chunks:
        emb = chunk.pop("embedding", None)
        if emb is None:
            print(f"  WARNING: chunk {chunk['chunk_id']} has no embedding, skipping")
            continue
        embeddings.append(emb)
        payloads.append(chunk)

    embedding_dim = len(embeddings[0])
    print(f"Embedding dimension: {embedding_dim}")

    # Connect to Qdrant
    client = (
        QdrantClient(url=qdrant_url, timeout=timeout, check_compatibility=False)
        if qdrant_url
        else QdrantClient(path=qdrant_path)
    )

    # Create collection. Keep the same named-vector layout used by src/retrieval.
    collection_name = "telecom_specs"
    sparse_vectors_config = {"sparse": SparseVectorParams()} if use_sparse else None
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config=sparse_vectors_config,
    )
    print(f"Created collection '{collection_name}'")

    sparse_model = None
    if use_sparse:
        print("Preparing local BM25 sparse embeddings (Qdrant/bm25)...")
        sparse_model = SparseTextEmbedding("Qdrant/bm25")

    # Index in batches
    total_indexed = 0
    for i in range(0, len(payloads), batch_size):
        batch_payloads = payloads[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_sparse = []
        if sparse_model:
            texts = [payload.get("content", "") for payload in batch_payloads]
            batch_sparse = list(sparse_model.embed(texts, batch_size=64))

        points = []
        for j, (payload, emb) in enumerate(zip(batch_payloads, batch_embeddings)):
            vector = {"dense": emb}
            if batch_sparse:
                sparse = batch_sparse[j]
                vector["sparse"] = SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                )
            points.append(PointStruct(
                id=i + j,
                vector=vector,
                payload=payload,
            ))

        for attempt in range(1, 4):
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                sleep_for = 2 * attempt
                print(f"  Upsert timeout/error at batch starting {i}; retrying in {sleep_for}s ({exc})")
                time.sleep(sleep_for)
        total_indexed += len(points)
        print(f"  Indexed {total_indexed}/{len(payloads)} chunks")

    # Create payload indexes for filtering
    for field in ["spec_number", "chunk_tier", "content_type", "clause_string"]:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # Local Qdrant may not support payload indexes

    print(f"\n=== INDEXING COMPLETE ===")
    print(f"Total chunks indexed: {total_indexed}")
    print(f"Qdrant storage: {qdrant_path}")


def copy_chunks_lite(src_chunks: str, dest_path: str = "data/processed/chunks.jsonl"):
    """Also save a version without embeddings for reference."""
    print(f"\nSaving lightweight chunks to {dest_path}...")
    with open(src_chunks) as fin, open(dest_path, "w") as fout:
        for line in fin:
            chunk = json.loads(line)
            chunk.pop("embedding", None)
            fout.write(json.dumps(chunk) + "\n")
    print("Done.")


def copy_graph_if_needed(src_graph: str, dest_path: str = "data/processed/section_graph.pkl"):
    src = Path(src_graph)
    dest = Path(dest_path)
    if src.resolve() == dest.resolve():
        print(f"KG already at {dest}; skipping copy.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"Copied KG to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Kaggle ingestion outputs to local Qdrant")
    parser.add_argument("--chunks", type=str, required=True, help="Path to chunks_with_embeddings.jsonl")
    parser.add_argument("--graph", type=str, default=None, help="Path to section_graph.pkl (optional, just copy)")
    parser.add_argument("--qdrant", type=str, default="data/qdrant_storage", help="Qdrant storage path")
    parser.add_argument("--qdrant-url", type=str, default=None, help="Qdrant server URL, e.g. http://localhost:6333")
    parser.add_argument("--dense-only", action="store_true", help="Skip local BM25 sparse-vector generation")
    parser.add_argument("--batch-size", type=int, default=100, help="Upsert batch size. Use 50-100 for server Qdrant with sparse vectors")
    parser.add_argument("--timeout", type=int, default=120, help="Qdrant client request timeout in seconds")

    args = parser.parse_args()

    import_to_qdrant(
        args.chunks,
        args.qdrant,
        qdrant_url=args.qdrant_url,
        use_sparse=not args.dense_only,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    copy_chunks_lite(args.chunks)

    if args.graph:
        copy_graph_if_needed(args.graph)

    print("\n✅ All done! Qdrant is now populated with real 3GPP content.")
