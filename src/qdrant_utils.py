from qdrant_client import QdrantClient

from src.config import QDRANT_PATH, QDRANT_URL


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client for server mode when QDRANT_URL is set, else local mode."""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=str(QDRANT_PATH))
