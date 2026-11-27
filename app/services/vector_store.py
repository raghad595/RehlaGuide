"""Single source of truth for the Qdrant client and the embedding model.

Used by the API (rag_service), data/scripts/ingest_vectors.py and
data/scripts/ingest_wikidata_template.py, so every collection is embedded the same way.

Qdrant *local* mode (a folder on disk) allows only ONE open client per folder, across
all processes. Inside a process, always go through get_qdrant_client(); stop the API
server before running an ingestion script.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import List, Optional, Sequence

from app.core.config import get_settings

# e5 models are trained with these prefixes; queries and documents MUST use them.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_client = None
_client_lock = threading.Lock()
# Serializes Qdrant + embedding calls coming from FastAPI's thread pool.
search_lock = threading.RLock()


def get_qdrant_client():
    global _client
    with _client_lock:
        if _client is None:
            from qdrant_client import QdrantClient

            path = get_settings().qdrant_path
            path.mkdir(parents=True, exist_ok=True)
            _client = QdrantClient(path=str(path))
        return _client


def close_qdrant_client() -> None:
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embed_model, device="cpu")


def embed_passages(texts: Sequence[str], batch_size: int = 16, show_progress: bool = False):
    """Embed documents (adds the 'passage: ' prefix). Returns an (n, dim) numpy array."""
    return get_embedder().encode(
        [PASSAGE_PREFIX + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )


def embed_query(text: str) -> List[float]:
    """Embed a user query (adds the 'query: ' prefix)."""
    return get_embedder().encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()


def collection_point_count(name: str) -> int:
    """Number of points in a collection, or 0 if it does not exist."""
    client = get_qdrant_client()
    if not client.collection_exists(name):
        return 0
    return client.count(name, exact=True).count


def embedding_dim() -> Optional[int]:
    return get_embedder().get_sentence_embedding_dimension()
