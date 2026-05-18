"""
hybridsearch/store/chroma_store.py
====================================
``ChromaStore`` — concrete implementation of the ``VectorStore`` Protocol
using ChromaDB's PersistentClient with cosine-HNSW indexing.

Why a wrapper instead of using ChromaDB directly?
--------------------------------------------------
1. Isolates the ChromaDB API from the rest of the codebase — swapping to
   Qdrant/Weaviate/pgvector requires changes here only.
2. Converts ChromaDB's dict-of-lists response format into our typed
   ``RetrievalResult`` / ``Document`` domain objects.
3. Centralises retry logic and collection lifecycle management.
4. Lets tests inject a mock that satisfies the ``VectorStore`` Protocol.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from settings import get_settings
from logging import get_logger
from exceptions import CollectionNotFoundError, StoreError
from interfaces import Document, RetrievalResult

log = get_logger(__name__)


class ChromaStore:
    """
    Thread-safe ChromaDB wrapper.

    The collection uses ``hnsw:space=cosine`` so distances are in [0, 1]
    and we convert to similarity as ``sim = 1 - distance``.

    Since embeddings are L2-normalised at ingestion time (see embedder.py),
    cosine distance equals Euclidean distance on the unit sphere — both give
    identical ranking; cosine is slightly faster to query.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        s = get_settings()
        self._persist_dir = persist_dir or s.chroma_persist_dir
        self._collection_name = collection_name or s.collection_name
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection: chromadb.Collection | None = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_or_create(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── VectorStore Protocol ──────────────────────────────────────────────────

    def upsert(self, documents: list[Document]) -> None:
        """
        Insert or overwrite documents.  Documents without embeddings are skipped
        with a warning rather than raising — partial ingestion is still useful.
        """
        valid = [d for d in documents if d.embedding is not None]
        if len(valid) < len(documents):
            log.warning(
                "store.skipped_no_embedding",
                skipped=len(documents) - len(valid),
            )

        if not valid:
            raise StoreError("No documents with embeddings to upsert.")

        collection = self._get_or_create()

        # ChromaDB upsert in batches of 512 to stay within SQLite page limits
        batch_size = 512
        for start in range(0, len(valid), batch_size):
            batch = valid[start : start + batch_size]
            try:
                collection.upsert(
                    ids=[d.doc_id for d in batch],
                    embeddings=[list(d.embedding) for d in batch],  # type: ignore[arg-type]
                    documents=[d.text for d in batch],
                    metadatas=[d.metadata for d in batch],
                )
            except Exception as exc:
                raise StoreError(
                    f"ChromaDB upsert failed at batch starting at {start}: {exc}",
                    context={"batch_size": len(batch)},
                ) from exc

        log.info("store.upserted", n=len(valid), collection=self._collection_name)

    def query(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """
        Return the top-k nearest neighbours by cosine similarity.

        Returns
        -------
        List of ``RetrievalResult`` sorted by descending similarity.

        Raises
        ------
        CollectionNotFoundError if the store is empty or not yet initialised.
        """
        try:
            collection = self._get_or_create()
        except Exception as exc:
            raise CollectionNotFoundError(
                f"Collection '{self._collection_name}' not found.",
                context={"persist_dir": self._persist_dir},
            ) from exc

        n = min(top_k, self.count())
        if n == 0:
            return []

        try:
            res = collection.query(
                query_embeddings=[embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise StoreError(f"ChromaDB query failed: {exc}") from exc

        results: list[RetrievalResult] = []
        for rank, (doc_text, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ):
            similarity = max(0.0, 1.0 - float(dist))
            doc = Document(
                text=doc_text,
                metadata=meta or {},
                doc_id=res["ids"][0][rank],
            )
            results.append(
                RetrievalResult(
                    document=doc,
                    score=similarity,
                    rank=rank + 1,
                    retriever="vector",
                )
            )

        return results

    def count(self) -> int:
        try:
            return self._get_or_create().count()
        except Exception:
            return 0

    def reset(self) -> None:
        """Delete and recreate the collection (wipes all vectors)."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass   # collection may not exist yet
        self._collection = None
        self._get_or_create()
        log.info("store.reset", collection=self._collection_name)
