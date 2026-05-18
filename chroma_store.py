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
from logger_config import get_logger
from exceptions import CollectionNotFoundError, StoreError
from interfaces import Document, RetrievalResult
from chunker import ParentChildChunk

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

    def list_collection_names(self) -> list[str]:
        """Return all persisted collection names (for diagnostics)."""
        return [c.name for c in self._client.list_collections()]

    def _collection_count(self, name: str) -> int:
        try:
            return self._client.get_collection(name).count()
        except Exception:
            return 0

    def _resolve_chunk_collection(self) -> tuple[chromadb.Collection, bool]:
        """
        Pick the collection that holds searchable child/flat chunks.

        Returns (collection, is_parent_child_mode).
        Prefers ``{base}_children`` when it has vectors; otherwise the flat
        ``{base}`` collection.  Supports legacy ``hybridsearch_bench`` name.
        """
        base = self._collection_name
        candidates: list[tuple[str, bool]] = [
            (f"{base}_children", True),
            (base, False),
            ("hybridsearch_bench", False),  # legacy flat ingest via ingestion.py
        ]
        for name, is_pc in candidates:
            if self._collection_count(name) > 0:
                return self._client.get_collection(name), is_pc
        # Nothing indexed yet — honour settings for the next ingest.
        s = get_settings()
        if s.parent_child_enabled:
            return self._client.get_or_create_collection(
                name=f"{base}_children",
                metadata={"hnsw:space": "cosine"},
            ), True
        return self._get_or_create(), False

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

    def upsert_parent_child(self, parent_child_chunks: list[ParentChildChunk], embedded_children: list[Document]) -> None:
        """Upsert into dual _children and _parents collections."""
        if not parent_child_chunks or not embedded_children: return
        
        child_collection = self._client.get_or_create_collection(
            name=f"{self._collection_name}_children",
            metadata={"hnsw:space": "cosine"}
        )
        parent_collection = self._client.get_or_create_collection(
            name=f"{self._collection_name}_parents",
            metadata={"hnsw:space": "cosine"}
        )
        
        batch_size = 512
        
        # Upsert parents (these don't necessarily have embeddings in this implementation, 
        # or we just store them to retrieve text by id)
        # We only need to store text/metadata for parents, no embeddings needed for retrieval
        unique_parents = {}
        for c in parent_child_chunks:
            if c.parent_id not in unique_parents:
                unique_parents[c.parent_id] = {
                    "text": c.parent_text,
                    "metadata": c.metadata,
                }
                
        parent_ids = list(unique_parents.keys())
        for start in range(0, len(parent_ids), batch_size):
            batch_ids = parent_ids[start : start + batch_size]
            parent_collection.upsert(
                ids=batch_ids,
                documents=[unique_parents[pid]["text"] for pid in batch_ids],
                metadatas=[unique_parents[pid]["metadata"] for pid in batch_ids]
            )
            
        # Upsert children (deduplicating to avoid DuplicateIDError)
        unique_children = {}
        for c in embedded_children:
            if c.doc_id not in unique_children:
                unique_children[c.doc_id] = c
                
        unique_child_list = list(unique_children.values())
        child_ids = [c.doc_id for c in unique_child_list]
        parent_id_by_child = {pc.child_id: pc.parent_id for pc in parent_child_chunks}
        child_texts = [c.text for c in unique_child_list]
        child_metadatas = []
        for c in unique_child_list:
            meta = dict(c.metadata)
            pid = parent_id_by_child.get(c.doc_id)
            if pid:
                meta.setdefault("parent_id", pid)
            child_metadatas.append(meta)
        child_embeddings = [list(c.embedding) for c in unique_child_list]
        
        for start in range(0, len(child_ids), batch_size):
            child_collection.upsert(
                ids=child_ids[start : start + batch_size],
                embeddings=child_embeddings[start : start + batch_size],
                documents=child_texts[start : start + batch_size],
                metadatas=child_metadatas[start : start + batch_size]
            )

    def query_children(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """Query the _children collection."""
        try:
            collection = self._client.get_collection(name=f"{self._collection_name}_children")
        except Exception:
            return []
            
        n = min(top_k, collection.count())
        if n == 0: return []
        
        res = collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        
        results = []
        if not res or not res["ids"] or not res["ids"][0]:
            return results
            
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
                    retriever="vector_child",
                )
            )
        return results

    def get_parents(self, parent_ids: list[str]) -> list[Document]:
        """Fetch parent documents directly by their IDs."""
        try:
            collection = self._client.get_collection(name=f"{self._collection_name}_parents")
        except Exception:
            return []
            
        # Deduplicate
        p_ids = list(set(parent_ids))
        if not p_ids: return []
        
        res = collection.get(ids=p_ids, include=["documents", "metadatas"])
        
        docs = []
        if res and res["ids"]:
            for i in range(len(res["ids"])):
                docs.append(Document(
                    text=res["documents"][i],
                    metadata=res["metadatas"][i] or {},
                    doc_id=res["ids"][i]
                ))
        return docs

    def get_all(self) -> list[dict]:
        """Fetch all documents for BM25 building."""
        try:
            collection, _ = self._resolve_chunk_collection()
            res = collection.get(include=["documents", "metadatas"])
            if not res or not res["ids"]:
                return []
                
            chunks = []
            for i in range(len(res["ids"])):
                chunks.append({
                    "text": res["documents"][i],
                    "metadata": res["metadatas"][i] or {},
                })
            return chunks
        except Exception as e:
            log.error(f"get_all failed: {e}")
            return []

    def count(self) -> int:
        """Number of indexed chunk vectors (children or flat collection)."""
        try:
            collection, _ = self._resolve_chunk_collection()
            return collection.count()
        except Exception:
            return 0

    def reset(self) -> None:
        """Delete and recreate all collections for this corpus (wipes all vectors)."""
        base = self._collection_name
        for name in (
            base,
            f"{base}_children",
            f"{base}_parents",
            "hybridsearch_bench",
        ):
            try:
                self._client.delete_collection(name)
            except Exception:
                pass
        self._collection = None
        self._get_or_create()
        log.info("store.reset", collection=self._collection_name)
