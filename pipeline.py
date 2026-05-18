"""
hybridsearch/ingestion/pipeline.py
====================================
Orchestrates the full ingestion flow:
  PDF bytes → pages → chunks → embeddings → ChromaDB + BM25 index

The pipeline is a plain function (not a class) because it has no state of its
own — all state is written into the vector store and returned as a chunk list.

Progress callbacks
------------------
The ``progress_cb`` parameter accepts any callable with signature
``(fraction: float, message: str) -> None``.  This keeps the pipeline
decoupled from Streamlit, Click, or any other UI layer.

Error handling
--------------
Every step raises a typed subclass of ``HybridSearchError`` so the calling
layer (CLI or Streamlit) can display a meaningful message and, if needed,
clean up the partial state.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from typing import Any

from langchain_community.document_loaders import PyPDFLoader

from settings import ChunkStrategy, get_settings
from logger_config import get_logger
from exceptions import ChunkingError, IngestionError, PDFLoadError
from interfaces import Document
from chunker import build_chunker, ParentChildChunk
from embedder import SentenceTransformerEmbedder
from chroma_store import ChromaStore
from reranker import CrossEncoderReranker
from hyde import HyDEQueryExpander
from colbert_retriever import ColBERTRetriever
import time

log = get_logger(__name__)

ProgressCb = Callable[[float, str], None]

_NOOP_CB: ProgressCb = lambda pct, msg: None   # noqa: E731


def _load_pdf(pdf_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    """
    Write bytes to a tempfile, load with PyPDFLoader, return page dicts.

    Returns
    -------
    List of ``{"text": str, "page": int, "source": str}``.

    Raises
    ------
    PDFLoadError if the file cannot be parsed or yields zero pages.
    """
    suffix = ".pdf"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    except Exception as exc:
        raise PDFLoadError(
            f"Failed to load PDF '{filename}': {exc}",
            context={"filename": filename, "byte_size": len(pdf_bytes)},
        ) from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not pages:
        raise PDFLoadError(
            f"PDF '{filename}' yielded zero pages — possibly encrypted or image-only.",
            context={"filename": filename},
        )

    result = [
        {
            "text": p.page_content,
            "page": p.metadata.get("page", i),
            "source": filename,
        }
        for i, p in enumerate(pages)
        if p.page_content.strip()
    ]

    log.info("pipeline.pdf_loaded", filename=filename, n_pages=len(result))
    return result


def _chunk_pages(
    pages: list[dict[str, Any]],
    strategy: str | ChunkStrategy,
    embedder: SentenceTransformerEmbedder | None,
) -> list[Document]:
    """
    Apply the selected chunking strategy to all pages.

    ``embedder`` is required only for ``semantic``.
    """
    kwargs: dict[str, Any] = {}
    strategy_str = strategy.value if hasattr(strategy, "value") else strategy
    if strategy_str == "semantic":
        if embedder is None:
            raise IngestionError("SemanticChunker requires an embedder.")
        kwargs["embedder"] = embedder

    chunker = build_chunker(strategy, **kwargs)

    all_chunks: list[Document] = []
    for page in pages:
        try:
            chunks = chunker.chunk(
                text=page["text"],
                metadata={"page": page["page"], "source": page["source"]},
            )
            all_chunks.extend(chunks)
        except ChunkingError:
            log.warning(
                "pipeline.empty_page",
                page=page["page"],
                source=page["source"],
            )
            continue   # skip pages that produce nothing (e.g. image-only pages)

    # Global de-duplication across all pages
    seen: set[str] = set()
    unique: list[Document] = []
    for d in all_chunks:
        cid = d.metadata["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            unique.append(d)

    log.info(
        "pipeline.chunks_created",
        total=len(all_chunks),
        unique=len(unique),
        strategy=strategy_str,
    )

    if not unique:
        raise ChunkingError("Pipeline produced zero chunks after de-duplication.")

    # Handle parent-child if enabled and not semantic
    settings = get_settings()
    if settings.parent_child_enabled and strategy_str != "semantic":
        parent_chunker = build_chunker(strategy, chunk_size=settings.parent_chunk_size, overlap=settings.parent_chunk_overlap)
        child_chunker = build_chunker(strategy, chunk_size=settings.child_chunk_size, overlap=settings.child_chunk_overlap)
        
        parent_child_chunks = []
        # Chunk pages into parents
        for page in pages:
            parents = parent_chunker.chunk(page["text"], {"page": page["page"], "source": page["source"]})
            for p in parents:
                children = child_chunker.chunk(p.text, p.metadata)
                for c in children:
                    child_meta = {**c.metadata, "parent_id": p.doc_id}
                    parent_child_chunks.append(ParentChildChunk(
                        child_id=c.doc_id,
                        parent_id=p.doc_id,
                        child_text=c.text,
                        parent_text=p.text,
                        source=p.metadata.get("source", ""),
                        metadata=child_meta,
                    ))
        
        return parent_child_chunks

    return unique

def _embed_parent_child(
    chunks: list[ParentChildChunk],
    embedder: SentenceTransformerEmbedder,
) -> list[Document]:
    texts = [c.child_text for c in chunks]
    vectors = embedder.embed(texts)
    docs = []
    for c, vec in zip(chunks, vectors):
        d = Document(text=c.child_text, metadata=c.metadata, doc_id=c.child_id)
        docs.append(d.with_embedding(vec))
    return docs


def _embed_documents(
    documents: list[Document],
    embedder: SentenceTransformerEmbedder,
) -> list[Document]:
    """Embed all documents and return new Document instances with embeddings set."""
    texts = [d.text for d in documents]
    vectors = embedder.embed(texts)   # raises EmbeddingError on failure

    return [doc.with_embedding(vec) for doc, vec in zip(documents, vectors)]


# ── Public entry point ────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_bytes: bytes,
    filename: str = "document.pdf",
    chunk_strategy: ChunkStrategy | None = None,
    progress_cb: ProgressCb = _NOOP_CB,
) -> tuple[list[Document], ChromaStore]:
    """
    Run the complete ingestion pipeline.

    Parameters
    ----------
    pdf_bytes      : Raw bytes of the PDF file.
    filename       : Display name used in metadata and logs.
    chunk_strategy : Override the ``chunk_strategy`` setting.
    progress_cb    : Called with (fraction ∈ [0,1], message) at each stage.

    Returns
    -------
    documents : List of embedded ``Document`` objects (used to build BM25 index).
    store     : Populated ``ChromaStore`` ready for vector queries.

    Raises
    ------
    PDFLoadError, ChunkingError, EmbeddingError, StoreError
    """
    settings = get_settings()
    strategy = chunk_strategy or getattr(settings, "chunker_type", "recursive")
    
    # ensure string for strategy logging
    strategy_val = strategy.value if hasattr(strategy, "value") else strategy
    log.info("pipeline.start", filename=filename, strategy=strategy_val)

    progress_cb(0.05, "Loading PDF pages…")
    pages = _load_pdf(pdf_bytes, filename)

    progress_cb(0.20, f"Loaded {len(pages)} pages — building chunks…")
    embedder = SentenceTransformerEmbedder()
    documents = _chunk_pages(pages, strategy, embedder)

    progress_cb(0.45, f"{len(documents)} chunks created — computing embeddings…")
    
    if isinstance(documents[0], ParentChildChunk):
        embedded_docs = _embed_parent_child(documents, embedder)
    else:
        embedded_docs = _embed_documents(documents, embedder)

    progress_cb(0.75, "Storing vectors in ChromaDB…")
    store = ChromaStore()
    
    if isinstance(documents[0], ParentChildChunk):
        store.upsert_parent_child(documents, embedded_docs)
    else:
        store.upsert(embedded_docs)
        
    if settings.colbert_enabled:
        colbert = ColBERTRetriever(settings.colbert_index_path)
        if isinstance(documents[0], ParentChildChunk):
            colbert.index([c.parent_text for c in documents], [c.parent_id for c in documents])
        else:
            colbert.index([d.text for d in documents], [d.doc_id for d in documents])

    progress_cb(1.00, f"Ingestion complete — {len(embedded_docs)} chunks indexed ✓")
    log.info("pipeline.complete", n_chunks=len(embedded_docs))

    return embedded_docs, store

class HybridSearchPipeline:
    def __init__(self, store: ChromaStore, bm25_index=None, chunks=None):
        self.store = store
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.settings = get_settings()
        
        self.embedder = SentenceTransformerEmbedder()
        self.llm = None
        
        if self.settings.hyde_enabled:
            try:
                import ollama
                _ollama_settings = self.settings

                class OllamaWrapper:
                    def invoke(self, messages):
                        client = ollama.Client(host=_ollama_settings.ollama_base_url)
                        prompt = messages[0].content
                        res = client.chat(
                            model=_ollama_settings.ollama_model,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        class Res:
                            content = res["message"]["content"]
                        return Res()

                self.llm = OllamaWrapper()
                self.hyde = HyDEQueryExpander(self.llm, self.embedder, blend_alpha=self.settings.hyde_blend_alpha, fallback_on_error=self.settings.hyde_fallback_on_error)
            except Exception as e:
                log.warning(f"Failed to initialize HyDE: {e}. Falling back to standard query.")
                self.settings.hyde_enabled = False
            
        if self.settings.reranker_enabled:
            try:
                self.reranker = CrossEncoderReranker(model_name=self.settings.reranker_model, top_k=self.settings.reranker_top_k)
            except Exception as e:
                log.warning(f"Failed to initialize Reranker: {e}. Disabling reranking.")
                self.settings.reranker_enabled = False
            
        if self.settings.colbert_enabled:
            try:
                self.colbert = ColBERTRetriever(self.settings.colbert_index_path)
            except Exception as e:
                log.warning(f"Failed to initialize ColBERT: {e}. Disabling ColBERT.")
                self.settings.colbert_enabled = False

    def run(self, query: str) -> dict:
        logs = []
        k = self.settings.top_k
        
        # 1. HyDE
        t0 = time.perf_counter()
        query_vector = None
        if self.settings.hyde_enabled:
            expanded_text, query_vector = self.hyde.expand(query)
            logs.append({"stage": "hyde", "latency_ms": int((time.perf_counter()-t0)*1000)})
        else:
            query_vector = self.embedder.embed([query])[0]
            
        # 2. BM25 Retrieval
        t0 = time.perf_counter()
        from retrieval import bm25_search
        bm25_res = bm25_search(query, self.bm25_index, self.chunks, top_k=k*4)
        logs.append({"stage": "bm25_retrieval", "latency_ms": int((time.perf_counter()-t0)*1000), "candidates": len(bm25_res)})
        
        # 3. Vector Retrieval
        t0 = time.perf_counter()
        from retrieval import vector_search
        vec_res = vector_search(query_vector, self.store, top_k=k * 4)
        logs.append({"stage": "vector_retrieval", "latency_ms": int((time.perf_counter()-t0)*1000), "candidates": len(vec_res)})
        
        # 4. RRF Fusion
        t0 = time.perf_counter()
        from retrieval import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(bm25_res, vec_res, top_k=k*2, k=self.settings.rrf_k)
        logs.append({"stage": "rrf_fusion", "latency_ms": int((time.perf_counter()-t0)*1000), "candidates": len(fused)})
        
        # 5. Parent Lookup
        t0 = time.perf_counter()
        if self.settings.parent_child_enabled:
            parent_ids = [doc["metadata"].get("parent_id") for doc in fused if "parent_id" in doc["metadata"]]
            parents = self.store.get_parents(parent_ids)
            parent_map = {p.doc_id: p.text for p in parents}
            for doc in fused:
                pid = doc["metadata"].get("parent_id")
                if pid in parent_map:
                    doc["text"] = parent_map[pid]
            logs.append({"stage": "parent_lookup", "latency_ms": int((time.perf_counter()-t0)*1000), "candidates": len(fused)})
            
        # 6. Reranking
        t0 = time.perf_counter()
        if self.settings.reranker_enabled:
            fused = self.reranker.rerank_with_threshold(query, fused, threshold=self.settings.reranker_threshold)
            fused = fused[:k]
            logs.append({"stage": "reranking", "latency_ms": int((time.perf_counter()-t0)*1000), "candidates": len(fused)})
        else:
            fused = fused[:k]
            
        # 7. ColBERT
        colbert_res = []
        if self.settings.colbert_enabled:
            t0 = time.perf_counter()
            colbert_res = self.colbert.search(query, top_k=k)
            logs.append({"stage": "colbert_retrieval", "latency_ms": int((time.perf_counter()-t0)*1000), "candidates": len(colbert_res)})
            
        return {
            "bm25": bm25_res[:k],
            "vector": vec_res[:k],
            "hybrid": fused,
            "colbert": colbert_res,
            "logs": logs
        }
