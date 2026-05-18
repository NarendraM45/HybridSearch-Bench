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
from logging import get_logger
from exceptions import ChunkingError, IngestionError, PDFLoadError
from interfaces import Document
from chunker import build_chunker
from embedder import SentenceTransformerEmbedder
from chroma_store import ChromaStore

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
    strategy: ChunkStrategy,
    embedder: SentenceTransformerEmbedder | None,
) -> list[Document]:
    """
    Apply the selected chunking strategy to all pages.

    ``embedder`` is required only for ``ChunkStrategy.SEMANTIC``.
    """
    kwargs: dict[str, Any] = {}
    if strategy == ChunkStrategy.SEMANTIC:
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
        strategy=strategy.value,
    )

    if not unique:
        raise ChunkingError("Pipeline produced zero chunks after de-duplication.")

    return unique


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
    strategy = chunk_strategy or settings.chunk_strategy

    log.info("pipeline.start", filename=filename, strategy=strategy.value)

    progress_cb(0.05, "Loading PDF pages…")
    pages = _load_pdf(pdf_bytes, filename)

    progress_cb(0.20, f"Loaded {len(pages)} pages — building chunks…")
    embedder = SentenceTransformerEmbedder()
    documents = _chunk_pages(pages, strategy, embedder)

    progress_cb(0.45, f"{len(documents)} chunks created — computing embeddings…")
    embedded_docs = _embed_documents(documents, embedder)

    progress_cb(0.75, "Storing vectors in ChromaDB…")
    store = ChromaStore()
    store.reset()
    store.upsert(embedded_docs)

    progress_cb(1.00, f"Ingestion complete — {len(embedded_docs)} chunks indexed ✓")
    log.info("pipeline.complete", n_chunks=len(embedded_docs))

    return embedded_docs, store
