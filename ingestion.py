"""
ingestion.py — Full PDF ingestion pipeline.

Flow:
  PDF file → load pages → recursive text split → sentence-transformer embeddings
           → ChromaDB (vector store) + raw chunk list (for BM25)
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, MIN_CHUNK_LEN
from settings import get_settings

logger = logging.getLogger(__name__)

# Shared encoder — loaded once per process
_encoder: SentenceTransformer | None = None


def get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


# ── PDF Loading & Chunking ────────────────────────────────────────────────────

def load_pdf(pdf_bytes: bytes, filename: str = "upload.pdf") -> List[dict]:
    """
    Write PDF bytes to a temp file, load with PyPDFLoader, return raw page dicts.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    finally:
        os.unlink(tmp_path)

    return [
        {"text": p.page_content, "page": p.metadata.get("page", i), "source": filename}
        for i, p in enumerate(pages)
        if p.page_content.strip()
    ]


def chunk_pages(pages: List[dict]) -> List[dict]:
    """
    Split raw page texts into overlapping chunks using RecursiveCharacterTextSplitter.
    Each chunk carries source + page provenance plus a short content hash as ID.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: List[dict] = []
    for page in pages:
        sub_texts = splitter.split_text(page["text"])
        for sub in sub_texts:
            sub = sub.strip()
            if len(sub) < MIN_CHUNK_LEN:
                continue
            chunk_id = hashlib.md5(sub.encode()).hexdigest()[:12]
            chunks.append(
                {
                    "text": sub,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "page": page["page"],
                        "source": page["source"],
                    },
                }
            )

    # De-duplicate on content hash (same paragraph on repeated pages)
    seen: set[str] = set()
    unique: List[dict] = []
    for c in chunks:
        if c["metadata"]["chunk_id"] not in seen:
            seen.add(c["metadata"]["chunk_id"])
            unique.append(c)

    logger.info("Chunking produced %d unique chunks.", len(unique))
    return unique


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(
    chunks: List[dict],
    progress_callback=None,
) -> List[List[float]]:
    """
    Encode chunk texts with the sentence-transformer model.
    Returns list of float vectors in the same order as `chunks`.
    """
    encoder = get_encoder()
    texts = [c["text"] for c in chunks]

    embeddings = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if progress_callback:
        progress_callback(1.0)

    return embeddings.tolist()


# ── ChromaDB Storage ──────────────────────────────────────────────────────────

def build_chroma_collection(
    chunks: List[dict],
    embeddings: List[List[float]],
    reset: bool = True,
) -> chromadb.Collection:
    """
    Persist chunks + pre-computed embeddings into ChromaDB.
    By default wipes the existing collection first (reset=True) so that
    re-ingesting a new PDF starts clean.
    """
    s = get_settings()
    client = chromadb.PersistentClient(path=s.chroma_persist_dir)

    if reset:
        try:
            client.delete_collection(s.collection_name)
        except Exception:
            pass  # Collection didn't exist yet — that's fine

    collection = client.get_or_create_collection(
        name=s.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch-insert (ChromaDB recommends ≤ 5 000 items per call)
    batch_size = 512
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        emb_batch = embeddings[start : start + batch_size]

        collection.add(
            documents=[c["text"] for c in batch],
            embeddings=emb_batch,
            ids=[c["metadata"]["chunk_id"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    logger.info("ChromaDB: stored %d vectors in collection '%s'.", len(chunks), s.collection_name)
    return collection


# ── Top-level Pipeline ────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_bytes: bytes,
    filename: str = "upload.pdf",
    progress_callback=None,
) -> Tuple[List[dict], chromadb.Collection]:
    """
    Full ingestion pipeline.

    Returns
    -------
    chunks      : List of chunk dicts (text + metadata) — needed by BM25
    collection  : ChromaDB collection — used for vector search
    """
    if progress_callback:
        progress_callback(0.05, "Loading PDF pages…")

    pages = load_pdf(pdf_bytes, filename)

    if progress_callback:
        progress_callback(0.20, f"Loaded {len(pages)} pages. Chunking…")

    chunks = chunk_pages(pages)

    if progress_callback:
        progress_callback(0.40, f"Created {len(chunks)} chunks. Embedding…")

    embeddings = embed_chunks(chunks)

    if progress_callback:
        progress_callback(0.75, "Storing in ChromaDB…")

    collection = build_chroma_collection(chunks, embeddings, reset=True)

    if progress_callback:
        progress_callback(1.0, "Ingestion complete ✓")

    return chunks, collection


def load_existing_collection() -> chromadb.Collection:
    """Return the already-persisted ChromaDB collection (raises if absent)."""
    s = get_settings()
    client = chromadb.PersistentClient(path=s.chroma_persist_dir)
    return client.get_collection(s.collection_name)
