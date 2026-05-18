"""
hybridsearch/ingestion/chunker.py
==================================
Three ``Chunker`` implementations behind a factory function.

Strategy          When to use
---------         -----------
RecursiveChunker  Baseline — reliable, fast, language-agnostic.
SentenceChunker   When sentence integrity matters (Q&A, summarisation).
SemanticChunker   When topic coherence matters; groups by embedding similarity.
                  Slower (requires an Embedder) but produces denser, richer chunks.

All three satisfy the ``Chunker`` Protocol from ``core.interfaces``.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Any

import nltk
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

from settings import ChunkStrategy, get_settings
from exceptions import ChunkingError
from interfaces import Document

# Download punkt tokeniser once (no-op if already cached)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_id(text: str) -> str:
    """Stable 12-char content hash used as chunk_id."""
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:12]


def _make_doc(text: str, base_meta: dict[str, Any], index: int) -> Document:
    """Construct a Document from a chunk string with provenance metadata."""
    text = text.strip()
    meta = {**base_meta, "chunk_id": _make_id(text), "chunk_index": index}
    return Document(text=text, metadata=meta, doc_id=_make_id(text))


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseChunker(ABC):
    """Shared validation logic; concrete classes implement ``_split``."""

    def __init__(self, min_length: int | None = None) -> None:
        settings = get_settings()
        self._min_length = min_length if min_length is not None else settings.min_chunk_length

    @abstractmethod
    def _split(self, text: str) -> list[str]:
        ...

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Document]:
        """Split text and filter sub-minimum chunks. Raises ChunkingError if empty."""
        raw = self._split(text)
        docs = [
            _make_doc(s, metadata, i)
            for i, s in enumerate(raw)
            if len(s.strip()) >= self._min_length
        ]

        # De-duplicate on content hash (repeated paragraphs across pages)
        seen: set[str] = set()
        unique: list[Document] = []
        for d in docs:
            cid = d.metadata["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                unique.append(d)

        if not unique:
            raise ChunkingError(
                "Chunking produced zero valid chunks — check min_chunk_length or PDF content.",
                context={"text_len": len(text), "raw_splits": len(raw)},
            )
        return unique


# ── Strategy 1: Recursive Character Splitter ──────────────────────────────────

class RecursiveChunker(BaseChunker):
    """
    LangChain's RecursiveCharacterTextSplitter with a priority separator list.

    Splits on ``\\n\\n`` → ``\\n`` → ``. `` → `` `` in order, so paragraph
    and sentence boundaries are respected before word-level splitting occurs.
    This is the most reliable baseline across all document types.
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        super().__init__()
        s = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or s.chunk_size,
            chunk_overlap=overlap or s.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def _split(self, text: str) -> list[str]:
        return self._splitter.split_text(text)


# ── Strategy 2: Sentence-aware Chunker ───────────────────────────────────────

class SentenceChunker(BaseChunker):
    """
    Groups NLTK-tokenised sentences into windows of ``sentences_per_chunk``
    with a ``stride`` overlap (in sentences, not characters).

    Guarantees that no chunk cuts mid-sentence — critical for Q&A tasks where
    an answer often spans exactly one sentence.
    """

    def __init__(
        self,
        sentences_per_chunk: int = 5,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self._n = sentences_per_chunk
        self._stride = stride

    def _split(self, text: str) -> list[str]:
        sentences = nltk.sent_tokenize(text)
        chunks: list[str] = []
        step = max(1, self._n - self._stride)
        for start in range(0, len(sentences), step):
            window = sentences[start : start + self._n]
            chunks.append(" ".join(window))
        return chunks


# ── Strategy 3: Semantic Chunker ─────────────────────────────────────────────

class SemanticChunker(BaseChunker):
    """
    Groups sentences into chunks by detecting semantic *breakpoints* — positions
    where the cosine similarity between adjacent sentence embeddings drops below
    ``similarity_threshold``.

    This mirrors the approach in (Chen et al., 2023) and produces topically
    coherent chunks that improve retrieval precision on longer documents.

    Time complexity: O(n) embedding calls + O(n) cosine comparisons,
    where n = number of sentences.  Use an encoder with ONNX or GPU for speed.
    """

    def __init__(
        self,
        embedder: Any,                          # Embedder protocol — injected
        similarity_threshold: float | None = None,
        max_chunk_tokens: int = 512,
    ) -> None:
        super().__init__()
        s = get_settings()
        self._embedder = embedder
        self._threshold = similarity_threshold or s.semantic_similarity_threshold
        self._max_tokens = max_chunk_tokens

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def _split(self, text: str) -> list[str]:
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            return sentences

        embeddings = np.array(self._embedder.embed(sentences), dtype=np.float32)

        # Compute pairwise cosine similarity between consecutive sentences
        similarities = [
            self._cosine(embeddings[i], embeddings[i + 1])
            for i in range(len(sentences) - 1)
        ]

        # Split at positions where similarity drops below threshold (topic shift)
        chunks: list[str] = []
        current: list[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            combined_len = len(" ".join(current + [next_sentence]))
            if sim < self._threshold or combined_len > self._max_tokens:
                chunks.append(" ".join(current))
                current = [next_sentence]
            else:
                current.append(next_sentence)

        if current:
            chunks.append(" ".join(current))

        return chunks


# ── Factory ───────────────────────────────────────────────────────────────────

def build_chunker(strategy: ChunkStrategy | None = None, **kwargs: Any) -> BaseChunker:
    """
    Factory that returns the correct chunker for ``strategy``.

    Parameters
    ----------
    strategy : ChunkStrategy (defaults to settings value)
    **kwargs : Forwarded to the concrete chunker constructor.
               e.g. ``embedder=my_embedder`` is required for SEMANTIC.

    Examples
    --------
    >>> chunker = build_chunker(ChunkStrategy.RECURSIVE, chunk_size=256)
    >>> chunker = build_chunker(ChunkStrategy.SEMANTIC, embedder=my_embedder)
    """
    s = get_settings()
    strategy = strategy or s.chunk_strategy

    match strategy:
        case ChunkStrategy.RECURSIVE:
            return RecursiveChunker(**kwargs)
        case ChunkStrategy.SENTENCE:
            return SentenceChunker(**kwargs)
        case ChunkStrategy.SEMANTIC:
            if "embedder" not in kwargs:
                raise ValueError("SemanticChunker requires 'embedder' kwarg.")
            return SemanticChunker(**kwargs)
        case _:
            raise ValueError(f"Unknown chunk strategy: {strategy!r}")
