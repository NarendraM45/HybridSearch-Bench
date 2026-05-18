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
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


from dataclasses import dataclass

@dataclass
class ParentChildChunk:
    child_id: str
    parent_id: str
    child_text: str
    parent_text: str
    source: str
    metadata: dict

# ── Strategy 3: Semantic Chunker ─────────────────────────────────────────────

class SemanticChunker(BaseChunker):
    """
    Groups sentences into chunks by detecting semantic *breakpoints* — positions
    where the cosine similarity between adjacent sentence embeddings drops below
    the breakpoint_threshold_percentile (default 95th).
    """

    def __init__(
        self,
        embedder: Any,
        breakpoint_percentile: int | None = None,
        min_chunk_size: int | None = None,
        max_chunk_size: int = 1000,
    ) -> None:
        super().__init__()
        s = get_settings()
        self._embedder = embedder
        self._percentile = breakpoint_percentile or s.semantic_breakpoint_percentile
        self._min_chunk_size = min_chunk_size or s.semantic_min_chunk_size
        self._max_chunk_size = max_chunk_size
        
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def _split(self, text: str) -> list[str]:
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if len(sentences) <= 1:
            return sentences

        embeddings = np.array(self._embedder.embed(sentences), dtype=np.float32)

        similarities = [
            self._cosine(embeddings[i], embeddings[i + 1])
            for i in range(len(sentences) - 1)
        ]

        threshold = np.percentile(similarities, 100 - self._percentile)

        chunks: list[str] = []
        current: list[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            combined_len = len(" ".join(current + [next_sentence]))
            if sim < threshold or combined_len > self._max_chunk_size:
                chunks.append(" ".join(current))
                current = [next_sentence]
            else:
                current.append(next_sentence)

        if current:
            chunks.append(" ".join(current))
            
        # Merge small chunks
        merged = []
        for c in chunks:
            if not merged:
                merged.append(c)
            elif len(c) < self._min_chunk_size:
                merged[-1] += " " + c
            else:
                merged.append(c)

        return merged


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
    strategy_str = strategy.value if isinstance(strategy, ChunkStrategy) else strategy
    strategy_str = strategy_str or getattr(s, "chunker_type", "recursive")

    if strategy_str == "recursive" or strategy_str == ChunkStrategy.RECURSIVE.value:
        return RecursiveChunker(**kwargs)
    elif strategy_str == "sentence" or strategy_str == ChunkStrategy.SENTENCE.value:
        return SentenceChunker(**kwargs)
    elif strategy_str == "semantic" or strategy_str == ChunkStrategy.SEMANTIC.value:
        if "embedder" not in kwargs:
            raise ValueError("SemanticChunker requires 'embedder' kwarg.")
        return SemanticChunker(**kwargs)
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy_str!r}")
