"""
hybridsearch/core/interfaces.py
================================
Runtime-checkable Protocols and immutable dataclasses that form the
*only* coupling between modules.  No concrete class appears here.

Design rationale
----------------
Protocols (PEP 544) are preferred over ABCs because:
  - They enable structural subtyping (duck typing with static checks).
  - Third-party retrievers/embedders can conform without inheriting from us.
  - mypy / pyright validate conformance at type-check time.

All domain objects are ``dataclasses(frozen=True, slots=True)``:
  - ``frozen`` → safe to hash and cache; prevents accidental mutation.
  - ``slots``  → ~25 % lower memory overhead; faster attribute access.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable


# ── Domain Objects ────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Document:
    """
    Atomic unit of text in the retrieval corpus.

    Attributes
    ----------
    text       : Raw chunk text.
    metadata   : Arbitrary key-value provenance (page, source, chunk_id …).
    doc_id     : Stable UUID assigned at ingestion time.
    embedding  : Pre-computed dense vector; ``None`` until embed step.
    """

    text: str
    metadata: dict[str, Any]
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: tuple[float, ...] | None = None

    def with_embedding(self, vec: Sequence[float]) -> "Document":
        """Return a new Document with the embedding set (immutability preserved)."""
        return Document(
            text=self.text,
            metadata=self.metadata,
            doc_id=self.doc_id,
            embedding=tuple(vec),
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """
    A single retrieved document together with its score and provenance.

    ``score`` is normalised to [0, 1] where possible:
      - BM25    : raw Okapi BM25 score (not bounded; interpret relatively)
      - Vector  : cosine similarity ∈ [0, 1]
      - Hybrid  : RRF score ∈ (0, 1/k] per retriever; summed across lists
    """

    document: Document
    score: float
    rank: int                   # 1-based
    retriever: str              # "bm25" | "vector" | "hybrid"
    rrf_score: float | None = None


@dataclass(frozen=True, slots=True)
class EvalSample:
    """
    One row in an evaluation dataset.

    ``ground_truth`` is optional; if absent, NDCG / Recall cannot be computed
    and RAGAS context_precision will be skipped.
    """

    query: str
    ground_truth: str | None = None
    relevant_doc_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class StrategyEvalResult:
    """Aggregated evaluation result for a single (query, strategy) pair."""

    strategy: str
    query: str
    answer: str
    contexts: tuple[str, ...]
    ragas_scores: dict[str, float]
    ir_scores: dict[str, float]


# ── Protocols (structural interfaces) ────────────────────────────────────────

@runtime_checkable
class Embedder(Protocol):
    """Converts text(s) into dense float vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of texts. Returns vectors in the same order."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single text."""
        ...

    @property
    def dimension(self) -> int:
        """Dimensionality of the output vectors."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a raw text string into a sequence of Documents."""

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Document]:
        """
        Parameters
        ----------
        text     : Full document text (e.g. one PDF page or the whole document).
        metadata : Provenance info (source filename, page number, …).

        Returns
        -------
        List of Documents with populated ``text`` and ``metadata`` fields.
        ``embedding`` is always ``None`` at this stage.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Persistent storage and retrieval of embedded documents."""

    def upsert(self, documents: list[Document]) -> None:
        """Insert or overwrite documents (keyed on ``doc_id``)."""
        ...

    def query(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """Return top-k nearest neighbours by cosine similarity."""
        ...

    def count(self) -> int:
        """Return the number of documents in the store."""
        ...

    def reset(self) -> None:
        """Delete all documents (idempotent)."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieves the most relevant documents for a free-text query."""

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """
        Parameters
        ----------
        query  : Natural-language query string.
        top_k  : Maximum number of results to return.

        Returns
        -------
        Results sorted by descending relevance score (rank=1 is best).
        """
        ...
