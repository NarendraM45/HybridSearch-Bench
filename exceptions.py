"""
hybridsearch/core/exceptions.py
================================
Typed exception hierarchy.

Every public operation raises a subclass of ``HybridSearchError`` so callers
can catch at whatever granularity they need:

    except IngestionError:          # only ingestion failures
    except HybridSearchError:       # any library failure
    except Exception:               # truly unexpected

All exceptions carry a ``context`` dict for structured logging.
"""

from __future__ import annotations

from typing import Any


class HybridSearchError(Exception):
    """Base class for all library exceptions."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self!s}, context={self.context})"


# ── Ingestion ──────────────────────────────────────────────────────────────────

class IngestionError(HybridSearchError):
    """Raised when the PDF ingestion pipeline fails."""


class PDFLoadError(IngestionError):
    """Could not load or parse the PDF file."""


class ChunkingError(IngestionError):
    """Text splitting produced zero or invalid chunks."""


class EmbeddingError(IngestionError):
    """Embedding model returned unexpected output."""


# ── Store ──────────────────────────────────────────────────────────────────────

class StoreError(HybridSearchError):
    """Raised on vector-store read/write failures."""


class CollectionNotFoundError(StoreError):
    """The requested ChromaDB collection does not exist."""


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievalError(HybridSearchError):
    """Raised when a retrieval strategy cannot complete."""


class IndexNotBuiltError(RetrievalError):
    """BM25 (or other in-memory) index has not been built yet."""


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvaluationError(HybridSearchError):
    """Raised when RAGAS or custom metric computation fails."""


class AnswerGenerationError(EvaluationError):
    """LLM call for answer generation failed."""


class RAGASError(EvaluationError):
    """RAGAS evaluate() call failed or returned unexpected schema."""


# ── Configuration ────────────────────────────────────────────────────────────

class ConfigurationError(HybridSearchError):
    """Missing or invalid configuration value."""
