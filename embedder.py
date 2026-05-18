"""
embedder.py
====================================
``OllamaEmbedder`` — the sole production ``Embedder`` implementation.

Design choices
--------------
* Uses Ollama backend instead of local SentenceTransformers.
* Fallback to nomic-embed-text as per settings.
* L2 normalisation: Vectors are unit-normalised before storage.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from settings import get_settings
from exceptions import EmbeddingError
from logging import get_logger

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings as LCEmbeddings

log = get_logger(__name__)


class OllamaEmbedder:
    """
    Thread-safe Ollama embedding model.

    Attributes
    ----------
    model_name : str
    batch_size : int
        Number of texts per forward pass.
    normalize  : bool
        L2-normalise output vectors.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> None:
        s = get_settings()
        self._model_name = model_name or s.ollama_embed_model
        self._batch_size = batch_size or s.embedding_batch_size
        self._normalize = normalize
        self._base_url = s.ollama_base_url
        import ollama
        self._client = ollama.Client(host=self._base_url)

    # ── Public API (satisfies Embedder Protocol) ──────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-encode ``texts`` into float32 vectors.
        """
        if not texts:
            return []

        try:
            # We encode sequentially or rely on Ollama concurrency
            # For simplicity, we just loop through texts, but Ollama supports batching in some cases.
            # Using the official ollama python client, we can call embed for multiple texts in some versions,
            # but to be safe we loop if batch isn't supported, or use the `embeddings` endpoint.
            vecs = []
            for text in texts:
                resp = self._client.embeddings(model=self._model_name, prompt=text)
                vec = np.array(resp["embedding"], dtype=np.float32)
                if self._normalize:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                vecs.append(vec.tolist())

        except Exception as exc:
            raise EmbeddingError(
                f"Ollama encoding failed: {exc}",
                context={"model": self._model_name, "n_texts": len(texts)},
            ) from exc

        # Sanity checks
        if len(vecs) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} vectors, got {len(vecs)}.",
                context={"model": self._model_name},
            )

        log.debug("embedder.encoded", n=len(texts), dim=len(vecs[0]) if vecs else 0)
        return vecs

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        # We can dynamically get dimension by embedding a tiny string.
        # nomic-embed-text is 768.
        return len(self.embed_one("test"))

    # ── LangChain compatibility shim ──────────────────────────────────────────

    def as_langchain_embeddings(self) -> "LCEmbeddings":
        """
        Return a LangChain ``Embeddings`` object wrapping this embedder.
        """
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=self._model_name, base_url=self._base_url)

# Keep the old name for compatibility if needed, or update pipeline.py
SentenceTransformerEmbedder = OllamaEmbedder
