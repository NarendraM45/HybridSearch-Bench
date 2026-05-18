import logging
from threading import Lock
from typing import Any, ClassVar, Dict, List
import torch

# Delay heavy import
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a Cross-Encoder (e.g., BAAI/bge-reranker-large).
    
    Architecture: cross-encoder jointly encodes (query, passage) pairs
    via full self-attention — unlike bi-encoders which encode independently.
    This is slower but far more precise for near-duplicate documents.
    """
    
    _MODEL_CACHE: ClassVar[dict[tuple[str, str], Any]] = {}
    _CACHE_LOCK: ClassVar[Lock] = Lock()

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "auto",
        top_k: int = 5
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

    @classmethod
    def clear_model_cache(cls) -> None:
        with cls._CACHE_LOCK:
            cls._MODEL_CACHE.clear()

    @classmethod
    def _load_shared_model(cls, model_name: str, device: str):
        cache_key = (model_name, device)
        with cls._CACHE_LOCK:
            cached_model = cls._MODEL_CACHE.get(cache_key)
            if cached_model is not None:
                return cached_model

            model = CrossEncoder(model_name, device=device)
            cls._MODEL_CACHE[cache_key] = model
            return model
            
    def _load_model(self):
        if self._model is not None:
            return
        
        if CrossEncoder is None:
            logger.warning("sentence_transformers not installed. Reranking disabled.")
            return

        try:
            self._model = self._load_shared_model(self.model_name, self.device)
        except Exception as e:
            logger.error(f"Failed to load primary reranker model {self.model_name}: {e}")
            if "large" in self.model_name:
                fallback = "BAAI/bge-reranker-base"
                logger.info(f"Falling back to {fallback}")
                try:
                    self._model = self._load_shared_model(fallback, self.device)
                    self.model_name = fallback
                except Exception as fb_err:
                    logger.error(f"Fallback model failed too: {fb_err}")
            else:
                logger.error("No fallback available.")

    def rerank(self, query: str, chunks: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Takes query + list of chunk dicts (each with 'text' and 'score' keys),
        returns top_k reranked chunks with new 'rerank_score' key added.
        Original retrieval score preserved as 'retrieval_score'.
        """
        if not chunks:
            return []
            
        self._load_model()
        if self._model is None:
            # Silently pass through if model failed to load
            return chunks[:top_k or self.top_k]

        pairs = [[query, c["text"]] for c in chunks]
        
        try:
            scores = self._model.predict(pairs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("CUDA OOM during reranking. Falling back to CPU.")
                self.device = "cpu"
                self._model = self._load_shared_model(self.model_name, "cpu")
                scores = self._model.predict(pairs)
            else:
                raise e

        # Attach scores and sort
        reranked = []
        for c, score in zip(chunks, scores):
            new_c = dict(c)
            new_c["retrieval_score"] = new_c.get("score")
            new_c["rerank_score"] = float(score)
            new_c["score"] = float(score)  # Use rerank score as primary score for uniformity
            reranked.append(new_c)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k or self.top_k]
    
    def rerank_with_threshold(self, query: str, chunks: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """Filter chunks below threshold score — handles no-relevant-doc case."""
        reranked = self.rerank(query, chunks, top_k=len(chunks))
        return [c for c in reranked if c.get("rerank_score", -999.0) >= threshold]
