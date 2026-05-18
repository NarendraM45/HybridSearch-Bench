"""
hybridsearch/config/settings.py
================================
Single source of truth for every tunable parameter.

Uses ``pydantic-settings`` (v2) so values can come from:
  1. A ``.env`` file   (dev convenience)
  2. Environment variables  (CI / Docker / k8s secrets)
  3. Programmatic override  (tests)

Validation rules are explicit — bad config surfaces at startup, not mid-run.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Enums for validated string choices ───────────────────────────────────────

class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SENTENCE  = "sentence"
    SEMANTIC  = "semantic"      # groups sentences by embedding similarity


class LogFormat(str, Enum):
    JSON = "json"
    TEXT = "text"


# ── Settings model ────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    All configuration for HybridSearch Bench.

    Every field has a sensible default so the app runs locally with
    only ``ANTHROPIC_API_KEY`` set.  Production deployments override via
    environment variables or a secrets manager.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",              # ignore unknown env vars rather than error
    )

    # ── Ollama Config ─────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    ollama_embed_model: str = Field(default="nomic-embed-text")
    max_answer_tokens: int = Field(default=512, ge=64, le=4096)

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name or HuggingFace path.",
    )
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Inference device for sentence-transformers.",
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunker_type: str = Field(default="recursive", description="Type of chunker: 'recursive' or 'semantic'")
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0)
    min_chunk_length: int = Field(default=60, ge=0)
    semantic_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold used by SemanticChunker to merge sentences.",
    )
    semantic_breakpoint_percentile: int = Field(default=95, ge=0, le=100)
    semantic_min_chunk_size: int = Field(default=100, ge=0)

    parent_child_enabled: bool = Field(default=True)
    child_chunk_size: int = Field(default=128, ge=16)
    parent_chunk_size: int = Field(default=512, ge=64)
    child_chunk_overlap: int = Field(default=16, ge=0)
    parent_chunk_overlap: int = Field(default=64, ge=0)

    # ── Vector Store (ChromaDB) ───────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        validation_alias="CHROMA_PERSIST_DIR",
    )
    collection_name: str = Field(
        default="hybrid_bench",
        validation_alias="CHROMA_COLLECTION_NAME",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(default=5, ge=1, le=50)
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF constant (k=60 per Cormack et al. 2009).  "
                    "Larger → flatter score distribution; smaller → amplifies rank gaps.",
    )
    
    hyde_enabled: bool = Field(default=True)
    hyde_blend_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    hyde_fallback_on_error: bool = Field(default=True)
    
    colbert_enabled: bool = Field(default=False)
    colbert_index_path: str = Field(default="./colbert_index")

    # ── Reranking ─────────────────────────────────────────────────────────────
    reranker_enabled: bool = Field(default=True)
    reranker_model: str = Field(default="BAAI/bge-reranker-large")
    reranker_top_k: int = Field(default=5, ge=1)
    reranker_threshold: float = Field(default=0.3)

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_backend: Literal["ollama", "ragas"] = Field(
        default="ollama",
        description="ollama = direct numeric prompts (recommended for local Ollama). "
                    "ragas = RAGAS library (fragile with llama3 JSON output).",
        validation_alias="EVAL_BACKEND",
    )
    eval_timeout: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Per-metric Ollama call timeout (seconds).",
        validation_alias="EVAL_TIMEOUT",
    )
    eval_bootstrap_n: int = Field(
        default=1000,
        ge=100,
        description="Bootstrap resamples for 95 % CI on aggregate IR metrics.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: LogFormat = Field(default=LogFormat.JSON)

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_lt_size(cls, v: int, info: object) -> int:
        # Pydantic v2: info.data contains already-validated fields
        data = getattr(info, "data", {})
        chunk_size = data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be strictly less than chunk_size ({chunk_size})."
            )
        return v

    @model_validator(mode="after")
    def api_key_warning(self) -> "Settings":
        # No API key needed for Ollama, but we can check if base URL is set
        if not self.ollama_base_url:
            import warnings
            warnings.warn(
                "OLLAMA_BASE_URL is not set. Using default http://localhost:11434.",
                stacklevel=2,
            )
        return self


# ── Singleton accessor (cached for the process lifetime) ─────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the global Settings singleton.

    Cached so that `.env` is read once and all modules share the same object.
    In tests, call ``get_settings.cache_clear()`` before constructing a new
    ``Settings`` with different values.
    """
    return Settings()
