"""
config.py — Global configuration for HybridSearch Bench.
Centralises every tunable constant so experiments are reproducible.
"""

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512          # Characters per chunk
CHUNK_OVERLAP: int = 64        # Overlap between consecutive chunks
MIN_CHUNK_LEN: int = 60        # Discard chunks shorter than this

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # ~80 MB, fast, good quality
EMBEDDING_DIM: int = 384

# ── Vector Store (ChromaDB) ───────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = "./chroma_db"
COLLECTION_NAME: str = "hybridsearch_bench"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = 5                 # Number of documents to retrieve per strategy
RRF_K: int = 60                # Reciprocal Rank Fusion constant (standard = 60)

# ── LLM (Answer Generation + RAGAS Evaluation) ───────────────────────────────
OLLAMA_MODEL: str = "llama3.2"
MAX_ANSWER_TOKENS: int = 512

# ── RAGAS ─────────────────────────────────────────────────────────────────────
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]
