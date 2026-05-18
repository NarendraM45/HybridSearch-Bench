"""
retrieval.py — BM25, dense-vector, and hybrid (RRF) retrieval strategies.

All three strategies return the same list-of-dict schema so the Streamlit UI
can treat them uniformly:

    {
      "text":      str,
      "metadata":  dict,          # chunk_id, page, source
      "score":     float,         # strategy-specific raw score
      "rrf_score": float | None,  # only filled for hybrid
      "rank":      int,           # 1-based rank within this strategy
      "retriever": str,           # "bm25" | "vector" | "hybrid"
    }
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Union

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, RRF_K, TOP_K
from pipeline import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)


# ── Tokeniser ─────────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "of", "and", "or",
    "to", "for", "with", "by", "from", "that", "this", "are", "was",
    "be", "as", "we", "our", "can", "has", "have", "but", "not",
}


def tokenise(text: str) -> List[str]:
    """Lower-case, strip punctuation, remove stop-words, split on whitespace."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in text.split() if t and t not in _STOP_WORDS]


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25_index(chunks: List[dict]) -> BM25Okapi:
    """Build an Okapi BM25 index from the full chunk list."""
    corpus = [tokenise(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    chunks: List[dict],
    top_k: int = TOP_K,
) -> List[dict]:
    """Return top-k chunks ranked by BM25 score."""
    q_tokens = tokenise(query)
    scores = bm25.get_scores(q_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results: List[dict] = []
    for rank, idx in enumerate(top_indices):
        results.append(
            {
                "text": chunks[idx]["text"],
                "metadata": chunks[idx]["metadata"],
                "score": float(scores[idx]),
                "rrf_score": None,
                "rank": rank + 1,
                "retriever": "bm25",
            }
        )

    logger.debug("BM25 top scores: %s", [r["score"] for r in results])
    return results


# ── Dense Vector Search ───────────────────────────────────────────────────────

def vector_search(
    query_or_embedding: Union[str, List[float]],
    store,
    top_k: int = TOP_K,
) -> List[dict]:
    """
    Dense vector search via ChromaStore.

    Accepts either a raw query string (embedded on the fly) or a pre-computed
    embedding (e.g. from HyDE in HybridSearchPipeline).
    """
    if isinstance(query_or_embedding, str):
        encoder = SentenceTransformerEmbedder()
        q_emb = encoder.embed([query_or_embedding])[0]
    else:
        q_emb = query_or_embedding

    from settings import get_settings
    s = get_settings()

    if s.parent_child_enabled:
        res = store.query_children(embedding=q_emb, top_k=top_k)
    else:
        res = store.query(embedding=q_emb, top_k=top_k)

    results: List[dict] = []
    for r in res:
        results.append(
            {
                "text": r.document.text,
                "metadata": r.document.metadata,
                "score": r.score,
                "rrf_score": None,
                "rank": r.rank,
                "retriever": "vector",
            }
        )

    logger.debug("Vector top scores: %s", [r["score"] for r in results])
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: List[dict],
    vector_results: List[dict],
    top_k: int = TOP_K,
    k: int = RRF_K,
) -> List[dict]:
    """
    Combine BM25 and vector ranked lists with RRF.

        RRF(d) = Σ  1 / (k + rank_r(d))
                r ∈ retrievers

    The constant k=60 is the Cormack et al. (2009) default.
    Chunks that appear in only one list still get a partial score.
    """
    rrf_scores: Dict[str, float] = {}
    doc_store: Dict[str, dict] = {}

    for result_list in (bm25_results, vector_results):
        for doc in result_list:
            cid = doc["metadata"]["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + doc["rank"])
            if cid not in doc_store:
                doc_store[cid] = doc

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results: List[dict] = []
    for rank, cid in enumerate(sorted_ids[:top_k]):
        base = doc_store[cid].copy()
        base["rrf_score"] = round(rrf_scores[cid], 6)
        base["score"] = round(rrf_scores[cid], 6)   # expose as "score" for uniformity
        base["rank"] = rank + 1
        base["retriever"] = "hybrid"
        results.append(base)

    logger.debug("Hybrid top RRF scores: %s", [r["rrf_score"] for r in results])
    return results


# ── Combined Runner ───────────────────────────────────────────────────────────

def run_all_strategies(
    query: str,
    bm25: BM25Okapi,
    chunks: List[dict],
    collection,
    top_k: int = TOP_K,
) -> Dict[str, List[dict]]:
    """
    Execute BM25, vector, and hybrid retrieval for a single query.

    Returns a dict keyed by strategy name so callers can iterate uniformly.
    """
    bm25_res = bm25_search(query, bm25, chunks, top_k)
    vec_res = vector_search(query, collection, top_k)
    hyb_res = reciprocal_rank_fusion(bm25_res, vec_res, top_k)

    return {"bm25": bm25_res, "vector": vec_res, "hybrid": hyb_res}


# ── Score Normalisation Helper ────────────────────────────────────────────────

def normalise_scores(results: List[dict]) -> List[dict]:
    """Min-max normalise 'score' to [0, 1] for cross-strategy comparison charts."""
    scores = [r["score"] for r in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo if hi > lo else 1.0
    for r in results:
        r["norm_score"] = (r["score"] - lo) / span
    return results
