"""
evaluation.py — Answer generation + RAGAS metric evaluation using Ollama.

For each retrieval strategy (BM25 / Vector / Hybrid):
  1. Build a context string from retrieved chunks.
  2. Send to Ollama to generate a grounded answer.
  3. Run RAGAS (faithfulness, answer_relevancy, context_precision) using
     LangChain's ChatOllama wrapper.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import ollama
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate as ragas_evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from config import OLLAMA_MODEL, MAX_ANSWER_TOKENS
from settings import get_settings

logger = logging.getLogger(__name__)

# ── Answer Generation ─────────────────────────────────────────────────────────

_ANSWER_SYSTEM = (
    "You are a precise research assistant. "
    "Answer the user's question using ONLY the provided context snippets. "
    "If the answer cannot be found in the context, say 'Not found in context.' "
    "Be concise (≤ 3 sentences unless detail is essential)."
)

_ANSWER_TMPL = """\
Context snippets:
{context}

Question: {question}

Answer:"""


def generate_answer(
    query: str,
    contexts: List[str],
    api_key: str = "",
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Call Ollama to produce a grounded answer from the retrieved contexts.
    api_key is kept for signature compatibility but unused.
    """
    settings = get_settings()
    
    # We use the raw ollama client here for simplicity, although we could use ChatOllama
    client = ollama.Client(host=settings.ollama_base_url)

    context_block = "\n\n---\n\n".join(
        f"[{i+1}] {ctx.strip()}" for i, ctx in enumerate(contexts)
    )

    prompt = _ANSWER_TMPL.format(context=context_block, question=query)

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        options={"num_predict": MAX_ANSWER_TOKENS}
    )

    return response['message']['content'].strip()


# ── RAGAS Evaluation ──────────────────────────────────────────────────────────

def evaluate_single(
    query: str,
    contexts: List[str],
    answer: str,
    api_key: str = "",
    ground_truth: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run RAGAS metrics for a single (query, answer, contexts) triple.

    Returns a dict like:
        {
          "faithfulness": 0.87,
          "answer_relevancy": 0.91,
          "context_precision": 0.75,    # only if ground_truth provided
        }
    """
    settings = get_settings()
    data: Dict[str, list] = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
    }
    if ground_truth:
        data["ground_truth"] = [ground_truth]

    dataset = Dataset.from_dict(data)

    try:
        # Wrap LLM for RAGAS
        lc_llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
        ragas_llm = LangchainLLMWrapper(lc_llm)
        
        # Wrap Embeddings for RAGAS
        lc_embeddings = OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)
        ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

        metrics = [faithfulness, answer_relevancy]
        if ground_truth:
            metrics.append(context_precision)

        # Inject shared LLM + embeddings into every metric
        for m in metrics:
            m.llm = ragas_llm
            if hasattr(m, "embeddings"):
                m.embeddings = ragas_embeddings

        result = ragas_evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False)
        scores = {k: round(float(v), 4) for k, v in result.items() if isinstance(v, float)}
        return scores
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed (Ollama running?): {e}")
        # Fallback dictionary with None for requested metrics
        fallback = {"faithfulness": None, "answer_relevancy": None}
        if ground_truth:
            fallback["context_precision"] = None
        return fallback


# ── Multi-strategy Evaluation Loop ────────────────────────────────────────────

def evaluate_all_strategies(
    query: str,
    retrieval_results: Dict[str, List[dict]],
    api_key: str = "",
    ground_truth: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Dict]:
    """
    For each strategy in retrieval_results:
      - Generate an answer
      - Compute RAGAS scores
      - Return combined dict

    Shape of return value:
        {
          "bm25":   {"answer": "...", "faithfulness": 0.9, ...},
          "vector": {"answer": "...", "faithfulness": 0.7, ...},
          "hybrid": {"answer": "...", "faithfulness": 0.95, ...},
        }
    """
    strategies = list(retrieval_results.keys())
    eval_out: Dict[str, Dict] = {}
    n = len(strategies)

    for i, strategy in enumerate(strategies):
        if progress_callback:
            progress_callback((i) / n, f"Evaluating **{strategy}** strategy…")

        contexts = [r["text"] for r in retrieval_results[strategy]]

        answer = generate_answer(query, contexts, api_key)
        logger.info("[%s] answer: %s", strategy, answer[:80])

        scores = evaluate_single(query, contexts, answer, api_key, ground_truth)
        eval_out[strategy] = {"answer": answer, **scores}

    if progress_callback:
        progress_callback(1.0, "Evaluation complete ✓")

    return eval_out
