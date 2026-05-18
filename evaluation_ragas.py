"""
evaluation_ragas.py — Optional RAGAS backend (EVAL_BACKEND=ragas).

Kept separate because RAGAS + local Ollama often fails JSON parsing for faithfulness.
Use the default ``ollama`` backend in evaluation.py for production.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate as ragas_evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, faithfulness
from ragas.run_config import RunConfig

from settings import get_settings

logger = logging.getLogger(__name__)

_RAGAS_INPUT_COLUMNS = frozenset({
    "question", "answer", "contexts", "ground_truth",
    "user_input", "response", "retrieved_contexts", "reference",
})

_KNOWN_METRIC_COLUMNS = frozenset({
    "faithfulness", "answer_relevancy", "context_precision",
    "context_recall", "context_utilization",
})


def _make_ragas_backends(settings):
    return (
        LangchainLLMWrapper(
            ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0,
                timeout=180.0,
            )
        ),
        LangchainEmbeddingsWrapper(
            OllamaEmbeddings(
                model=settings.ollama_embed_model,
                base_url=settings.ollama_base_url,
            )
        ),
    )


def _scores_for_row(df, row_idx: int) -> Dict[str, float]:
    import pandas as pd

    scores: Dict[str, float] = {}
    for col in df.columns:
        if col in _RAGAS_INPUT_COLUMNS:
            continue
        if col not in _KNOWN_METRIC_COLUMNS and not pd.api.types.is_numeric_dtype(df[col]):
            continue
        val = df[col].iloc[row_idx]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            scores[col] = round(float(val), 4)
        except (TypeError, ValueError):
            continue
    return scores


def evaluate_batch_ragas(
    query: str,
    rows: List[tuple[str, List[str], str]],
    ground_truth: Optional[str],
) -> Dict[str, Dict]:
    """Run batched RAGAS; returns legacy dict + ``__meta__``."""
    settings = get_settings()
    out: Dict[str, Dict] = {}
    messages: list[str] = []

    if not rows:
        out["__meta__"] = {"status": "failed", "backend": "ragas", "messages": ["No rows"], "errors": []}
        return out

    for strategy, _, answer in rows:
        out[strategy] = {"answer": answer}

    n = len(rows)
    data: Dict[str, list] = {
        "question": [query] * n,
        "answer": [a for _, _, a in rows],
        "contexts": [c for _, c, _ in rows],
    }
    if ground_truth:
        data["ground_truth"] = [ground_truth] * n

    try:
        ragas_llm, ragas_embeddings = _make_ragas_backends(settings)
        metrics = [faithfulness]
        if ground_truth:
            metrics.append(context_precision)
        for m in metrics:
            m.llm = ragas_llm
            if hasattr(m, "embeddings"):
                m.embeddings = ragas_embeddings

        result = ragas_evaluate(
            dataset=Dataset.from_dict(data),
            metrics=metrics,
            raise_exceptions=False,
            run_config=RunConfig(timeout=180, max_workers=1, max_retries=1),
        )
        df = result.to_pandas()
        ok_count = 0
        for i, (strategy, _, _) in enumerate(rows):
            scores = _scores_for_row(df, i)
            out[strategy].update(scores)
            if scores.get("faithfulness") is not None:
                ok_count += 1
            else:
                out[strategy]["faithfulness"] = None
                messages.append(f"{strategy}: RAGAS returned no faithfulness score")
        status = "ok" if ok_count == n else ("partial" if ok_count else "failed")
    except Exception as exc:
        logger.exception("RAGAS batch failed")
        status = "failed"
        messages.append(str(exc))
        for strategy, _, _ in rows:
            out[strategy]["faithfulness"] = None
            out[strategy]["error"] = str(exc)

    out["__meta__"] = {
        "status": status,
        "backend": "ragas",
        "messages": messages,
        "errors": [{"strategy": s, "error": out[s].get("error")} for s, _, _ in rows if out[s].get("error")],
    }
    return out
