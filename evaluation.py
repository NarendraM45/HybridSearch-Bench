"""
evaluation.py — Answer generation + RAG quality scoring.

Default backend: ``ollama`` (direct prompts, numeric scores — reliable on local Ollama).
Optional backend: ``ragas`` (set EVAL_BACKEND=ragas; requires JSON-compliant model output).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import ollama

from config import MAX_ANSWER_TOKENS, OLLAMA_MODEL
from ollama_eval import EvaluationReport, StrategyEvalResult, evaluate_strategy
from settings import get_settings

logger = logging.getLogger(__name__)

_EVAL_STRATEGIES = ("bm25", "vector", "hybrid")
ProgressCb = Callable[[float, str], None]

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
    """Call Ollama to produce a grounded answer from retrieved contexts."""
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_base_url, timeout=180.0)

    context_block = "\n\n---\n\n".join(
        f"[{i+1}] {ctx.strip()}" for i, ctx in enumerate(contexts[:3])
    )
    prompt = _ANSWER_TMPL.format(context=context_block, question=query)

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        options={"num_predict": MAX_ANSWER_TOKENS},
    )
    return response["message"]["content"].strip()


def _evaluate_with_ollama(
    query: str,
    rows: list[tuple[str, list[str], str]],
    ground_truth: Optional[str],
    progress_callback: ProgressCb | None = None,
) -> EvaluationReport:
    report = EvaluationReport(strategies={}, backend="ollama")
    n = len(rows)

    for i, (strategy, contexts, answer) in enumerate(rows):
        if progress_callback:
            progress_callback(
                0.6 + (i / max(n, 1)) * 0.35,
                f"Scoring **{strategy}** (Ollama evaluator)…",
            )
        try:
            res = evaluate_strategy(query, contexts, answer, ground_truth)
            report.strategies[strategy] = res
            if res.status == "failed":
                report.messages.append(f"{strategy}: {res.error}")
            elif res.status == "partial":
                report.messages.append(f"{strategy} (partial): {res.error}")
        except Exception as exc:
            logger.exception("Evaluation failed for %s", strategy)
            report.strategies[strategy] = StrategyEvalResult(
                answer=answer,
                status="failed",
                error=str(exc),
            )
            report.messages.append(f"{strategy}: {exc}")

    statuses = [r.status for r in report.strategies.values()]
    if all(s == "ok" for s in statuses):
        report.overall_status = "ok"
    elif any(s == "ok" for s in statuses):
        report.overall_status = "partial"
    else:
        report.overall_status = "failed"

    return report


def _evaluate_with_ragas(
    query: str,
    rows: list[tuple[str, list[str], str]],
    ground_truth: Optional[str],
) -> EvaluationReport:
    """Optional RAGAS path — only used when EVAL_BACKEND=ragas."""
    from evaluation_ragas import evaluate_batch_ragas

    report = EvaluationReport(strategies={}, backend="ragas")
    try:
        legacy = evaluate_batch_ragas(query, rows, ground_truth)
        for strategy, _, answer in rows:
            data = legacy.get(strategy, {})
            report.strategies[strategy] = StrategyEvalResult(
                answer=data.get("answer", answer),
                faithfulness=data.get("faithfulness"),
                context_precision=data.get("context_precision"),
                status="ok" if data.get("faithfulness") is not None else "failed",
                error=data.get("error"),
            )
        meta = legacy.get("__meta__", {})
        report.overall_status = meta.get("status", "partial")
        report.messages.extend(meta.get("messages", []))
    except Exception as exc:
        logger.exception("RAGAS backend failed")
        report.overall_status = "failed"
        report.messages.append(f"RAGAS backend error: {exc}")
        for strategy, _, answer in rows:
            report.strategies[strategy] = StrategyEvalResult(
                answer=answer, status="failed", error=str(exc)
            )
    return report


def evaluate_all_strategies(
    query: str,
    retrieval_results: Dict[str, List[dict]],
    api_key: str = "",
    ground_truth: Optional[str] = None,
    progress_callback: ProgressCb | None = None,
) -> Dict[str, Dict]:
    """
    Generate answers and score each retrieval strategy.

    Returns a dict keyed by strategy plus ``__meta__`` with overall status/errors.
    """
    settings = get_settings()
    strategies = [s for s in _EVAL_STRATEGIES if s in retrieval_results]
    eval_rows: list[tuple[str, list[str], str]] = []
    pre_failed: dict[str, StrategyEvalResult] = {}

    for i, strategy in enumerate(strategies):
        if progress_callback:
            progress_callback(
                (i / max(len(strategies), 1)) * 0.55,
                f"Generating answer for **{strategy}**…",
            )
        contexts = [r["text"] for r in retrieval_results[strategy]][:3]
        try:
            answer = generate_answer(query, contexts, api_key)
            logger.info("[%s] answer: %s", strategy, answer[:80])
            eval_rows.append((strategy, contexts, answer))
        except Exception as exc:
            logger.exception("Answer generation failed for %s", strategy)
            pre_failed[strategy] = StrategyEvalResult(
                answer=f"[Answer generation failed: {exc}]",
                status="failed",
                error=str(exc),
            )

    if not eval_rows and not pre_failed:
        empty = EvaluationReport(strategies={}, overall_status="failed")
        empty.messages.append("No strategies to evaluate.")
        return empty.to_legacy_dict()

    backend = settings.eval_backend.lower()
    if backend == "ragas":
        report = _evaluate_with_ragas(query, eval_rows, ground_truth)
    else:
        report = _evaluate_with_ollama(query, eval_rows, ground_truth, progress_callback)

    report.strategies.update(pre_failed)
    if pre_failed:
        report.messages.extend(f"{s}: answer generation failed" for s in pre_failed)
        if report.overall_status == "ok":
            report.overall_status = "partial"

    if progress_callback:
        progress_callback(1.0, "Evaluation complete ✓")

    return report.to_legacy_dict()


def evaluate_single(
    query: str,
    contexts: List[str],
    answer: str,
    api_key: str = "",
    ground_truth: Optional[str] = None,
) -> Dict[str, float | None]:
    """Score a single triple (CLI / tests)."""
    res = evaluate_strategy(query, contexts[:3], answer, ground_truth)
    out: Dict[str, float | None] = {"faithfulness": res.faithfulness}
    if ground_truth:
        out["context_precision"] = res.context_precision
    return out
