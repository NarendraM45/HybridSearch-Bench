"""
ollama_eval.py — Ollama-native RAG metrics (no RAGAS / no JSON schema parsing).

Designed for local llama3.x + nomic-embed-text stacks where RAGAS faithfulness
routinely fails with OutputParserException on free-form JSON outputs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

import ollama

from settings import get_settings

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"(?:^|[^\d])(0?\.\d+|[01](?:\.0+)?)(?:[^\d]|$)")

_FAITHFULNESS_PROMPT = """\
You are a strict RAG evaluator. Judge whether the ANSWER is fully supported by the CONTEXT snippets only (no outside knowledge).

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONE number from 0.0 to 1.0 where:
0.0 = answer contradicts or is unsupported by context
1.0 = answer is fully grounded in context

Output only the number, e.g. 0.75"""

_CONTEXT_PRECISION_PROMPT = """\
You are a retrieval evaluator. Judge how useful the CONTEXT snippets are for producing the REFERENCE answer to the QUESTION.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CONTEXT:
{context}

Respond with ONE number from 0.0 to 1.0 where:
0.0 = context is irrelevant
1.0 = context contains everything needed for the reference answer

Output only the number, e.g. 0.80"""


@dataclass
class MetricResult:
    name: str
    score: float | None
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None
    raw_response: str | None = None


@dataclass
class StrategyEvalResult:
    answer: str
    faithfulness: float | None = None
    context_precision: float | None = None
    status: Literal["ok", "partial", "failed"] = "ok"
    error: str | None = None
    metrics: dict[str, MetricResult] = field(default_factory=dict)


@dataclass
class EvaluationReport:
  """Aggregate outcome for a multi-strategy evaluation run."""

  strategies: dict[str, StrategyEvalResult]
  backend: str = "ollama"
  overall_status: Literal["ok", "partial", "failed"] = "ok"
  messages: list[str] = field(default_factory=list)

  def to_legacy_dict(self) -> dict:
    """Shape expected by Streamlit / CLI (plus ``__meta__``)."""
    out: dict = {}
    for name, res in self.strategies.items():
      row: dict = {"answer": res.answer, "status": res.status, "error": res.error}
      if res.faithfulness is not None:
        row["faithfulness"] = res.faithfulness
      elif res.status != "ok":
        row["faithfulness"] = None
      if res.context_precision is not None:
        row["context_precision"] = res.context_precision
      elif any(m.name == "context_precision" for m in res.metrics.values()):
        row["context_precision"] = None
      out[name] = row
    out["__meta__"] = {
      "status": self.overall_status,
      "backend": self.backend,
      "messages": self.messages,
      "errors": [
        {"strategy": n, "error": r.error}
        for n, r in self.strategies.items()
        if r.error
      ],
    }
    return out


def parse_score(text: str) -> float | None:
  """Extract the first plausible score in [0, 1] from model output."""
  if not text:
    return None
  text = text.strip()
  try:
    val = float(text)
    if 0.0 <= val <= 1.0:
      return round(val, 4)
  except ValueError:
    pass
  for match in _SCORE_RE.finditer(text):
    try:
      val = float(match.group(1))
      if 0.0 <= val <= 1.0:
        return round(val, 4)
    except ValueError:
      continue
  return None


def _ollama_chat(prompt: str, model: str, base_url: str, timeout: float) -> str:
  client = ollama.Client(host=base_url, timeout=timeout)
  response = client.chat(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0, "num_predict": 16},
  )
  return (response.get("message") or {}).get("content", "").strip()


def score_faithfulness(
  answer: str,
  contexts: list[str],
  *,
  model: str | None = None,
  base_url: str | None = None,
  timeout: float | None = None,
) -> MetricResult:
  settings = get_settings()
  model = model or settings.ollama_model
  base_url = base_url or settings.ollama_base_url
  timeout = timeout or float(getattr(settings, "eval_timeout", 120))

  context_block = "\n\n---\n\n".join(f"[{i+1}] {c.strip()}" for i, c in enumerate(contexts[:3]))
  prompt = _FAITHFULNESS_PROMPT.format(context=context_block, answer=answer.strip())

  try:
    raw = _ollama_chat(prompt, model, base_url, timeout)
    score = parse_score(raw)
    if score is None:
      return MetricResult(
        name="faithfulness",
        score=None,
        status="failed",
        error=f"Could not parse score from model output: {raw[:200]!r}",
        raw_response=raw,
      )
    return MetricResult(name="faithfulness", score=score, raw_response=raw)
  except Exception as exc:
    logger.warning("faithfulness scoring failed: %s", exc)
    return MetricResult(name="faithfulness", score=None, status="failed", error=str(exc))


def score_context_precision(
  question: str,
  reference: str,
  contexts: list[str],
  *,
  model: str | None = None,
  base_url: str | None = None,
  timeout: float | None = None,
) -> MetricResult:
  settings = get_settings()
  model = model or settings.ollama_model
  base_url = base_url or settings.ollama_base_url
  timeout = timeout or float(getattr(settings, "eval_timeout", 120))

  context_block = "\n\n---\n\n".join(f"[{i+1}] {c.strip()}" for i, c in enumerate(contexts[:3]))
  prompt = _CONTEXT_PRECISION_PROMPT.format(
    question=question.strip(),
    reference=reference.strip(),
    context=context_block,
  )

  try:
    raw = _ollama_chat(prompt, model, base_url, timeout)
    score = parse_score(raw)
    if score is None:
      return MetricResult(
        name="context_precision",
        score=None,
        status="failed",
        error=f"Could not parse score from model output: {raw[:200]!r}",
        raw_response=raw,
      )
    return MetricResult(name="context_precision", score=score, raw_response=raw)
  except Exception as exc:
    logger.warning("context_precision scoring failed: %s", exc)
    return MetricResult(name="context_precision", score=None, status="failed", error=str(exc))


def evaluate_strategy(
  question: str,
  contexts: list[str],
  answer: str,
  ground_truth: str | None = None,
) -> StrategyEvalResult:
  """Score one (question, contexts, answer) triple."""
  contexts = contexts[:3]
  result = StrategyEvalResult(answer=answer)
  errors: list[str] = []

  f = score_faithfulness(answer, contexts)
  result.metrics["faithfulness"] = f
  result.faithfulness = f.score
  if f.status == "failed":
    errors.append(f"faithfulness: {f.error}")

  if ground_truth:
    cp = score_context_precision(question, ground_truth, contexts)
    result.metrics["context_precision"] = cp
    result.context_precision = cp.score
    if cp.status == "failed":
      errors.append(f"context_precision: {cp.error}")

  if errors and result.faithfulness is None and result.context_precision is None:
    result.status = "failed"
    result.error = "; ".join(errors)
  elif errors:
    result.status = "partial"
    result.error = "; ".join(errors)
  else:
    result.status = "ok"

  return result
