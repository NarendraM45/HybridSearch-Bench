"""
hybridsearch/config/logging.py
================================
Structured logging configuration using ``structlog``.

Why structlog?
--------------
- Every log event is a dict → trivially ingested by Datadog / ELK / CloudWatch.
- Processors are composable: swap in ConsoleRenderer for human-readable output
  without changing call sites.
- ``bound_logger = logger.bind(retriever="hybrid", query_id="…")`` adds
  context to every subsequent log call — zero repetition.

Usage
-----
    from logger_config import get_logger

    log = get_logger(__name__)
    log.info("retrieval.complete", strategy="hybrid", n_results=5, latency_ms=42.3)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from settings import LogFormat, get_settings


def configure_logging() -> None:
    """
    Configure structlog once at application startup.

    Call this in ``app.py`` and ``cli.py`` before any module logs.
    Idempotent — safe to call multiple times.
    """
    settings = get_settings()

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == LogFormat.JSON:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # Silence noisy third-party loggers
    for name in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to ``name``."""
    return structlog.get_logger(name)
