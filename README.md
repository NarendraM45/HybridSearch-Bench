<div align="center">

# 🔍 HybridSearch Bench

**A research-grade RAG evaluation framework built for production.**  
*BM25 + Dense Vector + RRF Fusion · Cross-Encoder Reranking · HyDE · Parent-Child Retrieval · ColBERT*  
*Fully local. Zero paid APIs. Runs entirely on Ollama.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=for-the-badge)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> Upload any PDF corpus → ingest with semantic or recursive chunking → retrieve with BM25, Vector, or Hybrid (RRF) → rerank with a cross-encoder → evaluate with RAGAS + custom IR metrics — all on your machine, no API key required.

</div>

---

## 📐 Advanced Pipeline Architecture

```
                        ┌────────────────────────────────────────────────┐
                        │                INGESTION LAYER                  │
                        │                                                  │
                        │  PDFs → PyPDFLoader → Chunker (selectable)      │
                        │                                                  │
                        │  ┌──────────────────┐  ┌──────────────────────┐ │
                        │  │  RecursiveChunker │  │   SemanticChunker    │ │
                        │  │  512 chars        │  │   spaCy sentences    │ │
                        │  │  64 overlap       │  │   cosine breakpoint  │ │
                        │  └──────────────────┘  └──────────────────────┘ │
                        │  Parent-Child mode: child=128c → retrieval       │
                        │                    parent=512c → LLM context     │
                        │  content-hash dedup (md5[:12])                   │
                        └───────────────────┬────────────────────────────-─┘
                                            │
                   ┌────────────────────────┴───────────────────────────┐
                   │                                                     │
                   ▼                                                     ▼
        ┌──────────────────────┐                  ┌─────────────────────────────┐
        │      BM25 INDEX       │                  │       CHROMA VECTOR DB       │
        │  rank_bm25            │                  │  nomic-embed-text (768d)     │
        │  stop-word filtered   │                  │  hnsw:space=cosine           │
        │  technical tokenizer  │                  │  dual collection: P/C mode   │
        └──────────┬────────────┘                  └──────────────┬───────────────┘
                   │                                               │
                   │                               ┌──────────────┴────────────┐
                   │                               │       HyDE EXPANSION       │
                   │                               │  llama3.2 generates a      │
                   │                               │  hypothetical answer →     │
                   │                               │  embed that, not query     │
                   │                               │  blend_α: query + HyDE     │
                   │                               └──────────────┬────────────┘
                   │                                              │
                   └─────────────────────┬────────────────────────┘
                                         │ top-4K candidates each
                                         ▼
                            ┌────────────────────────────┐
                            │         RRF FUSION          │
                            │  score(d) = Σ 1/(k + rank)  │  k=60 (Cormack 2009)
                            │  k exposed as UI slider     │
                            └──────────────┬──────────────┘
                                           │ top-2K fused results
                                           ▼
                            ┌────────────────────────────┐
                            │    PARENT-CHILD LOOKUP      │
                            │  child text → parent text  │
                            │  (better context for LLM)  │
                            └──────────────┬──────────────┘
                                           │
                                           ▼
                            ┌────────────────────────────┐
                            │  CROSS-ENCODER RERANKER     │
                            │  BAAI/bge-reranker-large    │
                            │  joint (query, chunk) attn  │
                            │  solves near-dupe docs ✅   │
                            └──────────────┬──────────────┘
                                           │ top-K precise chunks
                                           ▼
                            ┌────────────────────────────┐
                            │     OLLAMA GENERATION       │
                            │     llama3.2                │
                            │     grounded system prompt  │
                            └──────────────┬──────────────┘
                                           │
          ┌────────────────────────────────┴──────────────────────────────┐
          │                        EVALUATION LAYER                        │
          │  RAGAS Metrics              │  Custom IR Metrics (from scratch) │
          │  ──────────────             │  ──────────────────────────────── │
          │  Faithfulness               │  MRR, MAP                         │
          │  Answer Relevancy           │  NDCG@k  (k = 1, 3, 5, 10)       │
          │  Context Precision          │  Precision@k, Reranking Delta     │
          └───────────────────────────────────────────────────────────────┘
```

> **ColBERT mode** (experimental, `COLBERT_ENABLED=true`): bypasses BM25+Vector+RRF entirely. Uses `RAGatouille` for token-level MaxSim late interaction — cross-encoder precision at retrieval speed.

---

## ✨ Why This Stands Out

| Capability | HybridSearch Bench | Typical RAG Demo |
|---|---|---|
| Retrieval strategies | BM25 + Vector + Hybrid (RRF) + ColBERT | Vector only |
| Near-duplicate doc problem | ✅ Cross-encoder reranking (joint attention) | ❌ Both signals collapse |
| Query-doc alignment | ✅ HyDE (hypothetical doc embedding) | ❌ Raw short query vs long passage |
| Context size for LLM | ✅ Parent-child dual-index | ❌ Fixed chunk size tradeoff |
| Chunking quality | ✅ Semantic (spaCy) or Recursive — configurable | Fixed character split only |
| API dependency | ❌ Zero — Ollama + HuggingFace cache | ✅ OpenAI required |
| Evaluation | RAGAS + MRR + MAP + NDCG@k + reranking delta | None, or basic |
| Architecture | Protocol interfaces, Pydantic v2 BaseSettings | Monolithic script |
| DevOps | Multi-stage Docker, Click CLI, Makefile, tests | `python app.py` |

---

## 🧠 Key Technical Decisions

### Reciprocal Rank Fusion
```
score(d) = Σᵢ  1 / (k + rankᵢ(d))        k = 60  (Cormack 2009 default)
```
Documents appearing in **both** BM25 and vector lists receive compounded scores. `k` controls compression — exposed as a live slider so you can tune per corpus.

### Why BM25 Still Matters
Dense vectors compress semantics but lose rare tokens. In medical/research corpora, terms like `EEG`, `FTD`, `HNSW`, or `p300` carry extreme discriminative weight that BM25 captures exactly — fusion gets the best of both representations.

### Cross-Encoder vs Bi-Encoder
Bi-encoders encode query and document **independently** — fast but imprecise when documents are near-identical. A cross-encoder encodes the `(query, chunk)` pair **jointly** via full self-attention, producing a true relevance score. That's what separates near-duplicate chunks correctly.

### HyDE — Closing the Lexical Gap
Queries are short and phrased differently from how answers appear in documents (`"what causes X?"` vs `"X is caused by..."`). HyDE generates a hypothetical answer via Ollama and embeds **that** instead. The blend parameter `α` controls the mix between raw query embedding and HyDE embedding.

### Parent-Child Retrieval
Child chunks (128 chars) retrieve precisely. Parent chunks (512 chars) give the LLM sufficient context. ChromaDB holds two collections — retrieval scores come from children, LLM receives the corresponding parent text.

---

## 🗂️ Project Structure

```
HybridSearch Bench/
│
├── 📊  app.py                  # 4-tab Streamlit dashboard
│                               #   Tab 1: Query & Compare (BM25 | Vector | Hybrid | ColBERT)
│                               #   Tab 2: Eval Dashboard (RAGAS radar + grouped bar chart)
│                               #   Tab 3: Latency Breakdown (per-stage stacked bar)
│                               #   Tab 4: History (metric trend lines across queries)
│
├── ⌨️  cli.py                  # Click CLI — ingest / query / evaluate / benchmark / serve
│
├── 🗄️  chroma_store.py         # ChromaDB — dual collections (parent/child), upsert, lifecycle
├── ✂️  chunker.py              # RecursiveChunker + SemanticChunker (spaCy sentence breakpoints)
├── ⚙️  config.py               # Compile-time constants (separator hierarchy, batch sizes)
├── 🔢  embedder.py             # Ollama embedding wrapper (batch=32, 768-dim nomic)
│
├── 🔍  retrieval.py            # BM25Strategy / VectorStrategy / HybridStrategy (RRF)
├── 🔀  reranker.py             # CrossEncoderReranker — BAAI/bge-reranker-large, CPU/GPU
├── 💡  hyde.py                 # HyDEQueryExpander — hypothetical doc embed + alpha blend
├── 🤖  colbert_retriever.py    # ColBERTRetriever via RAGatouille (experimental)
│
├── 📈  metrics.py              # MRR, MAP, NDCG@k, Precision@k, reranking_delta — from scratch
├── 🧪  evaluation.py           # RAGAS via LangchainLLMWrapper(ChatOllama) + Ollama embeddings
├── 🧪  evaluation_ragas.py     # RAGAS dataset builder and metric aggregation helpers
├── 🧪  ollama_eval.py          # Ollama-native grounded answer generation + local eval
├── 🔄  pipeline.py             # HybridSearchPipeline — full stage orchestration + per-stage timing
│
├── 📐  interfaces.py           # Protocol definitions — Retriever, Embedder, Reranker, Evaluator
├── 🚨  exceptions.py           # Custom exception hierarchy (IngestionError, RetrievalError…)
├── 📝  logger_config.py        # Structured JSON logging with contextual stage fields
├── ⚙️  settings.py             # Pydantic BaseSettings — all config from .env, validated at start
│
├── 🐳  Dockerfile              # Multi-stage build (builder + slim runtime, ~200MB final image)
├── 🐳  docker-compose.yml      # app + ollama + ollama-init service (auto model pull on boot)
├── 🔧  Makefile                # install / lint / test / run / docker-up / clean
├── 📋  pyproject.toml          # Package config, entry points, ruff + mypy settings
├── 📦  requirements.txt
├── 📦  requirements-dev.txt
├── 🔑  .env.example
└── 🧪  tests/
    ├── conftest.py             # Shared fixtures
    ├── test_metrics.py         # IR metric correctness — edge cases + hand-computed values
    ├── test_chunker.py         # Chunk size bounds, overlap, dedup
    ├── test_retrieval.py       # RRF fusion math, score normalization (mock ChromaDB)
    ├── test_reranker.py        # CrossEncoder output ordering, threshold filter, OOM fallback
    ├── test_hyde.py            # Blend math (α=0.5 → midpoint), LLM failure fallback
    └── test_parent_child.py    # Child retrieval returns parent text, dual collection separation
```

---

## 🚀 Quickstart

### Option A — Docker *(recommended — one command)*

```bash
# 1. Clone and configure
git clone https://github.com/YOUR_USERNAME/HybridSearch-Bench.git
cd HybridSearch-Bench
cp .env.example .env

# 2. Start the full stack
#    Spins up: app + Ollama + ollama-init (auto-pulls models ~4 GB first run)
make docker-up

# 3. Open the dashboard
open http://localhost:8501
```

> `ollama-init` automatically pulls `llama3.2` and `nomic-embed-text` on first boot. Subsequent starts skip this and are instant.

---

### Option B — Local Python

**Prerequisites:** [Ollama](https://ollama.com/download) installed and running.

```bash
# 1. Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Install project
git clone https://github.com/YOUR_USERNAME/HybridSearch-Bench.git
cd HybridSearch-Bench
cp .env.example .env
make install                          # pip install -e ".[dev]"

# 3. Install spaCy model (for SemanticChunker)
python -m spacy download en_core_web_sm

# 4. Ingest your PDFs
mkdir -p data/pdfs
# drop PDFs into data/pdfs/
make ingest

# 5. Launch
make run                              # → http://localhost:8501
```

> The reranker (`BAAI/bge-reranker-large`, ~1.3 GB) downloads automatically to your HuggingFace cache on first query. Nothing to pull manually.

---

### Option C — ColBERT (Experimental)

```bash
# Enable in .env
COLBERT_ENABLED=true

# Ingest with ColBERT indexing
python -m cli ingest --pdf-dir data/pdfs --colbert

# Query using ColBERT strategy
python -m cli query --question "EEG signal classification" --strategy colbert --top-k 5
```

> Downloads `colbert-ir/colbertv2.0` (~700 MB) via RAGatouille on first use.

---

## ⌨️ CLI Reference

```
python -m cli [COMMAND] [OPTIONS]
```

| Command | What it does | Key flags |
|---|---|---|
| `ingest` | PDF → chunks → embeddings → ChromaDB | `--pdf-dir`, `--collection`, `--chunker [recursive\|semantic]` |
| `query` | Single question, one strategy | `--question`, `--strategy [bm25\|vector\|hybrid\|colbert]`, `--top-k`, `--no-rerank` |
| `evaluate` | RAGAS + IR metrics on JSONL set | `--questions-file`, `--output`, `--strategy` |
| `benchmark` | All strategies side-by-side | `--questions-file`, `--output` |
| `serve` | Launch Streamlit dashboard | — |

**Examples:**

```bash
# Ingest with semantic chunking
python -m cli ingest --pdf-dir data/pdfs --collection eeg_papers --chunker semantic

# Hybrid + reranker (default on)
python -m cli query --question "attention mechanism in EEG classification" \
  --strategy hybrid --top-k 5

# Hybrid without reranker (faster, for comparison)
python -m cli query --question "attention mechanism in EEG classification" \
  --strategy hybrid --top-k 10 --no-rerank

# RAGAS evaluation
python -m cli evaluate --questions-file eval_set.jsonl \
  --output results/eval.csv --strategy hybrid

# Cross-strategy benchmark
python -m cli benchmark --questions-file eval_set.jsonl \
  --output results/benchmark.csv
```

**JSONL format for `--questions-file`:**
```json
{"question": "What is RAG?", "ground_truth": "Retrieval-Augmented Generation is..."}
{"question": "How does BM25 rank documents?", "ground_truth": "BM25 uses TF-IDF with..."}
```

---

## 📊 Evaluation Metrics

### RAGAS — Answer Quality

| Metric | What it measures | Range |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved contexts? Detects hallucination. | 0 → 1 |
| **Answer Relevancy** | Does the answer directly address the question? | 0 → 1 |
| **Context Precision** | Are the most relevant chunks ranked highest in the retrieved set? | 0 → 1 |

### Custom IR Metrics — Retrieval Quality

Implemented **from first principles** in `metrics.py`. No external IR library used.

**Mean Reciprocal Rank (MRR)**
```
MRR = (1/|Q|) × Σ  1 / rank_first_relevant(q)
```

**Mean Average Precision (MAP)**
```
AP(q) = Σₖ [ P(k) × rel(k) ] / |relevant(q)|
MAP   = (1/|Q|) × Σ AP(q)
```

**Normalized Discounted Cumulative Gain (NDCG@k)**
```
DCG@k  = Σᵢ₌₁ᵏ  relᵢ / log₂(i + 1)
NDCG@k = DCG@k / IDCG@k          (IDCG = perfect-ordering DCG)
```
Reported at `k ∈ {1, 3, 5, 10}`.

**Reranking Delta**
```
Δ_NDCG@k = NDCG@k(post_rerank) − NDCG@k(pre_rerank)
```
Quantifies how much the cross-encoder improves over RRF fusion alone.

---

## ⚙️ Configuration

All features are toggleable via `.env`. Copy `.env.example` → `.env` to start.

**Core**

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | LLM for answer generation |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model (768-dim) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `hybrid_bench` | Collection name |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `LOG_FORMAT` | `json` | `json` for prod, `text` for local dev |

**Chunking**

| Variable | Default | Description |
|---|---|---|
| `CHUNKER_TYPE` | `recursive` | `recursive` or `semantic` |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `SEMANTIC_BREAKPOINT_PERCENTILE` | `95` | Cosine-drop threshold for semantic splits |

**Retrieval**

| Variable | Default | Description |
|---|---|---|
| `TOP_K` | `5` | Final chunks returned per query |
| `RRF_K` | `60` | RRF constant (Cormack 2009 default) |

**Parent-Child**

| Variable | Default | Description |
|---|---|---|
| `PARENT_CHILD_ENABLED` | `true` | Dual-index parent-child retrieval |
| `CHILD_CHUNK_SIZE` | `128` | Retrieval chunk size |
| `PARENT_CHUNK_SIZE` | `512` | LLM context chunk size |

**HyDE**

| Variable | Default | Description |
|---|---|---|
| `HYDE_ENABLED` | `true` | Hypothetical document query expansion |
| `HYDE_BLEND_ALPHA` | `0.5` | `1.0` = pure HyDE · `0.0` = raw query · `0.5` = average |
| `HYDE_FALLBACK_ON_ERROR` | `true` | Fall back to raw query if LLM fails |

**Reranker**

| Variable | Default | Description |
|---|---|---|
| `RERANKER_ENABLED` | `true` | Cross-encoder reranking stage |
| `RERANKER_MODEL` | `BAAI/bge-reranker-large` | Auto-downloads to HF cache (~1.3 GB) |
| `RERANKER_TOP_K` | `5` | Chunks to return after reranking |
| `RERANKER_THRESHOLD` | `0.3` | Filter chunks below this relevance score |

**ColBERT (experimental)**

| Variable | Default | Description |
|---|---|---|
| `COLBERT_ENABLED` | `false` | Token-level late interaction via RAGatouille |
| `COLBERT_INDEX_PATH` | `./colbert_index` | ColBERT index storage directory |

---

## 🧱 Tech Stack

| Layer | Technology | Justification |
|---|---|---|
| **LLM** | Ollama (`llama3.2`) | Fully local, swappable, zero API cost |
| **Embeddings** | `nomic-embed-text` via Ollama | 768-dim, outperforms MiniLM on BEIR benchmarks |
| **Sparse retrieval** | `rank-bm25` | Gold standard for keyword-heavy technical corpora |
| **Vector store** | ChromaDB (HNSW, cosine) | Embedded, no server, production-serializable |
| **Fusion** | Reciprocal Rank Fusion | Theoretically grounded, single parameter (Cormack 2009) |
| **Query expansion** | HyDE (custom, Ollama-backed) | Closes lexical gap with no external dependencies |
| **Reranking** | `BAAI/bge-reranker-large` | Joint attention — ~40% precision gain over bi-encoder |
| **Late interaction** | ColBERT via RAGatouille | Token-level MaxSim scoring |
| **Chunking** | Recursive + Semantic (spaCy) | Semantic-aware boundaries vs character-count split |
| **Settings** | Pydantic v2 `BaseSettings` | Type-safe, env-driven, startup-validated |
| **CLI** | Click | Composable, testable, self-documenting |
| **Dashboard** | Streamlit | Interactive eval + latency visualization |
| **Containerization** | Docker multi-stage + Compose | Reproducible, lean runtime (~200 MB) |
| **Testing** | Pytest + coverage | Pure-function tests — no mocks for IR math |

---

## 🧪 Tests

```bash
make test
# → pytest tests/ -v --cov=. --cov-report=term-missing
```

| Test file | Covers |
|---|---|
| `test_metrics.py` | MRR, MAP, NDCG@k — empty lists, perfect retrieval, partial, wrong order |
| `test_chunker.py` | Chunk size bounds, overlap correctness, content-hash deduplication |
| `test_retrieval.py` | RRF fusion math with hand-computed expected values, score normalization |
| `test_reranker.py` | Output ordering, threshold filtering, graceful CPU fallback on OOM |
| `test_hyde.py` | Blend math (α=0.5 → exact midpoint vector), LLM error fallback |
| `test_parent_child.py` | Child retrieval maps to correct parent, collection isolation |

```bash
make lint     # ruff check . && mypy .
make format   # ruff format .
```

---

## 🛠️ Makefile Targets

```bash
make install      # pip install -e ".[dev]"
make run          # streamlit run app.py → http://localhost:8501
make ingest       # python -m cli ingest --pdf-dir data/pdfs
make test         # pytest tests/ -v --cov=. --cov-report=term-missing
make lint         # ruff check . && mypy .
make format       # ruff format .
make docker-up    # docker compose up --build -d
make docker-down  # docker compose down
make clean        # rm -rf __pycache__ .pytest_cache .mypy_cache dist
```

---

## 📁 Recommended Datasets

| Domain | Source | Why it showcases the pipeline |
|---|---|---|
| 🧠 **EEG / Neuroscience** | ArXiv on EEG classification | Dense technical jargon — BM25 advantage clearly visible |
| 🏥 **Medical AI** | Clinical guidelines, PubMed | Near-duplicate docs — cross-encoder advantage clearly visible |
| 🤖 **ML Systems** | Transformer / LLM papers | Mixed keyword + semantic — hybrid advantage clearly visible |
| 📚 **Technical Docs** | Framework/API documentation | Long docs with buried answers — parent-child advantage visible |

> Use 20–50 papers on a narrow topic for meaningful contrast between strategies.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Ensure both pass: `make lint && make test`
4. Open a PR with a clear description of the problem it solves

---

<div align="center">

**Built for the ML engineering portfolio — every design decision is defensible in an interview.**  
*If this helped you, a ⭐ goes a long way.*

<br/>

MIT License © 2026

</div>