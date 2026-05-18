<div align="center">

# 🔍 HybridSearch Bench

**Production-grade RAG evaluation framework with hybrid retrieval and automatic RAGAS-based scoring.**  
*Fully local. Zero paid APIs. Runs on Ollama.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=for-the-badge)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> Upload any PDF corpus → get BM25 + Vector + Hybrid retrieval → compare strategies head-to-head → auto-score with RAGAS + custom IR metrics. Everything runs locally.

</div>

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER                              │
│                                                                       │
│   PDFs  ──►  PyPDFLoader  ──►  RecursiveTextSplitter  ──►  Chunks   │
│                                   (512 chars / 64 overlap)           │
│                                         │                             │
│                              content-hash dedup (md5[:12])           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┴─────────────────────┐
          │                                           │
          ▼                                           ▼
┌─────────────────────┐                   ┌──────────────────────────┐
│    BM25 INDEX        │                   │     CHROMA VECTOR DB      │
│                      │                   │                           │
│  rank_bm25 tokenizer │                   │  nomic-embed-text (768d)  │
│  stop-word filtered  │                   │  hnsw:space=cosine        │
│  technical corpus    │                   │  batch_size=32            │
└────────┬────────────┘                   └───────────┬──────────────┘
         │   BM25 ranked list                         │   cosine similarity
         │                                            │   (1 - dist)
         └─────────────────┬──────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │     RRF FUSION          │
              │                        │
              │  score(d) = Σ 1/(k+r)  │   k=60 (Cormack 2009)
              │  k exposed as slider   │
              └────────────┬───────────┘
                           │  top-k fused chunks
                           ▼
              ┌────────────────────────┐
              │    OLLAMA GENERATION    │
              │    llama3.2             │
              │    grounded prompt      │
              └────────────┬───────────┘
                           │  answer + contexts
                           ▼
         ┌─────────────────────────────────────┐
         │           EVALUATION LAYER           │
         │                                      │
         │  RAGAS          │  Custom IR Metrics  │
         │  ─────────────  │  ─────────────────  │
         │  Faithfulness   │  MRR, MAP           │
         │  Ans Relevancy  │  NDCG@k (k=1,3,5)  │
         │  Ctx Precision  │  Precision@k        │
         └─────────────────────────────────────┘
```

---

## ✨ What Makes This Different

| Capability | This Project | Typical RAG Demo |
|---|---|---|
| Retrieval strategies | BM25 + Vector + Hybrid (RRF) | Vector only |
| API dependency | ❌ Zero — fully local via Ollama | ✅ OpenAI required |
| Evaluation | RAGAS + custom IR metrics from scratch | None or basic |
| Architecture | Protocol interfaces, Pydantic v2 settings | Monolithic script |
| Ops | Multi-stage Docker, CLI, Makefile | `python app.py` |
| Tests | Pytest suite with coverage | None |

---

## 🧠 Core Concepts

### Reciprocal Rank Fusion (RRF)
```
score(d) = Σᵢ  1 / (k + rankᵢ(d))
```
Documents appearing in **both** BM25 and vector result lists receive compounded scores. `k=60` is the Cormack (2009) default — exposed as a tunable slider in the dashboard.

### Why BM25 Matters for Technical Text
Dense vectors compress semantics but lose rare tokens. In medical/research corpora, terms like `"EEG"`, `"FTD"`, or `"p300"` carry extreme discriminative weight that BM25 captures exactly — fusion gets the best of both.

### RAGAS via Local LLM
The same `nomic-embed-text` encoder used for indexing is **reused** as the RAGAS embedding backend — no second model load, no API cost.

---

## 🗂️ Project Structure

```
HybridSearch Bench/
│
├── 📊  app.py                # 3-tab Streamlit dashboard
│                             #   Tab 1: Query & Compare (side-by-side chunk cards)
│                             #   Tab 2: Eval Dashboard (RAGAS radar + bar charts)
│                             #   Tab 3: History (metric trend lines)
│
├── ⌨️  cli.py                # Click CLI — ingest / query / evaluate / benchmark / serve
│
├── 🗄️  chroma_store.py       # ChromaDB client — upsert, query, collection lifecycle
├── ✂️  chunker.py            # RecursiveCharacterTextSplitter + md5 dedup
├── ⚙️  config.py             # Hard constants (separator hierarchy, batch sizes)
├── 🔢  embedder.py           # Ollama embedding wrapper (batch=32, 768-dim)
│
├── 📐  interfaces.py         # Protocol definitions — Retriever, Embedder, Evaluator
├── 🚨  exceptions.py         # Custom exception hierarchy (IngestionError, RetrievalError…)
├── 📝  logging.py            # Structured JSON logging with contextual fields
├── ⚙️  settings.py           # Pydantic BaseSettings — all config from .env
│
├── 🔍  retrieval.py          # BM25Strategy / VectorStrategy / HybridStrategy (RRF)
├── 📈  metrics.py            # MRR, MAP, NDCG@k — implemented from first principles
├── 🧪  evaluation.py         # RAGAS runner via LangchainLLMWrapper(ChatOllama)
├── 🔄  pipeline.py           # HybridSearchPipeline — orchestrates ingest→retrieve→eval
│
├── 🐳  Dockerfile            # Multi-stage build (builder + runtime, ~200MB final)
├── 🐳  docker-compose.yml    # app + ollama + ollama-init (auto model pull)
├── 🔧  Makefile              # install / lint / test / run / docker-up / docker-down
├── 📋  pyproject.toml        # Package config, entry points, tool configs
├── 📦  requirements.txt
├── 📦  requirements-dev.txt
├── 🔑  .env.example
└── 🧪  tests/
    ├── conftest.py           # Shared fixtures
    ├── test_metrics.py       # Pure-function IR metric tests (edge + known values)
    ├── test_chunker.py       # Chunk size, overlap, dedup assertions
    └── test_retrieval.py     # RRF math unit tests with mock ChromaDB
```

---

## 🚀 Quickstart

### Option A — Docker *(recommended, one command)*

```bash
# 1. Clone and configure
git clone https://github.com/YOUR_USERNAME/HybridSearch-Bench.git
cd HybridSearch-Bench
cp .env.example .env

# 2. Start the full stack (app + Ollama + auto model pull ~4 GB first run)
make docker-up

# 3. Open the dashboard
open http://localhost:8501
```

> The `ollama-init` service automatically pulls `llama3.2` and `nomic-embed-text` on first boot. Subsequent starts are instant.

---

### Option B — Local Python

**Prerequisites:** [Ollama](https://ollama.com/download) installed and running.

```bash
# 1. Pull models
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Install project
git clone https://github.com/YOUR_USERNAME/HybridSearch-Bench.git
cd HybridSearch-Bench
cp .env.example .env
make install          # pip install -e ".[dev]"

# 3. Ingest your PDFs
mkdir -p data/pdfs
# drop your PDFs into data/pdfs/
make ingest           # python -m cli ingest --pdf-dir data/pdfs

# 4. Launch dashboard
make run              # streamlit run app.py → http://localhost:8501
```

---

## ⌨️ CLI Reference

```bash
python -m cli [COMMAND] [OPTIONS]
```

| Command | What it does | Key flags |
|---|---|---|
| `ingest` | PDF → chunks → embeddings → ChromaDB | `--pdf-dir`, `--collection` |
| `query` | Single question across strategies | `--question`, `--strategy [bm25\|vector\|hybrid]`, `--top-k` |
| `evaluate` | RAGAS + IR metrics on JSONL question set | `--questions-file`, `--output` |
| `benchmark` | All 3 strategies, side-by-side comparison table | `--questions-file`, `--output` |
| `serve` | Launch Streamlit dashboard | — |

**Examples:**

```bash
# Ingest a corpus
python -m cli ingest --pdf-dir data/pdfs --collection eeg_papers

# Single hybrid query
python -m cli query --question "What is attention mechanism?" --strategy hybrid --top-k 5

# Full evaluation run
python -m cli evaluate --questions-file eval_set.jsonl --output results/eval.csv

# Strategy benchmark
python -m cli benchmark --questions-file eval_set.jsonl --output results/benchmark.csv
```

**JSONL format for `--questions-file`:**
```json
{"question": "What is RAG?", "ground_truth": "Retrieval-Augmented Generation..."}
{"question": "How does BM25 work?", "ground_truth": "BM25 ranks documents..."}
```

---

## 📊 Evaluation Metrics

### RAGAS (Answer Quality)

| Metric | What it measures | Range |
|---|---|---|
| **Faithfulness** | Does the answer stay grounded in retrieved contexts? No hallucination. | 0 → 1 |
| **Answer Relevancy** | Is the answer on-topic and directly addresses the question? | 0 → 1 |
| **Context Precision** | Are the most relevant chunks ranked highest in the retrieved set? | 0 → 1 |

### Custom IR Metrics (Retrieval Quality)

These are implemented **from first principles** in `metrics.py` — no external IR library.

**Mean Reciprocal Rank (MRR)**
```
MRR = (1/|Q|) × Σ  1 / rank_first_relevant(q)
```

**Mean Average Precision (MAP)**
```
AP(q) = Σₖ [ P(k) × rel(k) ] / |relevant|
MAP   = (1/|Q|) × Σ AP(q)
```

**Normalized Discounted Cumulative Gain (NDCG@k)**
```
DCG@k  = Σᵢ₌₁ᵏ  relᵢ / log₂(i + 1)
NDCG@k = DCG@k / IDCG@k          (IDCG = perfect ranking DCG)
```

Reported at `k ∈ {1, 3, 5, 10}`.

---

## ⚙️ Configuration

Copy `.env.example` → `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | LLM for answer generation |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model (768-dim) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `hybrid_bench` | Collection name |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Chunks retrieved per strategy |
| `RRF_K` | `60` | RRF constant (Cormack 2009 default) |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `LOG_FORMAT` | `json` | `json` for structured, `text` for dev |

---

## 🧪 Tests

```bash
make test
# → pytest tests/ -v --cov=. --cov-report=term-missing
```

```
tests/test_metrics.py        # Pure IR metric correctness (edge cases + known values)
tests/test_chunker.py        # Chunk size bounds, overlap, dedup
tests/test_retrieval.py      # RRF fusion math, score normalization
```

```bash
make lint     # ruff check . && mypy .
make format   # ruff format .
```

---

## 🛠️ Makefile Targets

```bash
make install      # pip install -e ".[dev]"
make run          # streamlit run app.py
make ingest       # ingest PDFs from data/pdfs/
make test         # pytest with coverage
make lint         # ruff + mypy
make format       # ruff format
make docker-up    # docker compose up --build -d
make docker-down  # docker compose down
make clean        # remove __pycache__, .pytest_cache
```

---

## 🧱 Tech Stack

| Layer | Technology | Justification |
|---|---|---|
| **LLM** | Ollama (`llama3.2`) | Fully local, no API cost, swappable |
| **Embeddings** | `nomic-embed-text` via Ollama | 768-dim, outperforms MiniLM on retrieval benchmarks |
| **Sparse retrieval** | `rank-bm25` | Gold standard for keyword-heavy technical corpora |
| **Vector store** | ChromaDB (HNSW, cosine) | Embedded, no server, production-serializable |
| **Fusion** | Reciprocal Rank Fusion | Theoretically grounded, parameter-light (Cormack 2009) |
| **Evaluation** | RAGAS + custom metrics | End-to-end RAG scoring + retrieval-level IR metrics |
| **Settings** | Pydantic v2 `BaseSettings` | Type-safe, `.env`-driven, validated at startup |
| **CLI** | Click | Composable, testable, `--help` autodoc |
| **Dashboard** | Streamlit | Interactive eval visualization without frontend overhead |
| **Containerization** | Docker multi-stage + Compose | Reproducible, lean runtime image (~200 MB) |
| **Testing** | Pytest + coverage | Pure-function unit tests, no mocks for IR metrics |

---

## 📁 Dataset Ideas

Works with any PDF corpus. Recommended for showcasing:

- 📄 **ArXiv papers** — single topic (e.g. EEG classification, medical AI, transformers)
- 📚 **Technical documentation** — framework docs, API references
- 🏥 **Clinical guidelines** — structured medical PDFs
- 📰 **Research reports** — dense, multi-section documents

> Pro tip: Use 20–50 papers on a narrow topic for meaningful retrieval contrast between BM25 and vector strategies.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Ensure `make lint` and `make test` pass
4. Open a PR with a clear description

---

<div align="center">

**Built with precision for the ML engineering portfolio.**  
*If this helped you, a ⭐ goes a long way.*

MIT License © 2026

</div>
