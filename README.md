# HybridSearch Bench

Production-grade RAG evaluation framework with hybrid retrieval (BM25 + Dense Vector + RRF fusion) and automatic RAGAS-based evaluation. Fully local — no paid APIs, runs on Ollama.

## Architecture Diagram

```
         [ PDFs ]
            |
            v
     +--------------+
     |   Chunking   | (Recursive)
     +--------------+
            |
    +-------+-------+
    |               |
    v               v
 [ BM25 ]      [ Vector ] (ChromaDB + Nomic Embed)
    |               |
    +-------+-------+
            |
            v
     +--------------+
     |  RRF Fusion  | (Hybrid)
     +--------------+
            |
            v
     +--------------+
     |   Ollama     | (llama3.2)
     | Generation   |
     +--------------+
            |
            v
     +--------------+
     |    RAGAS     | (Faithfulness, Relevancy, Precision)
     |  Evaluation  |
     +--------------+
```

## Features

- **Hybrid Retrieval**: Combines sparse (BM25) and dense (Vector) retrieval using Reciprocal Rank Fusion (RRF).
- **Local LLM Integration**: Fully powered by Ollama (`llama3.2` and `nomic-embed-text`) - no API keys required.
- **First-Principles Metrics**: Custom IR metrics implemented from scratch (MRR, MAP, NDCG@k).
- **RAGAS Evaluation**: Automatic assessment of context precision, answer relevancy, and faithfulness.
- **Production-Ready**: Click-based CLI, Docker-compose setup, and Streamlit dashboard.

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| UI | Streamlit | Rapid prototyping and interactive data exploration |
| LLM Backend | Ollama | Run powerful LLMs locally without paid APIs |
| Vector Store | ChromaDB | Lightweight, embedded, fast similarity search |
| Evaluation | RAGAS | Industry standard for RAG metrics |
| CLI | Click | Clean, robust, production-ready command line interface |

## Project Structure

```
├── app.py               # Streamlit dashboard
├── cli.py               # Click CLI entrypoint
├── chroma_store.py      # ChromaDB client interface
├── chunker.py           # Recursive chunking logic
├── config.py            # Hardcoded settings and constants
├── embedder.py          # Ollama embedding wrapper
├── evaluation.py        # RAGAS metrics and LLM generation
├── metrics.py           # Custom IR metrics from first principles
├── pipeline.py          # Ingestion pipeline logic
├── retrieval.py         # BM25, Vector, and Hybrid retrieval
├── settings.py          # Pydantic BaseSettings config
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Compose stack for app and ollama
├── Makefile             # Helper targets
└── tests/               # Pytest suite
```

## Quickstart (Docker — recommended)

1. Ensure Docker and Docker Compose are installed.
2. Run the stack:
   ```bash
   make docker-up
   ```
3. Wait for the `ollama-init` service to pull models.
4. Access the Streamlit dashboard at `http://localhost:8501`.

## Quickstart (Local)

1. Install Ollama locally and start the service.
2. Pull required models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Install Python dependencies:
   ```bash
   make install
   ```
4. Run the application:
   ```bash
   make run
   ```

## CLI Reference

### Ingest
Ingests a directory of PDFs into ChromaDB.
```bash
python -m cli ingest --pdf-dir data/pdfs --collection hybrid_bench
```

### Query
Queries the index using a specific strategy.
```bash
python -m cli query --question "What is RAG?" --strategy hybrid --top-k 5
```

### Evaluate
Evaluates a set of questions from a JSONL file and outputs a CSV report.
```bash
python -m cli evaluate --questions-file eval_set.jsonl --output results.csv
```

### Benchmark
Runs all strategies on a question set and outputs a comparison table.
```bash
python -m cli benchmark --questions-file eval_set.jsonl --output benchmark.csv
```

### Serve
Launches the Streamlit app.
```bash
python -m cli serve
```

## Evaluation Metrics

The evaluation utilizes two main paradigms:

**1. RAGAS Metrics**
- **Faithfulness**: Measures if the answer is hallucinated.
- **Answer Relevancy**: Measures if the answer directly addresses the question.
- **Context Precision**: Measures if relevant contexts are ranked higher.

**2. Custom IR Metrics**
- **MRR (Mean Reciprocal Rank)**: 
  $$ MRR = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{rank_i} $$
- **MAP (Mean Average Precision)**:
  $$ MAP = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{\sum_{k=1}^n (P(k) \times rel(k))}{|rel|} $$
- **NDCG@k (Normalized Discounted Cumulative Gain)**:
  $$ DCG@k = \sum_{i=1}^k \frac{rel_i}{\log_2(i + 1)} \quad \Rightarrow \quad NDCG@k = \frac{DCG@k}{IDCG@k} $$

## Configuration

Available `.env` variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama chat model |
| `OLLAMA_EMBED_MODEL`| `nomic-embed-text` | Ollama embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Path to store Chroma vectors |
| `CHROMA_COLLECTION_NAME`| `hybrid_bench` | Collection name |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Chunk overlap |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `RRF_K` | `60` | Constant for Reciprocal Rank Fusion |

## Running Tests

Run the test suite with coverage:
```bash
make test
```

## Contributing

PRs welcome! Ensure `make lint` and `make test` pass.

## License

MIT
