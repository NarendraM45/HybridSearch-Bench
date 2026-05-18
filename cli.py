import sys
import json
import logging as python_logging
import click
import pandas as pd
from pathlib import Path

from settings import get_settings
from logging import get_logger
from pipeline import IngestionPipeline
from retrieval import run_all_strategies, build_bm25_index
from evaluation import evaluate_all_strategies

log = get_logger(__name__)

# Ensure python-dotenv is loaded
from dotenv import load_dotenv
load_dotenv()


@click.group()
def main():
    """HybridSearch Bench CLI. Production-grade RAG evaluation."""
    pass


@main.command()
@click.option('--pdf-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help="Directory containing PDF files.")
@click.option('--collection', type=str, required=True, help="ChromaDB collection name to ingest into.")
def ingest(pdf_dir: str, collection: str):
    """Run ingestion pipeline on a directory of PDFs."""
    from ingestion import ingest_pdf
    from chroma_store import ChromaStore
    
    pdf_path = Path(pdf_dir)
    pdf_files = list(pdf_path.glob('*.pdf'))
    
    if not pdf_files:
        click.secho(f"No PDF files found in {pdf_dir}", fg="red", err=True)
        sys.exit(1)
        
    s = get_settings()
    s.collection_name = collection
    
    # Actually use the pipeline
    pipeline = IngestionPipeline()
    try:
        pipeline.run(pdf_path)
        click.secho("Ingestion complete.", fg="green")
    except Exception as e:
        log.exception("Ingestion failed")
        click.secho(str(e), fg="red", err=True)
        sys.exit(1)


@main.command()
@click.option('--question', type=str, required=True, help="Question to ask.")
@click.option('--strategy', type=click.Choice(['bm25', 'vector', 'hybrid']), default='hybrid', help="Retrieval strategy.")
@click.option('--top-k', type=int, default=5, help="Number of documents to retrieve.")
def query(question: str, strategy: str, top_k: int):
    """Retrieve documents using a specific strategy."""
    from chroma_store import ChromaStore
    from config import CHUNK_SIZE, CHUNK_OVERLAP
    
    try:
        store = ChromaStore()
        all_docs = store.get_all()
        if not all_docs:
            click.secho("No documents found in store. Did you ingest?", fg="red", err=True)
            sys.exit(1)
            
        bm25_idx = build_bm25_index(all_docs)
        results = run_all_strategies(question, bm25_idx, all_docs, store, top_k=top_k)
        
        click.secho(f"--- Results for {strategy.upper()} ---", fg="green")
        for doc in results[strategy]:
            score = doc.get("rrf_score") or doc.get("score")
            click.echo(f"Score: {score:.4f} | Chunk: {doc['text'][:100]}...")
    except Exception as e:
        log.exception("Query failed")
        click.secho(str(e), fg="red", err=True)
        sys.exit(1)


@main.command()
@click.option('--questions-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help="JSONL file with questions and optional ground_truth.")
@click.option('--output', type=click.Path(), required=True, help="Path to output CSV results.")
def evaluate(questions_file: str, output: str):
    """Run evaluation on a question set."""
    from chroma_store import ChromaStore
    
    try:
        store = ChromaStore()
        all_docs = store.get_all()
        bm25_idx = build_bm25_index(all_docs)
        
        results_list = []
        with open(questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                q = data['question']
                gt = data.get('ground_truth')
                
                ret_results = run_all_strategies(q, bm25_idx, all_docs, store, top_k=5)
                eval_res = evaluate_all_strategies(q, ret_results, ground_truth=gt)
                
                for strat, metrics in eval_res.items():
                    row = {"question": q, "strategy": strat, "answer": metrics.get("answer")}
                    row.update({k: v for k, v in metrics.items() if k != "answer"})
                    results_list.append(row)
                    
        df = pd.DataFrame(results_list)
        df.to_csv(output, index=False)
        click.secho(f"Evaluation complete. Results saved to {output}", fg="green")
    except Exception as e:
        log.exception("Evaluation failed")
        click.secho(str(e), fg="red", err=True)
        sys.exit(1)


@main.command()
@click.option('--questions-file', type=click.Path(exists=True), required=True, help="JSONL file with questions.")
@click.option('--output', type=click.Path(), default="benchmark_results.csv", help="Output path for benchmark comparison.")
def benchmark(questions_file: str, output: str):
    """Run all three strategies on a question set and output comparison table."""
    # Similar to evaluate but just focuses on comparison
    click.echo("Running benchmark...")
    # For now we use evaluate logic but summarize
    # Since they do the same under the hood
    from click.testing import CliRunner
    click.echo("Delegating to evaluate...")
    sys.argv = ['cli', 'evaluate', '--questions-file', questions_file, '--output', output]
    main()


@main.command()
def serve():
    """Launch the Streamlit app."""
    import subprocess
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except Exception as e:
        click.secho(f"Failed to start streamlit: {e}", fg="red", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
