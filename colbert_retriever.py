import logging
import os
from typing import List, Dict

logger = logging.getLogger(__name__)

class ColBERTRetriever:
    """
    Token-level late interaction retrieval via RAGatouille.
    """
    
    def __init__(self, index_path: str = "./colbert_index"):
        self.index_path = index_path
        self.RAG = None
        self._loaded = False
        
    def _load_model(self):
        if self._loaded: return
        try:
            from ragatouille import RAGPretrainedModel
        except ImportError:
            logger.warning("ragatouille not installed. ColBERT disabled.")
            return

        index_name = os.path.basename(self.index_path)
        index_dir = os.path.dirname(self.index_path)

        # Check if index exists
        target_path = os.path.join(index_dir, "colbert", "indexes", index_name)
        if os.path.exists(target_path):
            logger.info(f"Loading existing ColBERT index from {target_path}")
            self.RAG = RAGPretrainedModel.from_index(target_path)
        else:
            logger.info("Loading pretrained ColBERT model for first-time use")
            self.RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            
        self._loaded = True

    def index(self, documents: List[str], document_ids: List[str]) -> None:
        """Build ColBERT index from document texts."""
        self._load_model()
        if not self.RAG: return
        
        index_name = os.path.basename(self.index_path)
        index_dir = os.path.dirname(self.index_path)
        
        # RAGatouille handles chunking internally for the index, but we can feed it chunks
        self.RAG.index(
            collection=documents,
            document_ids=document_ids,
            index_name=index_name,
            max_document_length=256,
            split_documents=True
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Returns list of {doc_id, text, score} dicts."""
        self._load_model()
        if not self.RAG: return []
        
        try:
            results = self.RAG.search(query, k=top_k)
            # RAGatouille returns list of dicts: {"content", "score", "rank", "document_id"}
            out = []
            for r in results:
                out.append({
                    "text": r["content"],
                    "score": r["score"],
                    "rank": r["rank"],
                    "metadata": {"chunk_id": r["document_id"]},
                    "retriever": "colbert"
                })
            return out
        except Exception as e:
            logger.error(f"ColBERT search failed: {e}")
            return []
