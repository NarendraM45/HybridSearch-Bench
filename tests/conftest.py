import pytest
from typing import List, Dict

from interfaces import Document

@pytest.fixture
def sample_chunks() -> List[str]:
    return [
        "This is the first test chunk about AI.",
        "This is the second test chunk, talking about RAG.",
        "The third chunk discusses BM25 retrieval."
    ]

@pytest.fixture
def sample_documents() -> List[Document]:
    return [
        {"id": "doc1", "text": "This is the first test chunk about AI.", "metadata": {"page": 1, "chunk_id": "chunk_0"}},
        {"id": "doc2", "text": "This is the second test chunk, talking about RAG.", "metadata": {"page": 1, "chunk_id": "chunk_1"}},
        {"id": "doc3", "text": "The third chunk discusses BM25 retrieval.", "metadata": {"page": 2, "chunk_id": "chunk_2"}},
    ]

@pytest.fixture
def mock_chroma_collection():
    class MockCollection:
        def __init__(self):
            self.docs = []
            
        def add(self, ids, documents, metadatas):
            for i, d, m in zip(ids, documents, metadatas):
                self.docs.append({"id": i, "document": d, "metadata": m})
                
        def get(self):
            return {
                "ids": [d["id"] for d in self.docs],
                "documents": [d["document"] for d in self.docs],
                "metadatas": [d["metadata"] for d in self.docs],
            }
            
        def query(self, query_texts, n_results):
            # very simple mock that just returns the first n docs
            n = min(n_results, len(self.docs))
            res = self.docs[:n]
            return {
                "ids": [[d["id"] for d in res]],
                "documents": [[d["document"] for d in res]],
                "metadatas": [[d["metadata"] for d in res]],
                "distances": [[0.5] * n]
            }
    
    return MockCollection()
