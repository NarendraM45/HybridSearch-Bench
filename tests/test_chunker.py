import pytest
from chunker import RecursiveChunker, SemanticChunker
from config import CHUNK_SIZE, CHUNK_OVERLAP
from unittest.mock import patch
from pathlib import Path

def test_recursive_chunker_size_and_overlap():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    text = "A" * 200
    docs = [{"id": "doc1", "text": text, "metadata": {}}]
    
    chunks = chunker.chunk(docs)
    assert len(chunks) > 0
    # Each chunk should not exceed chunk_size + minor tolerance for overlap logic in LangChain RecursiveCharacterTextSplitter
    for chunk in chunks:
        assert len(chunk["text"]) <= 100
        
    # Check overlap roughly by ensuring the total length of chunks is larger than original
    total_len = sum(len(c["text"]) for c in chunks)
    assert total_len > 200

def test_chunk_deduplication():
    # If the chunker implementation removes exact duplicates
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    text1 = "This is a unique sentence."
    text2 = "This is a unique sentence."
    docs = [
        {"id": "doc1", "text": text1, "metadata": {}},
        {"id": "doc2", "text": text2, "metadata": {}}
    ]
    
    chunks = chunker.chunk(docs)
    texts = [c["text"] for c in chunks]
    assert len(set(texts)) == len(texts), "Chunks should be deduplicated"

@patch('langchain_community.document_loaders.PyPDFLoader')
def test_pypdf_loader_mock(mock_loader_class, tmp_path):
    # Mocking the PyPDFLoader behavior
    class MockLoader:
        def __init__(self, file_path):
            self.file_path = file_path
        def load(self):
            class DummyDoc:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata
            return [DummyDoc("Fake PDF Content", {"page": 1})]
            
    mock_loader_class.side_effect = MockLoader
    
    from ingestion import load_pdf
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("dummy binary content")
    
    docs = load_pdf(str(pdf_file))
    assert len(docs) == 1
    assert docs[0]["text"] == "Fake PDF Content"
