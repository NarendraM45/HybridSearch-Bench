import pytest
from unittest.mock import MagicMock
from reranker import CrossEncoderReranker


@pytest.fixture(autouse=True)
def clear_reranker_cache():
    CrossEncoderReranker.clear_model_cache()
    yield
    CrossEncoderReranker.clear_model_cache()


@pytest.fixture
def mock_cross_encoder():
    mock = MagicMock()
    # Return mock scores for 3 pairs
    mock.predict.return_value = [0.1, 0.9, 0.5]
    return mock

def test_reranker_ordering_and_scores(monkeypatch, mock_cross_encoder):
    monkeypatch.setattr("reranker.CrossEncoder", lambda *args, **kwargs: mock_cross_encoder)
    
    reranker = CrossEncoderReranker(device="cpu")
    chunks = [
        {"text": "Chunk A", "score": 1.0},
        {"text": "Chunk B", "score": 2.0},
        {"text": "Chunk C", "score": 3.0}
    ]
    
    result = reranker.rerank("Query", chunks)
    
    assert len(result) == 3
    # B should be first (score 0.9), C second (0.5), A third (0.1)
    assert result[0]["text"] == "Chunk B"
    assert result[0]["rerank_score"] == 0.9
    assert result[0]["retrieval_score"] == 2.0
    
    assert result[1]["text"] == "Chunk C"
    assert result[2]["text"] == "Chunk A"

def test_rerank_with_threshold(monkeypatch, mock_cross_encoder):
    monkeypatch.setattr("reranker.CrossEncoder", lambda *args, **kwargs: mock_cross_encoder)
    reranker = CrossEncoderReranker(device="cpu")
    chunks = [
        {"text": "A", "score": 1.0},
        {"text": "B", "score": 1.0},
        {"text": "C", "score": 1.0}
    ]
    
    # 0.9 and 0.5 pass, 0.1 fails
    result = reranker.rerank_with_threshold("Query", chunks, threshold=0.4)
    assert len(result) == 2
    assert result[0]["text"] == "B"
    assert result[1]["text"] == "C"


def test_reranker_reuses_cached_model(monkeypatch, mock_cross_encoder):
    constructor = MagicMock(return_value=mock_cross_encoder)
    monkeypatch.setattr("reranker.CrossEncoder", constructor)

    first = CrossEncoderReranker(device="cpu")
    second = CrossEncoderReranker(device="cpu")
    chunks = [{"text": "Chunk A", "score": 1.0}]

    first.rerank("Query", chunks)
    second.rerank("Query", chunks)

    assert constructor.call_count == 1
