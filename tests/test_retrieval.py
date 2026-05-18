import pytest
from retrieval import reciprocal_rank_fusion

def test_rrf_fusion_math():
    # RRF formula: 1 / (k + rank)
    # Assume k=60
    # Two strategies: "bm25" and "vector"
    # Overlap on "doc1"
    
    results = {
        "bm25": [
            {"id": "doc1", "text": "...", "metadata": {}, "score": 1.5, "rank": 1},
            {"id": "doc2", "text": "...", "metadata": {}, "score": 1.2, "rank": 2}
        ],
        "vector": [
            {"id": "doc3", "text": "...", "metadata": {}, "score": 0.9, "rank": 1},
            {"id": "doc1", "text": "...", "metadata": {}, "score": 0.8, "rank": 2}
        ]
    }
    
    fused = reciprocal_rank_fusion(results, rrf_k=60)
    
    # doc1 is in both: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.01639 + 0.01612 = 0.0325
    # doc2 is in bm25 only: 1/(60+2) = 1/62 ≈ 0.01612
    # doc3 is in vector only: 1/(60+1) = 1/61 ≈ 0.01639
    
    # The normalization divides by max score (0.0325)
    
    assert len(fused) == 3
    
    # Max score doc is doc1
    assert fused[0]["id"] == "doc1"
    assert fused[0]["rrf_score"] == 1.0 # Normalized
    
    # Verify scores are normalized between 0 and 1
    for doc in fused:
        assert 0.0 <= doc["rrf_score"] <= 1.0

def test_retrieval_all_strategies(mock_chroma_collection):
    # Test run_all_strategies with the mock collection
    from retrieval import run_all_strategies, build_bm25_index
    
    # Setup mock
    docs = [
        {"id": "d1", "text": "RAG test doc", "metadata": {"chunk_id": "c1"}},
        {"id": "d2", "text": "another doc", "metadata": {"chunk_id": "c2"}}
    ]
    mock_chroma_collection.add(
        [d["id"] for d in docs],
        [d["text"] for d in docs],
        [d["metadata"] for d in docs]
    )
    
    bm25_idx = build_bm25_index(docs)
    
    res = run_all_strategies("RAG", bm25_idx, docs, mock_chroma_collection, top_k=2)
    
    assert "bm25" in res
    assert "vector" in res
    assert "hybrid" in res
    
    assert len(res["bm25"]) > 0
    assert len(res["vector"]) > 0
    assert len(res["hybrid"]) > 0
