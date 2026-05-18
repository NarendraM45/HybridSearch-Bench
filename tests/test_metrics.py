import pytest
import math
from metrics import (
    reciprocal_rank,
    mean_reciprocal_rank,
    precision_at_k,
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    retrieval_report
)

def test_reciprocal_rank():
    # Empty retrieved list -> 0
    assert reciprocal_rank({"doc1"}, []) == 0.0
    
    # Perfect retrieval
    assert reciprocal_rank({"doc1"}, ["doc1", "doc2"]) == 1.0
    
    # Wrong order
    assert reciprocal_rank({"doc2"}, ["doc1", "doc2", "doc3"]) == 0.5
    
    # Not found
    assert reciprocal_rank({"doc99"}, ["doc1", "doc2"]) == 0.0

def test_mean_reciprocal_rank():
    queries = [
        {"relevant_ids": {"doc1"}, "retrieved_ids": ["doc1", "doc2"]}, # RR = 1.0
        {"relevant_ids": {"doc3"}, "retrieved_ids": ["doc1", "doc2"]}, # RR = 0.0
        {"relevant_ids": {"doc2"}, "retrieved_ids": ["doc1", "doc2"]}  # RR = 0.5
    ]
    assert mean_reciprocal_rank(queries) == pytest.approx(0.5)
    assert mean_reciprocal_rank([]) == 0.0

def test_precision_at_k():
    rel = {"doc1", "doc3"}
    ret = ["doc1", "doc2", "doc3", "doc4"]
    
    assert precision_at_k(rel, ret, 0) == 0.0
    assert precision_at_k(rel, ret, 1) == 1.0 # doc1
    assert precision_at_k(rel, ret, 2) == 0.5 # doc1, doc2
    assert precision_at_k(rel, ret, 3) == 2/3 # doc1, doc2, doc3

def test_average_precision():
    rel = {"doc1", "doc3", "doc5"}
    ret = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    
    # P(1) = 1/1 = 1.0
    # P(3) = 2/3
    # P(5) = 3/5
    # AP = (1.0 + 2/3 + 3/5) / 3 = (15/15 + 10/15 + 9/15) / 3 = 34 / 45 ≈ 0.7555
    expected_ap = (1.0 + 2/3 + 3/5) / 3
    assert average_precision(rel, ret) == pytest.approx(expected_ap)
    
    # No relevant found
    assert average_precision(rel, ["doc9", "doc8"]) == 0.0

def test_mean_average_precision():
    queries = [
        {"relevant_ids": {"doc1"}, "retrieved_ids": ["doc1"]}, # AP = 1.0
        {"relevant_ids": {"doc2"}, "retrieved_ids": ["doc1", "doc2"]}, # AP = 0.5
    ]
    assert mean_average_precision(queries) == 0.75
    assert mean_average_precision([]) == 0.0

def test_ndcg_at_k():
    rel = {"doc1", "doc3"}
    ret = ["doc1", "doc2", "doc3", "doc4"]
    
    # DCG@3:
    # i=0: doc1 is relevant -> 1/log2(2) = 1.0
    # i=1: doc2 not -> 0
    # i=2: doc3 relevant -> 1/log2(4) = 0.5
    # dcg@3 = 1.5
    
    # IDCG@3 (ideal docs: min(3, |rel|) = 2)
    # i=0 -> 1.0
    # i=1 -> 1/log2(3) ≈ 0.6309
    # idcg@3 = 1 + 1/log2(3)
    
    expected_ndcg_3 = 1.5 / (1.0 + 1.0/math.log2(3))
    assert ndcg_at_k(rel, ret, 3) == pytest.approx(expected_ndcg_3)
    
    # Perfect retrieval -> 1.0
    assert ndcg_at_k({"doc1"}, ["doc1", "doc2"], 1) == 1.0
    
    # Empty -> 0.0
    assert ndcg_at_k({"doc1"}, [], 3) == 0.0

def test_retrieval_report():
    queries = [
        {"relevant_ids": {"doc1"}, "retrieved_ids": ["doc1", "doc2"]}
    ]
    report = retrieval_report(queries, k_values=[1, 5])
    assert "MRR" in report
    assert "MAP" in report
    assert "NDCG@1" in report
    assert "NDCG@5" in report
    assert report["MRR"] == 1.0
    assert report["NDCG@1"] == 1.0
