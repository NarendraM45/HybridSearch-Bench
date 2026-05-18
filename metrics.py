"""
metrics.py — Custom IR metrics from first principles.

Implements MRR, MAP, and NDCG@k.
No external IR libraries are used.
"""

from typing import List, Set, Dict
import math

def reciprocal_rank(relevant_ids: Set[str], retrieved_ids: List[str]) -> float:
    """
    MRR component for a single query.
    Formula: RR = 1 / rank_i, where rank_i is the rank of the first relevant document.
    Returns 0.0 if no relevant document is found.
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def mean_reciprocal_rank(queries: List[Dict]) -> float:
    """
    MRR over query set. Each query dict: {relevant_ids, retrieved_ids}
    Formula: MRR = (1 / |Q|) * sum_{q=1}^{|Q|} RR(q)
    """
    if not queries:
        return 0.0
    total_rr = sum(reciprocal_rank(q['relevant_ids'], q['retrieved_ids']) for q in queries)
    return total_rr / len(queries)

def precision_at_k(relevant_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    """
    Precision at k.
    Formula: P@k = |{relevant documents} \cap {retrieved documents at top k}| / k
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k

def average_precision(relevant_ids: Set[str], retrieved_ids: List[str]) -> float:
    """
    AP for a single query.
    Formula: AP = sum_{k=1}^n (P(k) * rel(k)) / |relevant documents|
    where rel(k) is an indicator function equaling 1 if the item at rank k is a relevant document, zero otherwise.
    """
    if not relevant_ids:
        return 0.0
        
    ap = 0.0
    relevant_count = 0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            relevant_count += 1
            ap += relevant_count / (i + 1)
            
    return ap / len(relevant_ids)

def mean_average_precision(queries: List[Dict]) -> float:
    """
    MAP over query set.
    Formula: MAP = (1 / |Q|) * sum_{q=1}^{|Q|} AP(q)
    """
    if not queries:
        return 0.0
    total_ap = sum(average_precision(q['relevant_ids'], q['retrieved_ids']) for q in queries)
    return total_ap / len(queries)

def ndcg_at_k(relevant_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    """
    Binary relevance NDCG@k.
    Formula:
    DCG@k = sum_{i=1}^k rel_i / log_2(i + 1)
    IDCG@k = sum_{i=1}^{min(k, |REL|)} 1 / log_2(i + 1)
    NDCG@k = DCG@k / IDCG@k
    """
    if k == 0 or not relevant_ids:
        return 0.0
        
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
            
    idcg = 0.0
    ideal_relevant_count = min(k, len(relevant_ids))
    for i in range(ideal_relevant_count):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def retrieval_report(queries: List[Dict], k_values: List[int] = [1,3,5,10]) -> Dict:
    """
    Returns full report dict: {MRR, MAP, NDCG@k for each k}
    """
    report = {
        "MRR": mean_reciprocal_rank(queries),
        "MAP": mean_average_precision(queries),
    }
    for k in k_values:
        ndcg_sum = sum(ndcg_at_k(q['relevant_ids'], q['retrieved_ids'], k) for q in queries)
    return report

def reranking_delta(before_queries: List[Dict], after_queries: List[Dict], k: int = 5) -> float:
    """
    Computes the improvement in mean NDCG@k from reranking.
    """
    if not before_queries or not after_queries:
        return 0.0
    ndcg_before = sum(ndcg_at_k(q['relevant_ids'], q['retrieved_ids'], k) for q in before_queries) / len(before_queries)
    ndcg_after = sum(ndcg_at_k(q['relevant_ids'], q['retrieved_ids'], k) for q in after_queries) / len(after_queries)
    return ndcg_after - ndcg_before
