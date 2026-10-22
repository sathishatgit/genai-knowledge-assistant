"""Reciprocal Rank Fusion (RRF) module.

================================================================================
EDUCATIONAL CONCEPTS FOR STUDENTS & WORKSHOP ATTENDEES:
================================================================================
1. WHY BM25 (Lexical Search)?
   - BM25 excels at exact keyword matching, acronyms, product numbers, and technical
     identifiers. However, it fails when users ask questions using synonyms or
     different phrasing (the "vocabulary mismatch" problem).

2. WHY VECTOR SEARCH (Dense Semantic Search)?
   - Vector search (using Jina AI embeddings) understands the conceptual meaning
     and context of words. It finds relevant documents even when zero identical
     words match. However, it can struggle with exact acronyms, rare names, or numbers.

3. WHY COMBINE THEM (Hybrid Search)?
   - Neither search system is perfect on its own. Hybrid search harnesses the
     complementary strengths of both: exact precision from BM25 and conceptual
     understanding from embeddings.

4. WHAT IS RECIPROCAL RANK FUSION (RRF)?
   - BM25 scores (unbounded positive numbers) and Vector cosine similarities (0 to 1)
     live on completely different numerical scales, so adding raw scores directly
     leads to unbalanced bias.
   - RRF solves this by ignoring raw score values and looking ONLY at item ranks:
     
         RRF_Score(doc) = Σ [ 1 / (k + rank(doc)) ]
     
     where 'k' is a smoothing constant (standard default is 60), and 'rank' is the
     1-indexed position of the document in each search result list.
   - If a document appears near the top in BOTH searches, its RRF score is amplified!
================================================================================
"""

from typing import List, Dict, Any
from src.config import RRF_K


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = RRF_K
) -> List[Dict[str, Any]]:
    """Combine multiple ranked search result lists using Reciprocal Rank Fusion.
    
    Args:
        ranked_lists: A list of ranked result lists (e.g. [bm25_results, vector_results]).
                      Each result must contain at least 'id', 'text', and 'metadata'.
        k: Smoothing constant (default: 60). Higher k reduces the penalty for lower ranks.
        
    Returns:
        A unified list of documents sorted in descending order of their fused RRF score.
    """
    rrf_scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Dict[str, Any]] = {}
    rank_details: Dict[str, Dict[str, int]] = {}

    for list_index, result_list in enumerate(ranked_lists):
        list_name = f"list_{list_index}"
        for rank_zero_indexed, item in enumerate(result_list):
            doc_id = item["id"]
            rank_1_indexed = rank_zero_indexed + 1  # 1-based rank

            # Compute reciprocal rank addition: 1 / (k + rank)
            contribution = 1.0 / (k + rank_1_indexed)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contribution

            # Cache the document details
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = {
                    "id": doc_id,
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {})
                }
                rank_details[doc_id] = {}

            rank_details[doc_id][list_name] = rank_1_indexed

    # Sort documents by their combined RRF score in descending order
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)

    fused_results: List[Dict[str, Any]] = []
    for doc_id in sorted_doc_ids:
        doc_entry = dict(doc_lookup[doc_id])
        doc_entry["rrf_score"] = round(rrf_scores[doc_id], 6)
        doc_entry["rank_contributions"] = rank_details.get(doc_id, {})
        fused_results.append(doc_entry)

    return fused_results
