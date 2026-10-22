"""Hybrid Search module.

Combines Okapi BM25 keyword search with ChromaDB dense vector search using
Reciprocal Rank Fusion (RRF) to produce robust, high-quality document retrieval.
"""

from typing import List, Dict, Any, Optional
from src.config import DEFAULT_TOP_K, RRF_K
from src.embeddings import embed_query
from src.bm25_search import BM25SearchEngine
from src.vector_store import ChromaVectorStore
from src.rrf import reciprocal_rank_fusion


def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    bm25_engine: Optional[BM25SearchEngine] = None,
    vector_store: Optional[ChromaVectorStore] = None,
    k_rrf: int = RRF_K
) -> List[Dict[str, Any]]:
    """Execute hybrid retrieval combining BM25 keyword matching and dense vector search.
    
    Workflow:
        1. BM25 search retrieves candidates based on exact lexical keyword matches.
        2. Jina AI embeddings convert query into dense vector -> ChromaDB retrieves candidates.
        3. Reciprocal Rank Fusion combines both candidate rankings into a unified list.
        4. Top-K highest-ranked fused documents are returned.
        
    Args:
        query: User search question.
        top_k: Number of final fused documents to return.
        bm25_engine: Initialized BM25SearchEngine instance.
        vector_store: Initialized ChromaVectorStore instance.
        k_rrf: Constant k for RRF calculation (default 60).
        
    Returns:
        List of fused result dictionaries:
        [
            {
                "id": "...",
                "text": "...",
                "metadata": {...},
                "rrf_score": float
            }
        ]
    """
    if not query or not query.strip():
        return []

    # 1. BM25 Search
    bm25_results: List[Dict[str, Any]] = []
    if bm25_engine is not None:
        try:
            # Retrieve a candidate pool for fusion
            candidate_k = max(top_k * 2, 10)
            bm25_results = bm25_engine.search(query, top_k=candidate_k)
        except Exception as e:
            print(f"Warning: BM25 search encountered an error: {e}")

    # 2. Vector Search (Jina Embeddings + ChromaDB)
    vector_results: List[Dict[str, Any]] = []
    if vector_store is not None:
        try:
            query_vector = embed_query(query)
            candidate_k = max(top_k * 2, 10)
            vector_results = vector_store.search(query_vector, top_k=candidate_k)
        except Exception as e:
            print(f"Warning: Vector search encountered an error: {e}")

    # Handle edge cases where one or both search systems returned results
    if not bm25_results and not vector_results:
        return []

    if bm25_results and not vector_results:
        # Fall back to BM25 results only
        return bm25_results[:top_k]

    if vector_results and not bm25_results:
        # Fall back to vector results only
        return vector_results[:top_k]

    # 3. Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion([bm25_results, vector_results], k=k_rrf)
    return fused[:top_k]
