"""Phase 6 Test: RRF and Hybrid Search Verification."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rrf import reciprocal_rank_fusion
from src.hybrid_search import hybrid_search


def test_rrf_math():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[*] Testing Reciprocal Rank Fusion formula...")
    # List 1: d1 (rank 1), d2 (rank 2)
    list1 = [
        {"id": "doc1", "text": "Doc 1 text", "metadata": {"source": "a.txt"}},
        {"id": "doc2", "text": "Doc 2 text", "metadata": {"source": "b.txt"}},
    ]
    # List 2: d2 (rank 1), d3 (rank 2)
    list2 = [
        {"id": "doc2", "text": "Doc 2 text", "metadata": {"source": "b.txt"}},
        {"id": "doc3", "text": "Doc 3 text", "metadata": {"source": "c.txt"}},
    ]

    # With k=60:
    # doc2 score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # doc1 score = 1/(60+1) = 1/61 = 0.016393
    # doc3 score = 1/(60+2) = 1/62 = 0.016129
    # doc2 should be ranked #1 because it appeared in both lists!
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    print(f"  * Fused ranking: {[item['id'] for item in fused]}")
    assert fused[0]["id"] == "doc2", "doc2 should rank first because it appeared in both lists"
    assert fused[1]["id"] == "doc1"
    assert fused[2]["id"] == "doc3"
    print(f"  * doc2 RRF score: {fused[0]['rrf_score']}")
    print("  * RRF ranking mathematics verified successfully.")


def test_hybrid_search():
    print("\n[*] Testing hybrid_search coordination...")
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        {"id": "doc_rag_1", "text": "RAG architecture BM25", "metadata": {"source": "rag.txt"}, "score": 2.5}
    ]

    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = [
        {"id": "doc_rag_1", "text": "RAG architecture Chroma", "metadata": {"source": "rag.txt"}, "score": 0.95},
        {"id": "doc_genai_1", "text": "LLM generation Chroma", "metadata": {"source": "generative_ai.txt"}, "score": 0.88}
    ]

    # Mock embed_query so vector search doesn't require live API call
    from unittest.mock import patch
    with patch("src.hybrid_search.embed_query", return_value=[0.1, 0.2, 0.3]):
        results = hybrid_search(
            query="Tell me about RAG architecture",
            top_k=2,
            bm25_engine=mock_bm25,
            vector_store=mock_vector_store
        )

    print(f"  * Hybrid results count: {len(results)}")
    assert len(results) == 2
    assert results[0]["id"] == "doc_rag_1"
    assert "rrf_score" in results[0]
    print(f"  * Top hybrid result ID: {results[0]['id']} (Score: {results[0]['rrf_score']})")
    print("\n[SUCCESS] Phase 6 tests passed successfully!")


if __name__ == "__main__":
    test_rrf_math()
    test_hybrid_search()
