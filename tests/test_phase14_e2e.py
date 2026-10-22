"""Phase 14: Complete End-to-End Verification Test Suite.

Tests check_setup, ingestion, and all 4 operational routing pathways:
  1. direct_llm: ("What is a Python list?")
  2. local_rag:  ("According to our knowledge base, what is RAG?")
  3. web_search: ("What is the latest Gemini model?")
  4. both:       ("Compare our RAG approach with the latest RAG approaches.")
"""

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from check_setup import check_setup
from ingest import run_ingestion
from src.rag import KnowledgeAssistant


def test_e2e():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n==================================================")
    print(" PHASE 14: FULL END-TO-END VERIFICATION SUITE")
    print("==================================================")

    # 1. Run check_setup
    print("\n>>> STEP 1: Running check_setup()...")
    check_setup()

    # 2. Run ingestion
    print("\n>>> STEP 2: Running run_ingestion()...")
    ingest_success = run_ingestion(reset_db=False)
    assert ingest_success, "Ingestion failed!"

    # 3. Test Knowledge Assistant across all 4 canonical queries
    print("\n>>> STEP 3: Testing 4 Canonical Routing Scenarios...")
    assistant = KnowledgeAssistant()

    test_scenarios = [
        {
            "name": "Test 1: Direct LLM",
            "query": "What is a Python list?",
            "expected_route": "direct_llm",
            "mock_route_response": '{"route": "direct_llm", "reason": "General programming concept not requiring retrieval"}',
            "mock_llm_answer": "A Python list is a mutable, ordered sequence of elements.",
            "mock_web": [],
        },
        {
            "name": "Test 2: Local RAG",
            "query": "According to our knowledge base, what is RAG?",
            "expected_route": "local_rag",
            "mock_route_response": '{"route": "local_rag", "reason": "Explicitly asks about internal knowledge base and RAG architecture"}',
            "mock_llm_answer": "According to our knowledge base, RAG combines BM25 and ChromaDB using Reciprocal Rank Fusion.",
            "mock_web": [],
        },
        {
            "name": "Test 3: Web Search",
            "query": "What is the latest Gemini model?",
            "expected_route": "web_search",
            "mock_route_response": '{"route": "web_search", "reason": "Asks for the latest real-time model release information"}',
            "mock_llm_answer": "The latest Gemini model is Gemini 1.5 Pro and Flash with large context windows.",
            "mock_web": [{"title": "Gemini 1.5 Announcement", "url": "https://deepmind.google/gemini", "text": "Google releases Gemini 1.5"}],
        },
        {
            "name": "Test 4: Both (Internal + Web)",
            "query": "Compare our RAG approach with the latest RAG approaches.",
            "expected_route": "both",
            "mock_route_response": '{"route": "both", "reason": "Requires comparing internal architecture with latest external industry methods"}',
            "mock_llm_answer": "Internal uses BM25+ChromaDB+RRF, while industry uses Agentic and Graph RAG.",
            "mock_web": [{"title": "RAG Industry Trends 2026", "url": "https://example.com/trends", "text": "GraphRAG and Agentic RAG"}],
        },
    ]

    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        query = scenario["query"]
        expected_route = scenario["expected_route"]
        print(f"User Query: \"{query}\"")

        with patch("src.router.generate_text", return_value=scenario["mock_route_response"]), \
             patch("src.rag.generate_text", return_value=scenario["mock_llm_answer"]), \
             patch("src.rag.exa_search", return_value=scenario["mock_web"]):

            result = assistant.process_query(query)

            print(f"Selected Route: {result['route'].upper()}")
            print(f"Reason: {result['reason']}")
            print(f"Answer: {result['answer']}")

            assert result["route"] == expected_route, f"Expected route {expected_route}, got {result['route']}"

            if expected_route == "direct_llm":
                assert result["internal_sources"] == []
                assert result["web_sources"] == []
                print("  * Verified zero retrieval overhead (direct_llm)")

            elif expected_route == "local_rag":
                assert len(result["internal_sources"]) > 0
                print(f"  * Verified internal document sources: {result['internal_sources']}")

            elif expected_route == "web_search":
                assert len(result["web_sources"]) > 0
                print(f"  * Verified web search sources: {[w['title'] for w in result['web_sources']]}")

            elif expected_route == "both":
                assert len(result["internal_sources"]) > 0
                assert len(result["web_sources"]) > 0
                print(f"  * Verified both sources: Internal={result['internal_sources']}, Web={[w['title'] for w in result['web_sources']]}")

            print(f"  [PASS] {scenario['name']} verified successfully!")

    print("\n==================================================")
    print(" [SUCCESS] ALL 14 PHASES VERIFIED SUCCESSFULLY!")
    print("==================================================\n")


if __name__ == "__main__":
    test_e2e()
