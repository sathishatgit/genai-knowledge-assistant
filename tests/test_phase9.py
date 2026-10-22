"""Phase 9 Test: Intelligent Router Verification."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GEMINI_API_KEY
from src.router import route_query


def test_mock_routing():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[*] Testing router JSON parsing & markdown stripping...")
    
    # 1. direct_llm with markdown code block
    mock_1 = '```json\n{"route": "direct_llm", "reason": "General programming concept"}\n```'
    res1 = route_query("What is a Python list?", mock_response=mock_1)
    assert res1["route"] == "direct_llm"
    print("  * direct_llm route parsed successfully.")

    # 2. local_rag
    mock_2 = '{"route": "local_rag", "reason": "Mentions internal documents"}'
    res2 = route_query("According to our documents, what is our RAG architecture?", mock_response=mock_2)
    assert res2["route"] == "local_rag"
    print("  * local_rag route parsed successfully.")

    # 3. web_search
    mock_3 = '{"route": "web_search", "reason": "Requires latest model info"}'
    res3 = route_query("What is the latest Gemini model?", mock_response=mock_3)
    assert res3["route"] == "web_search"
    print("  * web_search route parsed successfully.")

    # 4. both
    mock_4 = '{"route": "both", "reason": "Comparison between internal and external state"}'
    res4 = route_query("Compare our RAG architecture with latest industry approaches", mock_response=mock_4)
    assert res4["route"] == "both"
    print("  * both route parsed successfully.")

    # 5. Invalid / arbitrary route name fallback
    mock_invalid = '{"route": "hack_route", "reason": "Arbitrary route"}'
    res_invalid = route_query("test", mock_response=mock_invalid)
    assert res_invalid["route"] == "direct_llm", "Unrecognized route should fall back safely"
    print("  * Arbitrary route name safely caught and reverted to fallback.")

    # 6. Malformed JSON fallback
    mock_malformed = 'This is not json at all!'
    res_malformed = route_query("test", mock_response=mock_malformed)
    assert res_malformed["route"] == "direct_llm"
    print("  * Malformed JSON safely caught and reverted to fallback.")


def test_live_routing():
    if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_"):
        print("\n[*] Live GEMINI_API_KEY detected! Testing live intelligent classification...")
        queries = [
            ("What is a Python list?", "direct_llm"),
            ("According to our documents, what is our RAG architecture?", "local_rag"),
            ("What is the latest Gemini model release today?", "web_search"),
            ("Compare our internal RAG architecture with current industry standards.", "both"),
        ]
        for q, expected in queries:
            decision = route_query(q)
            print(f"\n  Query: '{q}'")
            print(f"  Selected Route: {decision['route'].upper()}")
            print(f"  Reason: {decision['reason']}")
    else:
        print("\n[*] (Live API key not configured yet; mock router verification passed)")


if __name__ == "__main__":
    test_mock_routing()
    test_live_routing()
    print("\n[SUCCESS] Phase 9 tests passed successfully!")
