"""Phase 10 Test: Orchestration & RAG Pipeline Verification."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import KnowledgeAssistant


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    assistant = KnowledgeAssistant()

    # 1. Test direct_llm route flow
    print("[*] Testing direct_llm execution flow...")
    with patch("src.rag.route_query", return_value={"route": "direct_llm", "reason": "General question"}), \
         patch("src.rag.generate_text", return_value="A Python list is a mutable sequence."):
        
        res = assistant.process_query("What is a Python list?")
        assert res["route"] == "direct_llm"
        assert res["internal_sources"] == []
        assert res["web_sources"] == []
        assert "Python list" in res["answer"]
        print("  * direct_llm flow verified successfully.")

    # 2. Test local_rag route flow
    print("\n[*] Testing local_rag execution flow...")
    mock_chunks = [
        {"id": "c1", "text": "Our internal RAG uses Jina and ChromaDB.", "metadata": {"source": "rag.txt"}, "rrf_score": 0.03}
    ]
    with patch("src.rag.route_query", return_value={"route": "local_rag", "reason": "Internal query"}), \
         patch("src.rag.hybrid_search", return_value=mock_chunks), \
         patch("src.rag.generate_text", return_value="Our RAG architecture uses Jina and ChromaDB."):
        
        res = assistant.process_query("According to our documents, what is our RAG architecture?")
        assert res["route"] == "local_rag"
        assert "rag.txt" in res["internal_sources"]
        assert res["web_sources"] == []
        print(f"  * local_rag flow verified. Retrieved internal sources: {res['internal_sources']}")

    # 3. Test web_search route flow
    print("\n[*] Testing web_search execution flow...")
    mock_web = [
        {"title": "Gemini 1.5 Pro Release", "url": "https://deepmind.google/gemini", "text": "Google announces 1.5 Pro"}
    ]
    with patch("src.rag.route_query", return_value={"route": "web_search", "reason": "Requires recent release info"}), \
         patch("src.rag.exa_search", return_value=mock_web), \
         patch("src.rag.generate_text", return_value="The latest model is Gemini 1.5 Pro."):
        
        res = assistant.process_query("What is the latest Gemini model?")
        assert res["route"] == "web_search"
        assert len(res["web_sources"]) == 1
        assert res["web_sources"][0]["title"] == "Gemini 1.5 Pro Release"
        print(f"  * web_search flow verified. Web sources: {[w['title'] for w in res['web_sources']]}")

    # 4. Test both route flow
    print("\n[*] Testing both route execution flow...")
    with patch("src.rag.route_query", return_value={"route": "both", "reason": "Comparison required"}), \
         patch("src.rag.hybrid_search", return_value=mock_chunks), \
         patch("src.rag.exa_search", return_value=mock_web), \
         patch("src.rag.generate_text", return_value="Comparing internal vs external: ..."):
        
        res = assistant.process_query("Compare our RAG architecture with external standards.")
        assert res["route"] == "both"
        assert "rag.txt" in res["internal_sources"]
        assert len(res["web_sources"]) == 1
        print(f"  * both flow verified. Internal sources: {res['internal_sources']}, Web sources count: {len(res['web_sources'])}")

    print("\n[SUCCESS] Phase 10 tests passed successfully!")


if __name__ == "__main__":
    main()
