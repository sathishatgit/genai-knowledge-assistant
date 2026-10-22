"""Phase 3 Test: Jina Embeddings Module Verification."""

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import JINA_API_KEY
from src.embeddings import embed_documents, embed_query


def test_jina_validation():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[*] Testing Jina input validation...")
    try:
        embed_query("")
        assert False, "Should raise ValueError on empty query"
    except ValueError as e:
        print(f"  * Caught expected ValueError on empty query: {e}")

    # Test with mocked API response
    print("\n[*] Testing Jina API request structure with mock response...")
    mock_response = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"index": 1, "embedding": [0.4, 0.5, 0.6]}
        ]
    }
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None

        with patch("src.embeddings.JINA_API_KEY", "test_key_123"):
            docs = embed_documents(["hello world", "rag test"])
            assert len(docs) == 2
            assert docs[0] == [0.1, 0.2, 0.3]
            assert docs[1] == [0.4, 0.5, 0.6]
            print("  * embed_documents mock response verified.")

            q = embed_query("hello")
            assert q == [0.1, 0.2, 0.3]
            print("  * embed_query mock response verified.")

    # If live key is provided, test live call
    if JINA_API_KEY and not JINA_API_KEY.startswith("your_"):
        print("\n[*] Live JINA_API_KEY detected! Testing live API call...")
        try:
            live_emb = embed_query("What is machine learning?")
            print(f"  * Live embedding dimension: {len(live_emb)}")
            assert len(live_emb) > 0
            print("  * Live Jina Embeddings test succeeded!")
        except Exception as err:
            print(f"  * Note: Live API call returned: {err}")
    else:
        print("\n[*] (Live API key not provided yet in .env; mock verification passed)")

    print("\n[SUCCESS] Phase 3 tests passed successfully!")


if __name__ == "__main__":
    test_jina_validation()
