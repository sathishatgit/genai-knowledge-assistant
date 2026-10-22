"""Phase 7 Test: Exa Web Search Verification."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXA_API_KEY
from src.exa_search import exa_search


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[*] Testing Exa Search module input validation...")
    # Empty query should return empty list
    empty_res = exa_search("", api_key="dummy_key")
    assert empty_res == []

    print("[*] Testing Exa API request and parsing with mock response...")
    mock_payload = {
        "results": [
            {
                "id": "https://example.com/gemini",
                "title": "Google Gemini Models Overview",
                "url": "https://example.com/gemini",
                "text": "Gemini 1.5 Pro and Flash feature a multi-million token context window."
            }
        ]
    }
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        results = exa_search("latest gemini model", api_key="test_exa_key")
        assert len(results) == 1
        first = results[0]
        assert first["title"] == "Google Gemini Models Overview"
        assert first["url"] == "https://example.com/gemini"
        assert "multi-million token" in first["text"]
        print(f"  * Mock search returned: {first['title']} ({first['url']})")

    # If live key is provided, test live call
    if EXA_API_KEY and not EXA_API_KEY.startswith("your_"):
        print("\n[*] Live EXA_API_KEY detected! Testing live API call...")
        try:
            live_results = exa_search("latest Google Gemini update", num_results=2)
            print(f"  * Live search returned {len(live_results)} results:")
            for item in live_results:
                print(f"    - {item['title']}: {item['url']}")
        except Exception as err:
            print(f"  * Note: Live Exa call returned: {err}")
    else:
        print("\n[*] (Live Exa API key not configured yet; mock verification passed)")

    print("\n[SUCCESS] Phase 7 tests passed successfully!")


if __name__ == "__main__":
    main()
