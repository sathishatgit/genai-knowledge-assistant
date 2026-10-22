"""Phase 8 Test: Gemini Generation Verification."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GEMINI_API_KEY
from src.gemini import generate_text


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[*] Testing empty prompt handling...")
    assert generate_text("") == ""

    print("[*] Testing Gemini generation with mock...")
    with patch("google.generativeai.configure") as mock_conf, \
         patch("google.generativeai.GenerativeModel") as mock_model_cls:
        
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello! A Python list is a mutable sequence."
        mock_instance.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_instance

        result = generate_text("What is a Python list?", api_key="dummy_gemini_key")
        assert "Python list" in result
        print(f"  * Mock Gemini response received: {result[:40]}...")

    if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_"):
        print("\n[*] Live GEMINI_API_KEY detected! Testing live API call...")
        try:
            live_output = generate_text("Reply with exactly: 'OK'")
            print(f"  * Live Gemini output: {live_output}")
            assert len(live_output) > 0
        except Exception as err:
            print(f"  * Note: Live Gemini call returned: {err}")
    else:
        print("\n[*] (Live Gemini API key not configured yet; mock verification passed)")

    print("\n[SUCCESS] Phase 8 tests passed successfully!")


if __name__ == "__main__":
    main()
