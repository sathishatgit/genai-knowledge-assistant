"""Phase 15 Test: Interactive UI & Web Server Verification."""

import sys
from pathlib import Path
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app


def test_ui_endpoints():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[*] Initializing TestClient for AGUI FastAPI application...")
    client = TestClient(app)

    # 1. Test GET / (HTML frontend)
    print("\n[*] Testing GET / (Single Page Application HTML)...")
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "AGUI Agent Hub" in res_index.text
    print("  * Frontend HTML served successfully (status 200).")

    # 2. Test GET /api/status
    print("\n[*] Testing GET /api/status (System Health & Component Status)...")
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    data_status = res_status.json()
    print(f"  * Status response: {data_status}")
    assert "gemini" in data_status
    assert "chroma_count" in data_status

    # 3. Test POST /api/query with direct_llm forced route
    print("\n[*] Testing POST /api/query endpoint...")
    from unittest.mock import patch
    with patch("src.rag.generate_text", return_value="A Python list is a mutable sequence."):
        res_query = client.post("/api/query", json={"query": "What is a Python list?", "route": "direct_llm"})
        assert res_query.status_code == 200
        data_query = res_query.json()
        print(f"  * Query response route: {data_query['route']}")
        print(f"  * Answer snippet: {data_query['answer'][:40]}...")
        assert data_query["route"] == "direct_llm"
        assert "Python list" in data_query["answer"]

    print("\n[SUCCESS] Phase 15 UI and API server verified successfully!")


if __name__ == "__main__":
    test_ui_endpoints()
