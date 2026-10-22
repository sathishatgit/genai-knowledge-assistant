"""Test deployed instance 10-query quota and local unlimited access."""

import os
import sys
from pathlib import Path
from starlette.testclient import TestClient
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app, client_usage_tracker


def test_quota_behavior():
    client = TestClient(app)

    # 1. Localhost should have unlimited access
    print("\n[*] Testing localhost access (should be unlimited)...")
    with patch("src.rag.generate_text", return_value="Local response"):
        for i in range(12):
            res = client.post(
                "/api/query",
                json={"query": f"Local query #{i}", "route": "direct_llm"},
                headers={"host": "localhost:8000"}
            )
            assert res.status_code == 200, f"Query #{i} failed on localhost"
    print("  * Localhost successfully executed 12 queries without restriction.")

    # 2. Deployed instance should be limited to 10 queries per client
    print("\n[*] Testing deployed instance access (should enforce 10-query limit)...")
    client_usage_tracker.clear()
    deployed_client_id = "test_deployed_user_42"

    with patch("src.rag.generate_text", return_value="Deployed response"):
        for i in range(10):
            res = client.post(
                "/api/query",
                json={"query": f"Deployed query #{i+1}", "route": "direct_llm"},
                headers={"host": "genai-knowledge-assistant.onrender.com", "x-client-id": deployed_client_id}
            )
            assert res.status_code == 200, f"Query #{i+1} should succeed within 10-query quota"
            data = res.json()
            assert data["remaining_credits"] == 10 - (i + 1)
        
        print("  * First 10 queries succeeded on deployed host.")

        # 11th query must be rejected with 429
        res_11 = client.post(
            "/api/query",
            json={"query": "Exceeded query 11", "route": "direct_llm"},
            headers={"host": "genai-knowledge-assistant.onrender.com", "x-client-id": deployed_client_id}
        )
        assert res_11.status_code == 429
        assert "Sathish" in res_11.json()["detail"]
        print("  * 11th query was rejected with 429 and referenced Sathish.")

    print("\n[SUCCESS] Deployed quota and local exemption verified 100%!")


if __name__ == "__main__":
    test_quota_behavior()
