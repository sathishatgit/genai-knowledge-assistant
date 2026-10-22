"""Phase 5 Test: BM25 Search Engine Verification."""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_search import BM25SearchEngine, simple_tokenize


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[*] Testing simple_tokenize...")
    tokens = simple_tokenize("Python 3.12: Fast, Interpreted & Dynamic!")
    print(f"  * Tokens: {tokens}")
    assert "python" in tokens and "dynamic" in tokens

    print("\n[*] Initializing BM25 Search Engine with sample chunks...")
    chunks = [
        {
            "id": "c1",
            "text": "Python lists are mutable ordered sequences.",
            "metadata": {"source": "python.txt"}
        },
        {
            "id": "c2",
            "text": "Supervised learning algorithms require labeled datasets.",
            "metadata": {"source": "machine_learning.txt"}
        },
        {
            "id": "c3",
            "text": "BM25Okapi is a probabilistic lexical ranking function.",
            "metadata": {"source": "rag.txt"}
        }
    ]

    engine = BM25SearchEngine()
    engine.build_index(chunks)

    print("\n[*] Searching for exact keyword 'BM25Okapi'...")
    results = engine.search("BM25Okapi lexical", top_k=2)
    print(f"  * Results returned: {len(results)}")
    assert len(results) > 0
    top = results[0]
    print(f"  * Top match ID: {top['id']} (Score: {top['score']})")
    assert top["id"] == "c3", f"Expected c3, got {top['id']}"

    print("\n[*] Searching for 'mutable sequences'...")
    results2 = engine.search("mutable sequences", top_k=2)
    assert len(results2) > 0
    print(f"  * Top match ID: {results2[0]['id']} (Score: {results2[0]['score']})")
    assert results2[0]["id"] == "c1"

    print("\n[*] Testing index serialization (save & load)...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        engine.save_index(tmp_path)
        engine_loaded = BM25SearchEngine(index_file_path=tmp_path)
        loaded_ok = engine_loaded.load_index()
        assert loaded_ok, "Failed to load saved BM25 index"
        results_loaded = engine_loaded.search("BM25Okapi", top_k=1)
        assert results_loaded[0]["id"] == "c3"
        print("  * BM25 index save/load successfully verified.")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print("\n[SUCCESS] Phase 5 tests passed successfully!")


if __name__ == "__main__":
    main()
