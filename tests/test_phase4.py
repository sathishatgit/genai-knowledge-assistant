"""Phase 4 Test: ChromaDB Vector Store Verification."""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import ChromaVectorStore


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_db_dir = PROJECT_ROOT / "data" / "test_chroma"
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir, ignore_errors=True)

    print("[*] Initializing test ChromaDB vector store...")
    store = ChromaVectorStore(persist_dir=test_db_dir, collection_name="test_collection")
    assert store.count() == 0, "Initial count should be 0"

    print("[*] Adding test chunks with synthetic embeddings...")
    sample_chunks = [
        {
            "id": "doc1_chunk_0",
            "text": "Python is a versatile programming language.",
            "metadata": {"source": "python.txt", "chunk_index": 0}
        },
        {
            "id": "doc2_chunk_0",
            "text": "Machine learning enables learning from data.",
            "metadata": {"source": "ml.txt", "chunk_index": 0}
        }
    ]
    # Unit vectors
    sample_embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]

    store.add_chunks(sample_chunks, sample_embeddings)
    assert store.count() == 2, f"Expected 2 documents in store, got {store.count()}"
    print(f"  * Total documents stored: {store.count()}")

    print("[*] Performing vector similarity search...")
    # Query vector close to doc1
    query_vec = [0.95, 0.05, 0.0]
    results = store.search(query_vec, top_k=2)
    
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    first = results[0]
    print(f"  * Top match ID: {first['id']}")
    print(f"  * Top match text: {first['text']}")
    print(f"  * Top match score: {first['score']}")
    print(f"  * Top match metadata: {first['metadata']}")
    
    assert first["id"] == "doc1_chunk_0", "Expected doc1_chunk_0 to be the top match"
    assert "score" in first and isinstance(first["score"], float)

    print("[*] Testing store reset...")
    store.reset()
    assert store.count() == 0, "Count should be 0 after reset"
    print("  * Store reset verified.")

    # Clean up test directory
    shutil.rmtree(test_db_dir, ignore_errors=True)
    print("\n[SUCCESS] Phase 4 tests passed successfully!")


if __name__ == "__main__":
    main()
