import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_documents
from src.chunker import chunk_documents


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    docs = load_documents()
    print(f"Loaded {len(docs)} documents:")
    for d in docs:
        print(f"  * {d['metadata']['source']} ({d['metadata']['char_count']} chars)")
    
    assert len(docs) == 4, f"Expected 4 documents, got {len(docs)}"
    
    chunks = chunk_documents(docs)
    print(f"\nGenerated {len(chunks)} total chunks.")
    assert len(chunks) > 0, "No chunks generated!"
    
    print("\nSample Chunk Preview:")
    first = chunks[0]
    print(f"ID: {first['id']}")
    print(f"Metadata: {first['metadata']}")
    print(f"Text preview: {first['text'][:120]}...\n")
    print("[SUCCESS] Phase 2 tests passed successfully!")


if __name__ == "__main__":
    main()
