"""Document Ingestion Script for GenAI Workshop Knowledge Assistant.

Loads local .txt documents, generates semantic chunks, computes dense vector embeddings
via Jina AI, stores vectors in persistent ChromaDB, and builds the Okapi BM25 lexical index.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DOCUMENTS_DIR, JINA_API_KEY, check_all_keys
from src.document_loader import load_documents
from src.chunker import chunk_documents
from src.embeddings import embed_documents
from src.vector_store import ChromaVectorStore
from src.bm25_search import BM25SearchEngine


def run_ingestion(reset_db: bool = True) -> bool:
    """Execute complete ingestion pipeline.
    
    Args:
        reset_db: If True, clears existing ChromaDB collection before inserting.
        
    Returns:
        True if ingestion succeeded, False otherwise.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("========================================")
    print(" GenAI Knowledge Assistant - Ingestion")
    print("========================================")

    # 1. Load documents
    print("\nLoading documents from data/documents/...")
    docs = load_documents(DOCUMENTS_DIR)
    if not docs:
        print("[!] No documents found in data/documents/. Please add text files to ingest.")
        return False
    print(f"[OK] Documents loaded ({len(docs)} files)")
    for d in docs:
        print(f"     - {d['metadata']['source']} ({d['metadata']['char_count']} characters)")

    # 2. Chunk documents
    print("\nCreating chunks...")
    chunks = chunk_documents(docs)
    if not chunks:
        print("[!] No chunks generated from documents.")
        return False
    print(f"[OK] Chunks created ({len(chunks)} chunks)")

    # 3. Generate Jina embeddings & update ChromaDB
    print("\nGenerating Jina embeddings...")
    chunk_texts = [c["text"] for c in chunks]

    # Check Jina API key
    if not JINA_API_KEY or JINA_API_KEY.startswith("your_"):
        print("[!] JINA_API_KEY is not configured in .env.")
        print("    Vector embeddings cannot be fetched from Jina API.")
        print("    Building BM25 lexical index only...")
        embeddings = None
    else:
        try:
            embeddings = embed_documents(chunk_texts)
            print(f"[OK] Embeddings created ({len(embeddings)} vectors, dim: {len(embeddings[0])})")
        except Exception as e:
            print(f"[!] Warning: Failed to generate Jina embeddings: {e}")
            print("    Proceeding with BM25 lexical index only.")
            embeddings = None

    # 4. Updating ChromaDB
    print("\nUpdating ChromaDB...")
    vector_store = ChromaVectorStore()
    if reset_db:
        vector_store.reset()

    if embeddings:
        try:
            vector_store.add_chunks(chunks, embeddings)
            print(f"[OK] ChromaDB updated ({vector_store.count()} chunks stored in data/chroma/)")
        except Exception as e:
            print(f"[!] Error storing vectors in ChromaDB: {e}")
    else:
        print("[INFO] ChromaDB skipped (no vector embeddings generated).")

    # 5. Building BM25 index
    print("\nBuilding BM25 index...")
    bm25_engine = BM25SearchEngine()
    bm25_engine.build_index(chunks)
    bm25_engine.save_index()
    print("[OK] BM25 index created and saved to data/bm25_index.json")

    print("\n========================================")
    print(" Ingestion completed successfully!")
    print("========================================\n")
    return True


if __name__ == "__main__":
    success = run_ingestion()
    if not success:
        sys.exit(1)
