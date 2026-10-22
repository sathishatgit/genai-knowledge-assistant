"""Local persistent ChromaDB vector store module.

Stores chunk embeddings and metadata locally on disk in data/chroma/ without
requiring external Docker services or servers.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.config import CHROMA_DIR, DEFAULT_TOP_K

COLLECTION_NAME = "workshop_knowledge_base"


_shared_client = None

def get_chroma_client(persist_dir: Path | str = CHROMA_DIR) -> chromadb.ClientAPI:
    """Return a shared PersistentClient instance with standardized settings."""
    global _shared_client
    path_str = str(Path(persist_dir).resolve())
    if _shared_client is None:
        _shared_client = chromadb.PersistentClient(
            path=path_str,
            settings=Settings(anonymized_telemetry=False)
        )
    return _shared_client


class ChromaVectorStore:
    """Persistent ChromaDB vector store wrapper for document chunks."""

    def __init__(self, persist_dir: Path | str = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = get_chroma_client(self.persist_dir)

    @property
    def collection(self):
        """Always return the active collection, avoiding stale handles."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        """Add chunk documents and their precomputed embeddings to ChromaDB.
        
        Args:
            chunks: List of chunk dictionaries containing 'id', 'text', and 'metadata'.
            embeddings: List of embedding vectors corresponding to chunks.
        """
        if not chunks:
            return
            
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided but {len(embeddings)} embeddings given."
            )

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        
        # Chroma requires metadata values to be str, int, float, or bool
        cleaned_metadatas = []
        for c in chunks:
            meta = {}
            for k, v in c.get("metadata", {}).items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            cleaned_metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=cleaned_metadatas
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search using a query embedding.
        
        Args:
            query_embedding: The dense vector representation of the query.
            top_k: Number of nearest neighbors to retrieve.
            
        Returns:
            Normalized list of result dictionaries:
            [
                {
                    "id": "...",
                    "text": "...",
                    "metadata": {...},
                    "score": float  # Cosine similarity score (higher is more similar)
                }
            ]
        """
        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        normalized_results: List[Dict[str, Any]] = []
        
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for doc_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # For cosine distance: distance is in [0, 2]. Similarity is 1.0 - distance.
            similarity_score = max(0.0, 1.0 - float(dist))
            normalized_results.append({
                "id": doc_id,
                "text": doc_text,
                "metadata": meta,
                "score": round(similarity_score, 4),
                "distance": round(float(dist), 4)
            })

        return normalized_results

    def count(self) -> int:
        """Return total number of chunks currently stored in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete collection to clear all data. Next access creates it fresh."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
