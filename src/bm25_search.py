"""BM25 Lexical Keyword Search module.

Uses the rank-bm25 implementation of Okapi BM25 to perform fast, keyword-based
information retrieval without neural embeddings. Ideal for exact keyword, acronym,
and technical terminology matching.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.config import BM25_INDEX_PATH, DEFAULT_TOP_K


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words.
    
    Beginner-friendly tokenizer that strips punctuation and splits words.
    """
    if not text:
        return []
    # Find all sequences of alphanumeric characters
    tokens = re.findall(r"\w+", text.lower())
    return tokens


class BM25SearchEngine:
    """Independent BM25 search engine for lexical retrieval."""

    def __init__(self, index_file_path: Path | str = BM25_INDEX_PATH):
        self.index_file = Path(index_file_path)
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build BM25 index from document chunks and save to disk.
        
        Args:
            chunks: List of chunk dictionaries containing 'id', 'text', and 'metadata'.
        """
        if not chunks:
            self.chunks = []
            self.bm25 = None
            self.corpus_tokens = []
            return

        self.chunks = chunks
        self.corpus_tokens = [simple_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def save_index(self, path: Optional[Path | str] = None) -> None:
        """Persist chunk data and tokenized index representation to JSON."""
        target_path = Path(path) if path else self.index_file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "chunks": self.chunks,
            "corpus_tokens": self.corpus_tokens
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_index(self, path: Optional[Path | str] = None) -> bool:
        """Load saved index from disk. Returns True if successfully loaded."""
        target_path = Path(path) if path else self.index_file
        if not target_path.exists():
            return False

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data.get("chunks", [])
            self.corpus_tokens = data.get("corpus_tokens", [])
            if self.corpus_tokens:
                self.bm25 = BM25Okapi(self.corpus_tokens)
                return True
        except Exception as e:
            print(f"Error loading BM25 index from {target_path}: {e}")
        return False

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Search chunks using Okapi BM25 keyword matching.
        
        Args:
            query: User search query string.
            top_k: Number of highest-ranked results to return.
            
        Returns:
            Normalized list of result dictionaries:
            [
                {
                    "id": "...",
                    "text": "...",
                    "metadata": {...},
                    "score": float  # BM25 relevance score
                }
            ]
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # Pair chunk with score and sort descending
        scored_chunks = []
        for chunk, score in zip(self.chunks, scores):
            # Only include chunks with positive BM25 score if possible,
            # or retain top candidates if query keywords appear
            if score > 0.0:
                scored_chunks.append((chunk, float(score)))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_matches = scored_chunks[:top_k]

        results = []
        for chunk, score in top_matches:
            results.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "score": round(score, 4)
            })

        return results
