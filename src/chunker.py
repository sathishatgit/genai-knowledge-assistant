"""Chunking module for GenAI Knowledge Assistant.

Splits larger documents into smaller, overlapping segments (chunks)
so that retrieval can pinpoint specific relevant paragraphs and pass them
within the LLM context window.
"""

from typing import List, Dict, Any
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping character slices.
    
    A sliding window moves forward by (chunk_size - chunk_overlap) characters.
    
    Args:
        text: Raw text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
        
    Returns:
        List of text chunks.
    """
    if not text:
        return []
        
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be strictly greater than chunk_overlap.")
        
    # If text is already shorter than chunk size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start += step

    return chunks


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """Split a list of loaded documents into chunks with inherited metadata.
    
    Args:
        documents: Output from document_loader.load_documents().
        chunk_size: Maximum size per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        List of chunk dictionaries with:
            - id: Unique chunk ID (e.g., 'doc_rag.txt_chunk_0')
            - text: Text of this chunk
            - metadata: Copy of doc metadata plus chunk_index and total_chunks
    """
    chunked_items: List[Dict[str, Any]] = []

    for doc in documents:
        doc_id = doc["id"]
        raw_text = doc["text"]
        base_meta = doc.get("metadata", {})
        
        text_pieces = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total_chunks = len(text_pieces)

        for idx, piece in enumerate(text_pieces):
            chunk_id = f"{doc_id}_chunk_{idx}"
            chunk_meta = dict(base_meta)
            chunk_meta["chunk_index"] = idx
            chunk_meta["total_chunks"] = total_chunks
            chunk_meta["chunk_char_length"] = len(piece)

            chunked_items.append({
                "id": chunk_id,
                "text": piece,
                "metadata": chunk_meta
            })

    return chunked_items
