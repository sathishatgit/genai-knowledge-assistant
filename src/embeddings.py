"""Jina AI Embeddings client module.

Uses the official Jina AI REST API (https://api.jina.ai/v1/embeddings) to generate
dense semantic vector embeddings for documents and search queries without needing
local model downloads or sentence-transformers.
"""

from typing import List
import requests
from src.config import JINA_API_KEY, JINA_EMBEDDING_MODEL, validate_api_key

JINA_API_URL = "https://api.jina.ai/v1/embeddings"


def _call_jina_api(
    texts: List[str],
    task: str = "retrieval.passage",
    model: str = JINA_EMBEDDING_MODEL,
    api_key: str | None = None
) -> List[List[float]]:
    """Helper function to call Jina AI Embeddings API.
    
    Args:
        texts: List of strings to embed.
        task: Jina task type ('retrieval.passage' for docs, 'retrieval.query' for queries).
        model: Jina embedding model name (defaults to 'jina-embeddings-v3').
        api_key: Optional Jina API key (falls back to config.JINA_API_KEY).
        
    Returns:
        List of embedding vectors (list of floats).
    """
    key = api_key if api_key is not None else JINA_API_KEY
    validate_api_key("JINA_API_KEY", key)
    
    if not texts:
        return []

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    # Clean input texts
    cleaned_texts = [t if t.strip() else " " for t in texts]

    payload = {
        "model": model,
        "task": task,
        "input": cleaned_texts
    }

    try:
        response = requests.post(JINA_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Sort embeddings by original index
        sorted_records = sorted(data["data"], key=lambda x: x["index"])
        return [record["embedding"] for record in sorted_records]
        
    except requests.exceptions.HTTPError as http_err:
        status_code = response.status_code if response else "Unknown"
        error_msg = f"Jina AI API HTTP error {status_code}: {http_err}"
        try:
            detail = response.json()
            error_msg += f" - Detail: {detail}"
        except Exception:
            pass
        raise RuntimeError(error_msg) from http_err
    except Exception as err:
        raise RuntimeError(f"Failed to obtain embeddings from Jina AI API: {err}") from err


def embed_documents(
    texts: List[str],
    batch_size: int = 32,
    model: str = JINA_EMBEDDING_MODEL,
    api_key: str | None = None
) -> List[List[float]]:
    """Generate dense vector embeddings for a list of document texts.
    
    Uses 'retrieval.passage' task to optimize document chunk representations.
    
    Args:
        texts: List of chunk texts to embed.
        batch_size: Number of texts per API request.
        model: Model name to use.
        api_key: Optional API key override.
        
    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []
        
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = _call_jina_api(batch, task="retrieval.passage", model=model, api_key=api_key)
        all_embeddings.extend(batch_embeddings)
        
    return all_embeddings


def embed_query(
    query: str,
    model: str = JINA_EMBEDDING_MODEL,
    api_key: str | None = None
) -> List[float]:
    """Generate a dense vector embedding for a single user search query.
    
    Uses 'retrieval.query' task to optimize search query representations.
    
    Args:
        query: User question string.
        model: Model name to use.
        api_key: Optional API key override.
        
    Returns:
        Embedding vector as list of floats.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty.")
        
    results = _call_jina_api([query], task="retrieval.query", model=model, api_key=api_key)
    return results[0]
