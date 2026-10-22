"""Exa AI Web Search module.

Uses the official Exa API (https://api.exa.ai/search) to retrieve current,
authoritative web documents, titles, URLs, and text snippets.
"""

from typing import List, Dict, Any, Optional
import requests
from src.config import EXA_API_KEY, EXA_NUM_RESULTS, validate_api_key

EXA_SEARCH_ENDPOINT = "https://api.exa.ai/search"


def exa_search(
    query: str,
    num_results: int = EXA_NUM_RESULTS,
    api_key: Optional[str] = None
) -> List[Dict[str, str]]:
    """Search the web using Exa AI search.
    
    Args:
        query: Search string or user question.
        num_results: Number of top web results to return.
        api_key: Optional Exa API key override.
        
    Returns:
        List of dictionaries with keys:
            - title: Web page title
            - url: URL of the source
            - text: Extracted content snippet
    """
    key = api_key if api_key is not None else EXA_API_KEY
    validate_api_key("EXA_API_KEY", key)

    if not query or not query.strip():
        return []

    headers = {
        "x-api-key": key,
        "Content-Type": "application/json",
        "User-Agent": "GenAI-Workshop-Assistant/1.0"
    }

    payload = {
        "query": query.strip(),
        "numResults": max(1, num_results),
        "contents": {
            "text": {
                "maxCharacters": 1000
            }
        }
    }

    try:
        response = requests.post(
            EXA_SEARCH_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=20
        )
        response.raise_for_status()
        data = response.json()
        
        raw_results = data.get("results", [])
        cleaned_results: List[Dict[str, str]] = []

        for item in raw_results:
            title = item.get("title") or "Untitled Document"
            url = item.get("url") or ""
            text = item.get("text") or ""
            
            cleaned_results.append({
                "title": title.strip(),
                "url": url.strip(),
                "text": text.strip()
            })

        return cleaned_results

    except requests.exceptions.HTTPError as http_err:
        status_code = response.status_code if response is not None else "Unknown"
        err_msg = f"Exa Search API HTTP error ({status_code}): {http_err}"
        try:
            detail = response.json()
            err_msg += f" - Detail: {detail}"
        except Exception:
            pass
        raise RuntimeError(err_msg) from http_err
    except Exception as err:
        raise RuntimeError(f"Failed to query Exa Search API: {err}") from err
