"""Intelligent Query Router module.

Analyzes user questions using Google Gemini and classifies them into one of
four execution strategies using strict structured JSON:
  1. direct_llm: General questions and concepts that do not require external/private data.
  2. local_rag: Questions about internal documents, architecture, or stored files.
  3. web_search: Real-time, latest updates, current events, or external facts.
  4. both: In-depth comparisons between internal documents and current external state.

DO NOT ALWAYS USE RAG: Unnecessary retrieval introduces latency, cost, and noise.
"""

import json
import re
from typing import Dict, Any, Optional
from src.llm import generate_text

ALLOWED_ROUTES = {"direct_llm", "local_rag", "web_search", "both"}
DEFAULT_FALLBACK_ROUTE = "direct_llm"

ROUTER_SYSTEM_PROMPT = """You are an intelligent query routing engine for an enterprise AI system.
Your sole responsibility is to classify user queries into the single most appropriate retrieval route.

CRITICAL PRINCIPLES:
1. DO NOT ALWAYS USE RAG. Retrieval adds latency, cost, and potential context noise.
2. DO NOT use web search unless current or external real-time information is genuinely needed.

ROUTES AND CRITERIA:
- "direct_llm":
  * General programming, mathematics, definitions, explanations, coding questions, general AI concepts.
  * No private/internal organization documents needed.
  * Information that does not change rapidly and does not require real-time lookup.
  * Example: "What is a Python list?", "Explain recursion", "How does gradient descent work?"

- "local_rag":
  * Questions explicitly referring to internal documents, company knowledge base, local files, our architecture, our system configuration.
  * Example: "According to our documents, what is our RAG architecture?", "What chunk size do we use in our pipeline?"

- "web_search":
  * Questions asking for latest releases, current events, recent news, benchmark updates, real-time pricing, or explicitly asking to search the web.
  * Example: "What is the latest Gemini model?", "Who won the latest election?", "What are the latest AI news this week?"

- "both":
  * Questions that require comparing our internal documents/architecture with current external/industry state.
  * Example: "Compare our RAG architecture with the latest industry RAG approaches.", "How does our chunking strategy compare to current 2025/2026 standards?"

RESPONSE FORMAT:
You MUST respond with valid JSON and NOTHING ELSE. No introductory text, no markdown outside the JSON block.
Format:
{
  "route": "direct_llm" | "local_rag" | "web_search" | "both",
  "reason": "Brief, clear explanation of why this route was selected."
}
"""


def _clean_json_string(raw_text: str) -> str:
    """Extract and clean raw JSON string from LLM output."""
    text = raw_text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _fallback_classify(query: str, err_detail: str) -> Dict[str, str]:
    """Smart heuristic classification fallback when LLM API is rate-limited or unavailable."""
    q_lower = query.lower()
    
    # 1. Check for 'both' (comparison between internal and external)
    if "compare" in q_lower and ("our" in q_lower or "internal" in q_lower or "approach" in q_lower or "architecture" in q_lower):
        return {
            "route": "both",
            "reason": "Query asks for comparison between internal architecture and industry standards (heuristic routing)."
        }
    
    # 2. Check for 'local_rag' (explicitly internal/local documents)
    if any(k in q_lower for k in ["our document", "our knowledge", "our rag", "internal", "according to our", "our system", "our policy"]):
        return {
            "route": "local_rag",
            "reason": "Query specifically asks about internal documents and knowledge base (heuristic routing)."
        }
        
    # 3. Check for 'web_search' (latest, recent, real-time)
    if any(k in q_lower for k in ["latest", "recent", "current", "news", "today", "search the web", "release", "price"]):
        return {
            "route": "web_search",
            "reason": "Query asks for latest or real-time external information (heuristic routing)."
        }

    # 4. Default to direct_llm
    return {
        "route": "direct_llm",
        "reason": "General question that does not require private documents or real-time lookup (heuristic routing)."
    }


def route_query(
    query: str,
    api_key: Optional[str] = None,
    mock_response: Optional[str] = None
) -> Dict[str, str]:
    """Classify the user's query into the optimal execution route.
    
    Args:
        query: The raw user prompt or question.
        api_key: Optional Gemini API key override.
        mock_response: Optional string to simulate LLM output during tests.
        
    Returns:
        Dictionary containing:
            - 'route': one of ['direct_llm', 'local_rag', 'web_search', 'both']
            - 'reason': explanation string
    """
    if not query or not query.strip():
        return {
            "route": DEFAULT_FALLBACK_ROUTE,
            "reason": "Empty query received; defaulting to direct LLM."
        }

    user_prompt = f"Classify the following query:\n\"{query}\""

    try:
        if mock_response is not None:
            raw_response = mock_response
        else:
            raw_response = generate_text(
                prompt=user_prompt,
                system_instruction=ROUTER_SYSTEM_PROMPT,
                temperature=0.0,
                api_key=api_key
            )

        cleaned_json = _clean_json_string(raw_response)
        parsed = json.loads(cleaned_json)

        route = str(parsed.get("route", "")).strip().lower()
        reason = str(parsed.get("reason", "No reason provided.")).strip()

        if route not in ALLOWED_ROUTES:
            print(f"Warning: Router returned unrecognized route '{route}'. Using heuristic routing.")
            return _fallback_classify(query, f"Unrecognized route '{route}'")

        return {
            "route": route,
            "reason": reason
        }

    except json.JSONDecodeError as json_err:
        print(f"Warning: Failed to parse router JSON output ({json_err}). Using heuristic routing.")
        return _fallback_classify(query, f"JSON parse error: {json_err}")
    except Exception as err:
        print(f"Warning: Routing encountered an error ({err}). Using heuristic routing.")
        return _fallback_classify(query, str(err))
