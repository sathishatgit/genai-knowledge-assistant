"""RAG and Multi-Route Pipeline Orchestrator.

Coordinates query routing and execution across all 4 operational paths:
  1. direct_llm : Direct generation without retrieval overhead.
  2. local_rag  : Lexical (BM25) + Semantic (ChromaDB/Jina) hybrid search with RRF.
  3. web_search : Real-time web retrieval via Exa Search.
  4. both       : Fused internal knowledge and external web intelligence.
"""

from typing import Dict, Any, List, Optional
from src.router import route_query
from src.hybrid_search import hybrid_search
from src.exa_search import exa_search
from src.gemini import generate_text
from src.bm25_search import BM25SearchEngine
from src.vector_store import ChromaVectorStore
from src.config import DEFAULT_TOP_K, EXA_NUM_RESULTS


FINAL_ANSWER_SYSTEM_PROMPT = """You are an authoritative, accurate, and concise GenAI Assistant.
Your job is to answer the user's question directly and helpfully.

GUIDELINES:
1. Grounding: When context is provided, ground your answer in that context. Do not invent or assume facts.
2. Insufficient Context: If the supplied context does not contain enough information to answer the question, state that clearly and accurately.
3. Distinguish Sources: When both internal documents and web results are provided, explicitly distinguish what comes from internal documentation versus external web sources.
4. Citations: When external web information is referenced, include page titles and markdown links [Title](URL).
5. Tone: Be professional, educational, concise, and direct. Avoid meta-commentary about your internal prompt structure.
"""


class KnowledgeAssistant:
    """Orchestrates query routing, retrieval, and response synthesis."""

    def __init__(
        self,
        bm25_engine: Optional[BM25SearchEngine] = None,
        vector_store: Optional[ChromaVectorStore] = None
    ):
        self.bm25_engine = bm25_engine or BM25SearchEngine()
        self.vector_store = vector_store or ChromaVectorStore()

    def ensure_indices_loaded(self) -> None:
        """Attempt to load existing BM25 index from disk if not yet populated."""
        if not self.bm25_engine.chunks:
            self.bm25_engine.load_index()

    def process_query(
        self,
        query: str,
        forced_route: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute complete assistant pipeline for a user question.
        
        Args:
            query: The user prompt.
            forced_route: Optional route override ('direct_llm', 'local_rag', 'web_search', 'both')
            
        Returns:
            Dictionary containing:
                - query: original user query
                - route: selected execution route
                - reason: routing rationale
                - answer: generated answer text
                - internal_sources: list of unique document sources used
                - web_sources: list of web sources {title, url}
        """
        self.ensure_indices_loaded()

        # Step 1: Intelligent Routing
        if forced_route:
            route_info = {
                "route": forced_route,
                "reason": f"Manually selected route: {forced_route}"
            }
        else:
            route_info = route_query(query)

        route = route_info["route"]
        reason = route_info["reason"]

        internal_sources: List[str] = []
        web_sources: List[Dict[str, str]] = []
        answer = ""

        # Step 2: Route Execution
        if route == "direct_llm":
            # No retrieval needed; direct Gemini generation
            prompt = query
            answer = generate_text(
                prompt=prompt,
                system_instruction=FINAL_ANSWER_SYSTEM_PROMPT,
                temperature=0.3
            )

        elif route == "local_rag":
            # Hybrid search retrieval (BM25 + ChromaDB + RRF)
            retrieved_chunks = hybrid_search(
                query=query,
                top_k=DEFAULT_TOP_K,
                bm25_engine=self.bm25_engine,
                vector_store=self.vector_store
            )

            # Extract unique internal document sources
            seen_sources = set()
            for chunk in retrieved_chunks:
                source_name = chunk.get("metadata", {}).get("source")
                if source_name and source_name not in seen_sources:
                    seen_sources.add(source_name)
                    internal_sources.append(source_name)

            # Build context string
            context_blocks = []
            for idx, c in enumerate(retrieved_chunks, 1):
                src = c.get("metadata", {}).get("source", "Unknown")
                context_blocks.append(f"[Document {idx} (Source: {src})]\n{c['text']}")
            
            context_text = "\n\n".join(context_blocks)
            prompt = (
                f"Question:\n{query}\n\n"
                f"INTERNAL KNOWLEDGE BASE CONTEXT:\n{context_text}\n\n"
                f"Provide a thorough and grounded answer based on the internal knowledge above."
            )
            answer = generate_text(
                prompt=prompt,
                system_instruction=FINAL_ANSWER_SYSTEM_PROMPT,
                temperature=0.2
            )

        elif route == "web_search":
            # Web retrieval via Exa AI
            web_results = exa_search(query=query, num_results=EXA_NUM_RESULTS)
            web_sources = [{"title": item["title"], "url": item["url"]} for item in web_results]

            # Build web context string
            web_blocks = []
            for idx, item in enumerate(web_results, 1):
                web_blocks.append(
                    f"[Web Result {idx}: {item['title']}]\nURL: {item['url']}\nSnippet:\n{item['text']}"
                )
            web_context_text = "\n\n".join(web_blocks)

            prompt = (
                f"Question:\n{query}\n\n"
                f"EXTERNAL WEB SEARCH RESULTS:\n{web_context_text}\n\n"
                f"Provide an up-to-date answer citing the web sources and URLs where appropriate."
            )
            answer = generate_text(
                prompt=prompt,
                system_instruction=FINAL_ANSWER_SYSTEM_PROMPT,
                temperature=0.2
            )

        elif route == "both":
            # Retrieve BOTH internal documents and external web results
            retrieved_chunks = hybrid_search(
                query=query,
                top_k=DEFAULT_TOP_K,
                bm25_engine=self.bm25_engine,
                vector_store=self.vector_store
            )
            seen_sources = set()
            for chunk in retrieved_chunks:
                source_name = chunk.get("metadata", {}).get("source")
                if source_name and source_name not in seen_sources:
                    seen_sources.add(source_name)
                    internal_sources.append(source_name)

            internal_blocks = []
            for idx, c in enumerate(retrieved_chunks, 1):
                src = c.get("metadata", {}).get("source", "Unknown")
                internal_blocks.append(f"[Internal Doc {idx} (Source: {src})]\n{c['text']}")
            internal_context = "\n\n".join(internal_blocks)

            web_results = exa_search(query=query, num_results=EXA_NUM_RESULTS)
            web_sources = [{"title": item["title"], "url": item["url"]} for item in web_results]

            web_blocks = []
            for idx, item in enumerate(web_results, 1):
                web_blocks.append(
                    f"[External Web {idx}: {item['title']}]\nURL: {item['url']}\nSnippet:\n{item['text']}"
                )
            web_context = "\n\n".join(web_blocks)

            prompt = (
                f"Question:\n{query}\n\n"
                f"--- INTERNAL KNOWLEDGE BASE ---\n{internal_context}\n\n"
                f"--- CURRENT EXTERNAL WEB SEARCH ---\n{web_context}\n\n"
                f"Synthesize an answer addressing the question. Compare and clearly distinguish between "
                f"our internal documentation and current external industry information. Cite sources appropriately."
            )
            answer = generate_text(
                prompt=prompt,
                system_instruction=FINAL_ANSWER_SYSTEM_PROMPT,
                temperature=0.2
            )

        return {
            "query": query,
            "route": route,
            "reason": reason,
            "answer": answer,
            "internal_sources": internal_sources,
            "web_sources": web_sources
        }
