"""Main CLI Interactive Interface for Intelligent GenAI Knowledge Assistant.

Provides interactive question answering with intelligent route selection
(direct_llm, local_rag, web_search, both) and clear source citations.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import KnowledgeAssistant
from src.bm25_search import BM25SearchEngine
from src.vector_store import ChromaVectorStore


def display_result(result: dict) -> None:
    """Format and print response according to workshop requirements."""
    route = result.get("route", "UNKNOWN").upper().replace("_", " ")
    reason = result.get("reason", "")
    answer = result.get("answer", "")
    internal_sources = result.get("internal_sources", [])
    web_sources = result.get("web_sources", [])

    print("\n----------------------------------------")
    print(f"Route: {route}")
    if reason:
        print(f"Reason: {reason}")
    print("----------------------------------------")

    # Display Sources
    if internal_sources and not web_sources:
        print("\nRetrieved Sources:")
        for src in internal_sources:
            print(f"- {src}")

    elif web_sources and not internal_sources:
        print("\nWeb Sources:")
        for item in web_sources:
            title = item.get("title", "External Link")
            url = item.get("url", "")
            print(f"- {title}")
            if url:
                print(f"  {url}")

    elif internal_sources and web_sources:
        print("\nInternal Sources:")
        for src in internal_sources:
            print(f"- {src}")
        print("\nWeb Sources:")
        for item in web_sources:
            title = item.get("title", "External Link")
            url = item.get("url", "")
            print(f"- {title}")
            if url:
                print(f"  {url}")

    print("\nAnswer:")
    print(answer)
    print("========================================\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Intelligent GenAI Knowledge Assistant CLI")
    parser.add_argument("--query", "-q", type=str, help="Run a single question non-interactively")
    parser.add_argument("--route", "-r", type=str, choices=["direct_llm", "local_rag", "web_search", "both"],
                        help="Force a specific execution route")
    args = parser.parse_args()

    # Initialize assistant
    bm25 = BM25SearchEngine()
    vector_store = ChromaVectorStore()
    assistant = KnowledgeAssistant(bm25_engine=bm25, vector_store=vector_store)

    # If single query provided via arguments
    if args.query:
        print(f"You: {args.query}")
        result = assistant.process_query(args.query, forced_route=args.route)
        display_result(result)
        return

    # Interactive Loop
    print("========================================")
    print(" Intelligent GenAI Knowledge Assistant")
    print("========================================")
    print("Ask a question.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "q"}:
                print("Exiting. Have a great day!")
                break

            result = assistant.process_query(user_input, forced_route=args.route)
            display_result(result)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\n[!] An error occurred while processing your request: {e}\n")


if __name__ == "__main__":
    main()
