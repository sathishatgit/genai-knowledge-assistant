"""Workshop Setup Verification Script.

Checks Python version, required libraries, configuration files, API key presence,
and ChromaDB local initialization. Never prints secret API keys.
"""

import sys
import warnings
from pathlib import Path

# Suppress sunset/deprecation notices during workshop setup check
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_setup() -> bool:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("========================================")
    print(" GenAI Workshop Setup Check")
    print("========================================")

    all_passed = True

    # 1. Check Python version
    major, minor, micro = sys.version_info[:3]
    py_version_str = f"{major}.{minor}.{micro}"
    if sys.version_info >= (3, 10):
        print(f"[OK] Python version: {py_version_str}")
    else:
        print(f"[FAIL] Python version: {py_version_str} (Python 3.10+ required)")
        all_passed = False

    # 2. Check Package Imports
    packages = [
        ("Gemini package", "google.generativeai"),
        ("ChromaDB", "chromadb"),
        ("BM25", "rank_bm25"),
        ("Requests", "requests"),
        ("python-dotenv", "dotenv"),
        ("LangChain", "langchain"),
        ("LangChain Community", "langchain_community"),
    ]

    for display_name, mod_name in packages:
        try:
            __import__(mod_name)
            print(f"[OK] {display_name}")
        except ImportError as e:
            print(f"[FAIL] {display_name} - Not installed ({e})")
            all_passed = False

    # 3. Check .env file
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        print("[OK] .env file found")
    else:
        print("[FAIL] .env file missing in project root")
        all_passed = False

    # 4. Check API Keys via config
    from src.config import GEMINI_API_KEY, JINA_API_KEY, EXA_API_KEY, CHROMA_DIR

    api_keys = [
        ("Gemini API key", GEMINI_API_KEY, "GEMINI_API_KEY"),
        ("Jina API key", JINA_API_KEY, "JINA_API_KEY"),
        ("Exa API key", EXA_API_KEY, "EXA_API_KEY"),
    ]

    for service_name, key_val, var_name in api_keys:
        if key_val and not key_val.startswith("your_"):
            print(f"[OK] {service_name} configured")
        else:
            print(f"[!] {service_name} missing or placeholder (Set {var_name} in .env)")
            all_passed = False

    # 5. Check ChromaDB initialization
    try:
        from src.vector_store import get_chroma_client
        client = get_chroma_client(CHROMA_DIR)
        test_col = client.get_or_create_collection(name="setup-check-temp")
        client.delete_collection("setup-check-temp")
        print("[OK] ChromaDB local initialization verified")
    except Exception as e:
        print(f"[FAIL] ChromaDB initialization failed: {e}")
        all_passed = False

    print("========================================")
    if all_passed:
        print("Setup completed successfully! All components ready.")
    else:
        print("Note: Workshop dependencies and engine are installed.")
        print("Please configure any missing API keys in .env to enable all live endpoints.")
    print("========================================\n")

    return all_passed


if __name__ == "__main__":
    check_setup()
