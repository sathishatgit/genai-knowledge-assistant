"""Configuration module for GenAI Knowledge Assistant.

Loads environment variables, manages default settings, and ensures sensitive
keys are never printed or logged.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.json"

# Load environment variables from .env if present
load_dotenv(dotenv_path=BASE_DIR / ".env")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
EXA_API_KEY = os.getenv("EXA_API_KEY", "").strip()

# Model Configurations
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3").strip()

# Chunking Parameters
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# Retrieval Parameters
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Exa Search Parameters
EXA_NUM_RESULTS = int(os.getenv("EXA_NUM_RESULTS", "3"))


def validate_api_key(service_name: str, key_value: str) -> None:
    """Validate that a required API key is set without revealing its value."""
    if not key_value or key_value.startswith("your_"):
        raise ValueError(
            f"Missing required API key: {service_name}. "
            f"Please set {service_name} in your .env file or environment variables."
        )


def check_all_keys(verbose: bool = False) -> dict:
    """Check the status of all required keys without exposing them.
    
    Returns a dictionary mapping service names to boolean presence flags.
    """
    status = {
        "GEMINI_API_KEY": bool(GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_")),
        "JINA_API_KEY": bool(JINA_API_KEY and not JINA_API_KEY.startswith("your_")),
        "EXA_API_KEY": bool(EXA_API_KEY and not EXA_API_KEY.startswith("your_")),
    }
    if verbose:
        for key, exists in status.items():
            print(f"[{'✓' if exists else '✗'}] {key}: {'Configured' if exists else 'Missing'}")
    return status
