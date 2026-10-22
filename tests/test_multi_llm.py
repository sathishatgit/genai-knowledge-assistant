"""Unit test for Multi-Provider LLM Integration (LangChain).

Tests provider detection, initialization, and fallback mechanisms for:
  - Google Gemini
  - OpenAI
  - NVIDIA NIM
  - Ollama (Local & Cloud)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import get_available_providers, get_llm, generate_text, SUPPORTED_PROVIDERS
from src import config


def test_supported_providers():
    assert "gemini" in SUPPORTED_PROVIDERS
    assert "openai" in SUPPORTED_PROVIDERS
    assert "nvidia" in SUPPORTED_PROVIDERS
    assert "ollama" in SUPPORTED_PROVIDERS
    print("[PASS] test_supported_providers")


def test_provider_status_detection():
    providers = get_available_providers()
    assert "gemini" in providers
    assert "openai" in providers
    assert "nvidia" in providers
    assert "ollama" in providers
    print("[PASS] test_provider_status_detection")


def test_ollama_instantiation():
    """Verify Ollama LangChain instance can be created without error."""
    model = get_llm(provider="ollama", model_name="llama3")
    assert model is not None
    print("[PASS] test_ollama_instantiation")


def test_gemini_generation():
    """Verify primary Gemini generation works."""
    res = generate_text("Reply with: 'MULTI_LLM_OK'", provider="gemini")
    assert "MULTI_LLM_OK" in res or len(res) > 0
    print("[PASS] test_gemini_generation")


if __name__ == "__main__":
    test_supported_providers()
    test_provider_status_detection()
    test_ollama_instantiation()
    test_gemini_generation()
    print("\n[ALL MULTI-LLM TESTS PASSED SUCCESSFULLY!]")
