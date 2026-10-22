"""Unified LangChain LLM Provider Module.

Supports multi-provider AI routing and generation via LangChain:
  1. Google Gemini (via resilient multi-model fallback chain)
  2. OpenAI (GPT-4o, GPT-4o-mini, etc.)
  3. NVIDIA NIM (Llama 3.1, Mixtral, etc. via build.nvidia.com)
  4. Ollama (Local and Ollama Cloud hosted instances)

Allows dynamic switching between providers without restarting the application.
"""

import warnings
from typing import Optional, Dict, Any, List

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src import config
from src.gemini import generate_text as gemini_generate_text


SUPPORTED_PROVIDERS = {"gemini", "openai", "nvidia", "ollama"}


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """Check readiness of all supported AI providers."""
    return {
        "gemini": {
            "name": "Google Gemini",
            "ready": bool(config.GEMINI_API_KEY and not config.GEMINI_API_KEY.startswith("your_")),
            "model": config.GEMINI_MODEL,
            "type": "Cloud API",
        },
        "openai": {
            "name": "OpenAI",
            "ready": bool(config.OPENAI_API_KEY and not config.OPENAI_API_KEY.startswith("your_")),
            "model": config.OPENAI_MODEL,
            "type": "Cloud API",
        },
        "nvidia": {
            "name": "NVIDIA NIM",
            "ready": bool(config.NVIDIA_API_KEY and not config.NVIDIA_API_KEY.startswith("your_")),
            "model": config.NVIDIA_MODEL,
            "type": "Cloud API",
        },
        "ollama": {
            "name": "Ollama (Local / Cloud)",
            "ready": bool(config.OLLAMA_BASE_URL),
            "model": config.OLLAMA_MODEL,
            "endpoint": config.OLLAMA_BASE_URL,
            "type": "Local / Cloud Endpoint",
        },
    }


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2
) -> Any:
    """Instantiate a LangChain chat model based on provider and model.
    
    Args:
        provider: 'gemini' | 'openai' | 'nvidia' | 'ollama' (default from config.LLM_PROVIDER)
        model_name: Optional model override.
        temperature: Sampling temperature.
        
    Returns:
        A LangChain ChatModel instance, or None for native Gemini handler.
    """
    prov = (provider or config.LLM_PROVIDER).strip().lower()
    if prov not in SUPPORTED_PROVIDERS:
        prov = "gemini"

    if prov == "openai":
        api_key = config.OPENAI_API_KEY
        if not api_key or api_key.startswith("your_"):
            raise ValueError(
                "OPENAI_API_KEY is not configured. Please set it in .env or choose another provider."
            )
        model = model_name or config.OPENAI_MODEL
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature
        )

    elif prov == "nvidia":
        api_key = config.NVIDIA_API_KEY
        if not api_key or api_key.startswith("your_"):
            raise ValueError(
                "NVIDIA_API_KEY is not configured. Get a free key at https://build.nvidia.com/ "
                "or set it in .env."
            )
        model = model_name or config.NVIDIA_MODEL
        base_url = config.NVIDIA_BASE_URL or "https://integrate.api.nvidia.com/v1"
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature
        )

    elif prov == "ollama":
        base_url = (config.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
        # Ollama supports OpenAI-compatible /v1 endpoint
        v1_url = f"{base_url}/v1" if not base_url.endswith("/v1") else base_url
        model = model_name or config.OLLAMA_MODEL
        api_key = config.OLLAMA_API_KEY or "ollama"
        return ChatOpenAI(
            api_key=api_key,
            base_url=v1_url,
            model=model,
            temperature=temperature
        )

    # For gemini, returns None to use native resilient multi-model fallback chain
    return None


def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2
) -> str:
    """Universal text generation function across Gemini, OpenAI, NVIDIA, and Ollama.
    
    Args:
        prompt: User question or task.
        system_instruction: System prompt or persona.
        provider: 'gemini' | 'openai' | 'nvidia' | 'ollama'. Defaults to config.LLM_PROVIDER.
        model_name: Optional model override.
        temperature: Temperature (0.0 - 1.0).
        
    Returns:
        Generated text string.
    """
    if not prompt or not prompt.strip():
        return ""

    prov = (provider or config.LLM_PROVIDER).strip().lower()
    if prov not in SUPPORTED_PROVIDERS:
        prov = "gemini"

    # 1. Google Gemini (Native fallback handler)
    if prov == "gemini":
        return gemini_generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            model_name=model_name or config.GEMINI_MODEL,
            temperature=temperature
        )

    # 2. LangChain ChatOpenAI-compatible providers: OpenAI, NVIDIA, Ollama
    try:
        chat_model = get_llm(provider=prov, model_name=model_name, temperature=temperature)
        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))

        response = chat_model.invoke(messages)
        if response and response.content:
            return str(response.content).strip()
        return ""
    except Exception as err:
        # If external provider fails and Gemini is configured, offer fallback option
        if config.GEMINI_API_KEY and not config.GEMINI_API_KEY.startswith("your_"):
            warnings.warn(
                f"Provider '{prov}' failed with ({err}). Falling back to Google Gemini.",
                UserWarning
            )
            return gemini_generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature
            )
        raise RuntimeError(f"LLM Provider '{prov}' generation failed: {err}") from err
