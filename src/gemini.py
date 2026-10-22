import warnings
from typing import Optional

# Suppress library deprecation notice during CLI workshop runs
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL, validate_api_key

_configured_key = None


def _ensure_gemini_configured(api_key: Optional[str] = None) -> None:
    """Configure genai with the active API key."""
    global _configured_key
    key = api_key if api_key is not None else GEMINI_API_KEY
    validate_api_key("GEMINI_API_KEY", key)
    
    if _configured_key != key:
        genai.configure(api_key=key)
        _configured_key = key


def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None
) -> str:
    """Generate text using Google Gemini.
    
    Args:
        prompt: The user or task prompt.
        system_instruction: Optional system instruction / persona.
        model_name: Optional Gemini model name override (default from config).
        temperature: Sampling temperature (0.0 for deterministic, 0.2 default).
        api_key: Optional API key override.
        
    Returns:
        Generated text string.
    """
    if not prompt or not prompt.strip():
        return ""

    _ensure_gemini_configured(api_key)
    primary_model = model_name or GEMINI_MODEL
    candidate_models = [primary_model, "gemini-3.5-flash", "gemini-3.1-flash-lite-preview", "gemini-3.8-flash"]
    # De-duplicate while preserving order
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_error = None
    for model_candidate in models_to_try:
        try:
            model = genai.GenerativeModel(
                model_name=model_candidate,
                system_instruction=system_instruction,
                generation_config={"temperature": temperature}
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as err:
            last_error = err
            err_str = str(err)
            if "429" in err_str or "quota" in err_str.lower() or "404" in err_str:
                # Try next model candidate
                continue
            raise RuntimeError(f"Gemini API generation error: {err}") from err

    raise RuntimeError(f"All available Gemini models reached quota limit: {last_error}") from last_error
