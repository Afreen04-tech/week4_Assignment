import os
import sys
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Initialize Gemini Client if available
_gemini_client = None
if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_"):
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        _gemini_client = None

# Initialize OpenAI Client if available
_openai_client = None
if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your_"):
    try:
        import openai
        _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        _openai_client = None


def generate_llm_completion(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    prefer_provider: str = "gemini",
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a completion against Gemini or OpenAI using the provided credentials in .env.
    Falls back gracefully if quota is exhausted or if network is unavailable.
    """
    # 1. Try Gemini
    if prefer_provider == "gemini" and _gemini_client is not None:
        target_model = model or "gemini-flash-latest"
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p if top_p is not None else 0.95
            )
            if system_instruction:
                config.system_instruction = system_instruction

            response = _gemini_client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config
            )
            return {
                "success": True,
                "provider": "gemini",
                "model": target_model,
                "text": response.text.strip(),
                "error": None
            }
        except Exception as err:
            pass  # Fall through to OpenAI or simulator

    # 2. Try OpenAI
    if _openai_client is not None:
        target_model = model or "gpt-4o-mini"
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            kwargs = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature
            }
            if top_p is not None:
                kwargs["top_p"] = top_p

            resp = _openai_client.chat.completions.create(**kwargs)
            return {
                "success": True,
                "provider": "openai",
                "model": target_model,
                "text": resp.choices[0].message.content.strip(),
                "error": None
            }
        except Exception as err:
            return {
                "success": False,
                "provider": "openai",
                "model": target_model,
                "text": None,
                "error": str(err)
            }

    return {
        "success": False,
        "provider": "none",
        "model": "simulation_fallback",
        "text": None,
        "error": "No active provider available; use simulation."
    }

