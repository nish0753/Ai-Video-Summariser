import os
import requests
from typing import Optional


def summarize_with_mistral(
    transcript: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Call Mistral's chat completions endpoint to summarize a transcript.

    Reads MISTRAL_API_KEY from environment variables.
    Falls back to mistral-small-latest.
    """
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    api_url = "https://api.mistral.ai/v1/chat/completions"
    model   = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY must be set to use the Mistral provider.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert meeting analyst. Your job is to produce a concise, "
                    "well-structured summary of the provided transcript. Focus on key points, "
                    "decisions, and outcomes. Write in clear, professional English."
                ),
            },
            {
                "role": "user",
                "content": f"Please summarize the following transcript:\n\n{transcript[:8000]}",
            },
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    resp = requests.post(api_url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        pass

    return str(data)


def translate_to_english(
    text: str,
    model: str = None,
    api_key: str = None,
) -> str:
    """Translate any non-English text to English using Mistral LLM.

    Used for the 'Hindi to English' pipeline mode:
    Groq Whisper transcribes Hindi audio -> Hindi text -> this function -> English text.

    Splits long texts into 6000-character chunks and translates each,
    then joins them back together.
    """
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    api_url  = "https://api.mistral.ai/v1/chat/completions"
    model    = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY must be set to use the Mistral translation provider.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Split into 6000-char chunks so we never hit the context window limit
    chunk_size = 6000
    parts = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    translated_parts = []

    for part in parts:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional Hindi-to-English translator. "
                        "Translate the provided Hindi text to clear, natural English. "
                        "Preserve the meaning, tone, and structure as closely as possible. "
                        "Output only the translated English text — no commentary, no explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Translate to English:\n\n{part}",
                },
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        resp = requests.post(api_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        try:
            translated_parts.append(data["choices"][0]["message"]["content"].strip())
        except (KeyError, IndexError, TypeError):
            translated_parts.append(part)  # fallback: keep original if translation fails

    return " ".join(translated_parts)


__all__ = ["summarize_with_mistral", "translate_to_english"]
