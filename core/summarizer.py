from typing import Optional
from .llm_providers import summarize_with_mistral


def summarize(transcript: str, provider: Optional[str] = None) -> str:
    """Summarize the transcript.

    If `provider` == 'mistral' and environment variable `MISTRAL_API_KEY`
    is set, attempt to use Mistral; otherwise fall back to a
    simple local summarizer.
    """
    if not transcript:
        return ""

    if provider == "mistral":
        try:
            return summarize_with_mistral(transcript)
        except Exception as e:
            # If Mistral fails, print error and fall back to local summarizer
            print(f"Mistral summary failed: {e}")
            pass

    # Local fallback summarizer
    return transcript[:500] + ("..." if len(transcript) > 500 else "")


def generate_title(transcript: str) -> str:
    """Tiny title generator: picks the first sentence-like chunk."""
    if not transcript:
        return "Untitled Meeting"
    # split on common sentence boundaries
    for sep in ("\n", ". ", "! ", "? "):
        parts = transcript.split(sep)
        if parts and parts[0].strip():
            title = parts[0].strip()
            return title[:60]
    return transcript[:60]

__all__ = ["summarize", "generate_title"]
