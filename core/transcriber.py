import os
import requests
from dotenv import load_dotenv

load_dotenv()

TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

def load_model(model_name: str = None):
    """Stub function to maintain compatibility with streamlit_app.py
    Since we are using an API now, there is no local model to load.
    """
    pass

def transcribe_chunk(chunk_path: str, translate: bool = False, language: str = None) -> str:
    """Transcribe a single audio chunk using Groq's Whisper API.

    Args:
        chunk_path: Path to the audio chunk file.
        translate:  Unused. Kept for backward compatibility with streamlit_app.py.
                    Hindi-to-English translation is handled by Mistral after transcription.
        language:   Language hint e.g. 'hi' for Hindi/Hinglish content.

    Returns:
        Transcribed text string (in the source language).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env. Required for fast API transcription.")

    headers = {"Authorization": f"Bearer {api_key}"}

    data = {
        "model": "whisper-large-v3",
        "response_format": "json"
    }

    if language:
        data["language"] = language

    # Detect correct MIME type from the chunk file extension
    ext = os.path.splitext(chunk_path)[1].lower()
    mime_map = {
        ".m4a":  "audio/mp4",
        ".mp4":  "audio/mp4",
        ".wav":  "audio/wav",
        ".mp3":  "audio/mpeg",
        ".webm": "audio/webm",
        ".ogg":  "audio/ogg",
    }
    mime_type = mime_map.get(ext, "audio/mp4")

    import time

    MAX_RETRIES = 3
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # The file needs to be sent as multipart/form-data
            with open(chunk_path, "rb") as f:
                files = {
                    "file": (os.path.basename(chunk_path), f, mime_type)
                }
                resp = requests.post(
                    TRANSCRIPTION_URL,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=300,  # 5 minutes per chunk
                )
            resp.raise_for_status()
            return resp.json().get("text", "")
        except requests.exceptions.Timeout as e:
            last_error = e
            print(f"Chunk timed out (attempt {attempt}/{MAX_RETRIES}). Retrying in 5s...")
            time.sleep(5)
        except requests.exceptions.HTTPError as e:
            last_error = e
            print(f"HTTP error on attempt {attempt}/{MAX_RETRIES}: {e}. Retrying in 5s...")
            time.sleep(5)

    raise RuntimeError(f"Failed to transcribe chunk after {MAX_RETRIES} attempts: {last_error}")


def transcribe_all(chunks: list, translate: bool = False, language: str = "english") -> str:
    """Transcribe a list of audio chunk file paths.

    Language logic:
      - 'english'  → no hint, standard transcription
      - 'hinglish' → language='hi', translate=False
    """
    lang_hint = "hi" if language.lower() == "hinglish" else None
    translate  = False 

    full_transcription = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)} via Groq API...")
        text = transcribe_chunk(chunk, translate=translate, language=lang_hint)
        full_transcription += text + " "

    print("Transcription complete.")
    return full_transcription.strip()


__all__ = ["load_model", "transcribe_chunk", "transcribe_all"]
