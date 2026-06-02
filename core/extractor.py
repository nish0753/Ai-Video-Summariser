"""Lightweight extractor stubs for local testing.

These functions provide simple, deterministic outputs so the
`main.py` pipeline can run without external LLM dependencies.
"""

def extract_action_items(transcript: str) -> str:
    if not transcript:
        return "No action items found."
    return "- No explicit action items detected in this transcript."


def extract_key_decisions(transcript: str) -> str:
    if not transcript:
        return "No key decisions found."
    return "- No clear decisions detected in this transcript."


def extract_questions(transcript: str) -> str:
    if not transcript:
        return "No questions found."
    return "- No explicit questions were detected."


def extract_conclusions(transcript: str) -> str:
    if not transcript:
        return "No conclusions found."
    return "- No explicit conclusions detected."