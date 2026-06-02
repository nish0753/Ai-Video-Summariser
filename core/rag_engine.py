from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any
import numpy as np
import os
import requests
from dotenv import load_dotenv

# Ensure we have the dot env loaded
load_dotenv()

def get_mistral_embeddings(texts: List[str]) -> np.ndarray:
    """Fetch embeddings from Mistral API for a list of text chunks."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing. Required for embeddings.")

    api_url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    payload = {
        "model": "mistral-embed",
        "input": texts
    }
    
    resp = requests.post(api_url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    # Extract the embeddings and ensure they are ordered correctly
    # Mistral returns data array where each element has an 'embedding' array
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    embeddings = [item["embedding"] for item in sorted_data]
    
    return np.array(embeddings)


def cosine_similarity_manual(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one vector and a matrix of vectors.
    Replaces sklearn's cosine_similarity to remove the heavy scikit-learn dependency.
    vec1 shape: (1, D)
    vec2 shape: (N, D)
    """
    # Normalize vec1
    v1_norm = np.linalg.norm(vec1, axis=1, keepdims=True)
    v1_normalized = vec1 / (v1_norm + 1e-10)
    
    # Normalize vec2
    v2_norm = np.linalg.norm(vec2, axis=1, keepdims=True)
    v2_normalized = vec2 / (v2_norm + 1e-10)
    
    # Dot product
    similarity = np.dot(v1_normalized, v2_normalized.T)
    return similarity


@dataclass
class RagChain:
    transcript: str
    chunks: List[str]
    chunk_embeddings: Any # numpy array


def _split_transcript(transcript: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    """Split transcript into overlapping character chunks for retrieval."""
    if not transcript:
        return []

    chunks: List[str] = []
    start = 0
    length = len(transcript)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = transcript[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(0, end - overlap)

    return chunks


def build_rag_chain(transcript: str) -> RagChain:
    """Build a proper RAG retriever object using semantic embeddings.

    Chunks the transcript and fetches embeddings from Mistral API.
    """
    chunks = _split_transcript(transcript)
    
    if not chunks:
        return RagChain(transcript=transcript, chunks=[], chunk_embeddings=np.array([]))

    # Compute semantic embeddings for all chunks via API
    # To avoid payload too large errors, we can batch it, but Mistral embed usually accepts 
    # large batches. We'll batch by 50 chunks just to be safe.
    all_embeddings = []
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embs = get_mistral_embeddings(batch)
        all_embeddings.extend(batch_embs)
        
    chunk_embeddings = np.array(all_embeddings)
    
    return RagChain(transcript=transcript, chunks=chunks, chunk_embeddings=chunk_embeddings)


def ask_question(rag_chain: RagChain, chat_history: list) -> str:
    """Answer a question by retrieving relevant chunks via embeddings and using Mistral to synthesize an answer."""
    if not rag_chain or not rag_chain.chunks:
        return "I couldn't find the answer in the meeting transcript."

    if not chat_history:
        return "Please ask a question."

    # The last message is the current question
    question = chat_history[-1]["content"]

    # 1. Retrieve relevant chunks using Semantic Similarity
    question_embedding = get_mistral_embeddings([question])
    
    # Compute cosine similarity
    scores = cosine_similarity_manual(question_embedding, rag_chain.chunk_embeddings).flatten()

    evidence = []
    if scores.size > 0:
        # Get top 5 most relevant chunks
        top_indices = scores.argsort()[::-1][:5]
        for idx in top_indices:
            if scores[idx] > 0.15: # Threshold for semantic similarity
                evidence.append(rag_chain.chunks[idx].strip())

    context = "\n\n".join(evidence) if evidence else "No direct matching segments found in transcript, but use your best judgment."

    # 2. Call Mistral API to generate a conversational answer
    api_key = os.getenv("MISTRAL_API_KEY")
    api_url = "https://api.mistral.ai/v1/chat/completions"
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    if not api_key:
        return f"*(Mistral API key not found. Showing raw transcript chunks instead)*\n\n{context}"

    system_prompt = (
        "You are an intelligent, helpful assistant answering questions about a video/meeting transcript. "
        "Use the provided transcript context to answer the user's question. "
        "Be conversational, clear, and direct. If the context doesn't contain the answer, say you don't know based on the transcript, but you can infer if it's general knowledge. "
        f"\n\n=== RELEVANT TRANSCRIPT CONTEXT ===\n{context}\n==================================="
    )

    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in chat_history[-5:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(api_url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error contacting Mistral LLM: {str(e)}\n\n*(Raw Context Found)*\n{context}"
