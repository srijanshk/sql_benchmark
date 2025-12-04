"""Helpers for interacting with a local Ollama server."""

from __future__ import annotations

import os
from typing import List, Optional

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def check_ollama_connection(timeout: float = 2.0) -> bool:
    """Return True if Ollama server responds at the configured URL."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def embed_with_ollama(text: str, model: str) -> Optional[List[float]]:
    """Request an embedding vector for the text via Ollama."""
    payload = {"model": model, "prompt": text}
    try:
        response = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("embedding")
    except requests.RequestException:
        return None


def generate_with_ollama(prompt: str, model: str) -> Optional[str]:
    """Generate text completion via Ollama."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response")
    except requests.RequestException:
        return None
