from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import requests


def _strip_thinking(text: str) -> str:
    """Strip chain-of-thought <think>...</think> blocks from text."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


class EmbeddingAdapter:
    """Embedding adapter for MaaS nomic-embed-text-v1-5."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "nomic-embed-text-v1-5",
        timeout: int = 30,
    ):
        self._base_url = (base_url or os.environ.get("LITELLM_API_BASE", "")).rstrip("/")
        self._api_key = api_key or os.environ.get("LITELLM_GPU_API_KEY", "")
        self._model = model
        self._timeout = timeout

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text. Returns 768-dim numpy array."""
        if not text.strip():
            return np.zeros(768)
        # Strip chain-of-thought blocks before embedding
        clean_text = _strip_thinking(text)
        if not clean_text:
            return np.zeros(768)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        last_err = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self._base_url}/v1/embeddings",
                    json={"model": self._model, "input": clean_text[:2000]},
                    headers=headers,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt < 2:
                    import time
                    time.sleep(3 * (attempt + 1))
        else:
            raise last_err
        data = resp.json()
        return np.array(data["data"][0]["embedding"])

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts' embeddings."""
        emb_a = self.embed(text_a)
        emb_b = self.embed(text_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(max(-1.0, min(1.0, np.dot(emb_a, emb_b) / (norm_a * norm_b))))


class MockEmbeddingAdapter:
    """Mock embedding adapter for testing. Produces deterministic embeddings from text hash."""

    def embed(self, text: str) -> np.ndarray:
        import hashlib
        clean_text = _strip_thinking(text)
        if not clean_text:
            return np.zeros(768)
        h = int(hashlib.sha256(clean_text.encode()).hexdigest(), 16)
        rng = np.random.RandomState(h % (2**31))
        return rng.randn(768).astype(np.float64)

    def similarity(self, text_a: str, text_b: str) -> float:
        emb_a = self.embed(text_a)
        emb_b = self.embed(text_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
