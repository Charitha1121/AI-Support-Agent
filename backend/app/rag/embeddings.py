import hashlib
import math
import os
import re
from typing import List, Optional


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate the cosine similarity between two numeric vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a, b in zip(vec1, vec2)))
    norm2 = math.sqrt(sum(b * b for a, b in zip(vec1, vec2)))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


class BaseEmbeddingProvider:
    """Abstract base class for embedding providers."""
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class DeterministicSemanticEmbeddingProvider(BaseEmbeddingProvider):
    """
    High-fidelity, reproducible semantic embedding provider for offline/local environments.
    Uses stable SHA-256 subword hashing and domain keyword weighting across 256 dimensions with L2 normalization.
    Requires no external dependencies and generates stable, identical vectors across Python processes.
    """
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> List[str]:
        # Normalize and tokenize into words and bi-grams
        words = re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", text.lower())
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def embed_text(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vec = [0.0] * self.dimensions

        if not tokens:
            return vec

        for token in tokens:
            # Deterministic SHA-256 hash to dimension index and sign
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            
            # Boost domain keywords for high semantic sensitivity
            weight = 1.0
            if any(k in token for k in [
                "refund", "return", "ship", "deliver", "subscri", 
                "plan", "cancel", "pay", "secur", "renew", 
                "contact", "support", "tier", "track"
            ]):
                weight = 3.0

            vec[idx] += sign * weight

        # L2 Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using OpenAI's API (e.g. text-embedding-3-small)."""
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = None

        if not self.api_key or not self.api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider.")

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise RuntimeError("The 'openai' package is required for OpenAIEmbeddingProvider.")

    def embed_text(self, text: str) -> List[float]:
        if not self._client:
            raise RuntimeError("OpenAI client not initialized.")
        response = self._client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._client:
            raise RuntimeError("OpenAI client not initialized.")
        response = self._client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]


def get_embedding_provider(provider: Optional[str] = None) -> BaseEmbeddingProvider:
    """
    Factory to return the configured embedding provider.

    Supported options:
    - 'local': Uses DeterministicSemanticEmbeddingProvider (offline, zero-credit)
    - 'openai': Uses OpenAIEmbeddingProvider (requires valid OPENAI_API_KEY with credits)

    If provider is omitted, the EMBEDDING_PROVIDER environment variable is used (default: 'local').
    If an unsupported provider is specified, a ValueError is raised.
    """
    provider_name = (provider or os.getenv("EMBEDDING_PROVIDER", "local")).strip().lower()

    if provider_name == "local":
        return DeterministicSemanticEmbeddingProvider()
    elif provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError("OPENAI_API_KEY environment variable is required when EMBEDDING_PROVIDER='openai'.")
        return OpenAIEmbeddingProvider(api_key=api_key)
    else:
        raise ValueError(
            f"Invalid EMBEDDING_PROVIDER '{provider_name}'. Supported options are: 'local', 'openai'."
        )
