"""Turn text into vectors so similar meanings end up close together."""
import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: list) -> list:
        """Return one vector (list of floats) per text."""


class SentenceTransformerEmbedder:
    """Runs a small open-source embedding model locally. No API key or cost."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list) -> list:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


class HashingEmbedder:
    """Keyword-based stand-in: each word is hashed into one of `dims` buckets.

    It knows nothing about meaning, so it is only for tests and offline smoke runs.
    """

    def __init__(self, dims: int = 512):
        self.name = f"hashing-{dims}"
        self._dims = dims

    def embed(self, texts: list) -> list:
        vectors = []
        for text in texts:
            vector = [0.0] * self._dims
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dims
                vector[bucket] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


def make_embedder(settings) -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder()
    if settings.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embed_model)
    raise ValueError(f"Unknown embedder: {settings.embedder}")
