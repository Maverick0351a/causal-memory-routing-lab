"""Local embedding protocol; deterministic feature hashing requires no weights."""
import hashlib
import re
from typing import Protocol, Sequence
import numpy as np


class Embedder(Protocol):
    dimension: int
    model_id: str
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashEmbedder:
    """Stable signed token hashing, not a learned semantic embedding model."""
    def __init__(self, dimension=16):
        if dimension < 2:
            raise ValueError("dimension must be >=2")
        self.dimension = dimension
        self.model_id = f"blake2-token-v1-{dimension}"

    def encode(self, texts):
        out = np.zeros((len(texts), self.dimension), dtype=np.float64)
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=16).digest()
                j = int.from_bytes(digest[:8], "little") % self.dimension
                out[i, j] += 1 if digest[8] % 2 else -1
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out
