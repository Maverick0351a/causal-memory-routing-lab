"""Only public data types and feedback features; no benchmark generator imports."""
from dataclasses import dataclass
import numpy as np
from .embedding import HashEmbedder


@dataclass(frozen=True, slots=True)
class Observation:
    bits: tuple[int, int]


def experts(obs):
    x0, x1 = obs.bits
    return np.array([x0 ^ (a & x1) ^ b for a in (0, 1) for b in (0, 1)], dtype=float)


class FeedbackState:
    dimension = 24
    def __init__(self):
        self.fast = np.full(4, 0.5)
        self.slow = np.full(4, 0.5)
        self.embedder = HashEmbedder(16)
        self.embeddings = {bits: self.embedder.encode([f"bit0_{bits[0]} bit1_{bits[1]}"])[0]
                           for bits in ((0, 0), (0, 1), (1, 0), (1, 1))}

    def features(self, obs):
        return np.concatenate([self.embeddings[obs.bits], self.fast - 0.5, self.slow - 0.5])

    def observe(self, obs, label):
        loss = (experts(obs) != label).astype(float)
        self.fast += 0.35 * (loss - self.fast)
        self.slow += 0.04 * (loss - self.slow)
