import hashlib
import math
from typing import List, Protocol

from rag_platform.core.config import settings


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> List[float]:
        ...


class DeterministicEmbeddingProvider:
    """Local deterministic embedding for development and tests.

    Production should replace this with a managed embedding provider and keep the interface stable.
    """

    def __init__(self, dimensions: int = settings.embedding_dim) -> None:
        self.dimensions = dimensions

    async def embed(self, text: str) -> List[float]:
        values: List[float] = []
        counter = 0
        while len(values) < self.dimensions:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            for byte in digest:
                values.append((byte / 255.0) - 0.5)
                if len(values) == self.dimensions:
                    break
            counter += 1
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

