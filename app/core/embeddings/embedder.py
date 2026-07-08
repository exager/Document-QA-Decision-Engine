from sentence_transformers import SentenceTransformer
from typing import List
from typing import Dict
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

class InMemoryEmbeddingCache:
    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, chunk_id: str) -> np.ndarray | None:
        return self._cache.get(chunk_id)

    def set(self, chunk_id: str, embedding: np.ndarray):
        self._cache[chunk_id] = embedding

    def bulk_get(self, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        return {
            cid: self._cache[cid] for cid in chunk_ids if cid in self._cache
        }

    def count(self) -> int:
        return len(self._cache)