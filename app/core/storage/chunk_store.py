from typing import Dict
from app.core.documents.models import Chunk


class InMemoryChunkStore:
    def __init__(self):
        self._chunks: Dict[str, Chunk] = {}

    def upsert(self, chunk: Chunk):
        self._chunks[chunk.chunk_id] = chunk
        
    def bulk_upsert(self, chunks: list[Chunk]):
        for chunk in chunks:
            self.upsert(chunk)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def count(self) -> int:
        return len(self._chunks)
