from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.documents.models import Chunk, Document


class Chunker(ABC):
    """
        Base class for all chunking strategies
    """
    name: str = "unnamed"

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into an ordered list of Chunks."""
        raise NotImplementedError