# app/core/documents/chunkers/character.py
"""
character_v1: the original character-based chunker.

Wraps the existing `chunk_document` function from app.core.documents.chunker
without modifying it. Kept as the baseline for A/B evaluation against the
newer structural / semantic strategies. 
"""
from __future__ import annotations

from app.core.documents.chunkers.chunker_v0 import chunk_document
from app.core.documents.chunkers import register
from app.core.documents.chunkers.base import Chunker
from app.core.documents.models import Chunk, Document


@register("character_v1")
class CharacterChunker(Chunker):
    name = "character_v1"

    def __init__(self, **_ignored) -> None:
        # Accept and ignore any kwargs (e.g. embed_fn) so the pipeline can
        # pass strategy-agnostic helpers uniformly.
        pass

    def chunk(self, document: Document) -> list[Chunk]:
        raw_chunks = chunk_document(document)

        # Rebuild each Chunk with a `chunker` detail in metadata.
        # Chunk is frozen, so we can't mutate; we return a new list.
        tagged: list[Chunk] = []
        for c in raw_chunks:
            tagged.append(
                Chunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    metadata={**c.metadata, "chunker": self.name},
                )
            )
        return tagged
