from dataclasses import dataclass, field
from typing import Any
from app.core.documents.elements import DocumentElement


@dataclass(frozen=True)
class Document:
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    elements: tuple[DocumentElement, ...] = ()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
