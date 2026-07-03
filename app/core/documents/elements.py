from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ElementType(str, Enum):
    TITLE = "title"           # doc title / H1 equivalent
    HEADING = "heading"       # H2..H6
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"           # rendered as markdown
    CODE = "code"
    CAPTION = "caption"       # figure / table caption
    FOOTER = "footer"         # page footer / header artifacts we chose to keep


@dataclass(frozen=True)
class DocumentElement:
    """
    A single semantic unit extracted from a source document.
    """
    type: ElementType
    text: str
    # heading_path lets us track hierarchy: e.g. ["Chapter 3", "3.2 Results"]
    heading_path: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    """
    Result of running a loader on raw bytes. Elements are in reading order.
    """
    elements: list[DocumentElement]
    metadata: dict[str, Any] 