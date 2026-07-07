"""
structural: element-aware chunker with zero embedding cost.

Strategy:
  - Consumes the DocumentElement list produced by the loaders.
  - Splits paragraphs / list items / captions into sentences.
  - Treats HEADING and TITLE elements as "anchors" that must attach to the
    next content unit (so a heading is never a singylar chunk).
  - Treats TABLE and CODE as atomic units that are never split, even at
    the cost of exceeding max_tokens for very large tables.
  - Packs units into token-budgeted chunks with 1-sentence overlap.

Fallback:
  - If the document has no `elements` (e.g. legacy ingest path), delegates
    to the character chunker to guarantee no ingest ever fails on this strategy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from app.core.documents.chunkers.chunker_v0 import chunk_document
from app.core.documents.chunkers import register
from app.core.documents.chunkers.base import Chunker
from app.core.documents.elements import DocumentElement, ElementType
from app.core.documents.models import Chunk, Document

logger = logging.getLogger(__name__)


# --- tuning constants ---
TARGET_TOKENS = 350   
MAX_TOKENS = 512      
MIN_TOKENS = 150      
OVERLAP_SENTENCES = 1 


# SENTENCE SPLITTER
# Defensible regex: splits on `. ! ?` followed by whitespace and an opener
# character. Handles most English prose. 
# Known imperfect cases: Dr., e.g.

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[]|\d+[.\)])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def approx_token_count(text: str) -> int:
    """
    Assumption: 1 token = 4 chars for english text
    Accuracy is fine for chunk budget estimation
    """
    return max(1, len(text) // 4)


# UNIT MODEL

@dataclass(frozen=True)
class _Unit:
    """
    A single atomic packable piece. 
    Important Checks:
      1. Anchors (headings) must stick to the next unit 
      2. Atomics (tables, code) must never be split.
    """
    text: str
    token_count: int
    element_type: ElementType
    heading_path: tuple[str, ...]
    source_page: int | None
    element_index: int
    anchor: bool = False
    atomic: bool = False


# Helper Functions

def elements_to_units(elements: Iterable[DocumentElement]) -> list[_Unit]:
    """
    Flatten elements into sentence-level (or atomic) units in reading order.
    """
    units: list[_Unit] = []
    for idx, el in enumerate(elements):
        page = el.metadata.get("page")

        if el.type in (ElementType.TITLE, ElementType.HEADING):
            units.append(_Unit(
                text=el.text,
                token_count=approx_token_count(el.text),
                element_type=el.type,
                heading_path=el.heading_path,
                source_page=page,
                element_index=idx,
                anchor=True,
            ))
            continue

        if el.type in (ElementType.TABLE, ElementType.CODE):
            units.append(_Unit(
                text=el.text,
                token_count=approx_token_count(el.text),
                element_type=el.type,
                heading_path=el.heading_path,
                source_page=page,
                element_index=idx,
                atomic=True,
            ))
            continue

        # PARAGRAPH / LIST_ITEM / CAPTION / FOOTER → sentence-split
        for sent in split_sentences(el.text):
            units.append(_Unit(
                text=sent,
                token_count=approx_token_count(sent),
                element_type=el.type,
                heading_path=el.heading_path,
                source_page=page,
                element_index=idx,
            ))

    return units


def structural_boundaries(units: list[_Unit]) -> set[int]:
    """
    Hard boundaries derived from structure:
      - Before every TITLE (start of a new top-level section).
      - After every atomic unit (never absorb text after a table into the
        same chunk unless we truly have to).
    """
    breaks: set[int] = set()
    for i, u in enumerate(units):
        if i > 0 and u.element_type == ElementType.TITLE:
            breaks.add(i - 1)
        if u.atomic and i < len(units) - 1:
            breaks.add(i)
    return breaks


def pack_units(
    units: list[_Unit],
    boundary_after: set[int],
    *,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    min_tokens: int = MIN_TOKENS,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> tuple[list[str], list[dict]]:
    """
    Greedy packer. Emits (chunk_texts, chunk_metas) pairs.

    Rules:
      - Never split an atomic unit
      - An anchor (heading) must be followed by a non-anchor unit before
        the chunk can be flushed
      - Prefer to flush at a soft boundary (i in boundary_after) once
        buf_tokens >= min_tokens. Otherwise flush when buf_tokens
        >= target_tokens.
      - Overlap: carry the last `overlap_sentences` non-anchor,
        non-atomic units into the next chunk.
    """
    chunks_text: list[str] = []
    chunks_meta: list[dict] = []
    buf: list[_Unit] = []
    buf_tokens = 0

    def only_anchors(bs: list[_Unit]) -> bool:
        return all(b.anchor for b in bs)

    def flush(*, overlap: bool = True) -> None:
        nonlocal buf, buf_tokens
        if not buf or only_anchors(buf):
            return
        text = "\n".join(u.text for u in buf)
        pages = sorted({u.source_page for u in buf if u.source_page is not None})
        # Deepest heading path found in the buffer wins (most specific location).
        best_path: tuple[str, ...] = ()
        for u in buf:
            if len(u.heading_path) > len(best_path):
                best_path = u.heading_path
        chunks_text.append(text)
        chunks_meta.append({
            "token_count": buf_tokens,
            "pages": pages,
            "heading_path": [h for h in best_path if h],
            "element_range": (buf[0].element_index, buf[-1].element_index),
        })
        # Sentence-level overlap for continuity between chunks.
        carry: list[_Unit] = []
        if overlap and overlap_sentences > 0:
            for u in reversed(buf):
                if not u.anchor and not u.atomic:
                    carry.append(u)
                    if len(carry) >= overlap_sentences:
                        break
            
        buf = carry[::-1]
        buf_tokens = sum(u.token_count for u in buf)

    for i, u in enumerate(units):
        # Oversized atomic — must be its own chunk regardless of budget.
        if u.atomic and u.token_count > max_tokens:
            flush(overlap=False)
            buf.append(u)
            buf_tokens += u.token_count
            flush(overlap=False)
            continue

        # Adding this unit would exceed max_tokens, hten flush first
        # But check if current buffer is only anchors, then no need 
        # to fulsh with headings at the end.
        if (
            buf_tokens + u.token_count > max_tokens
            and buf
            and not only_anchors(buf)
        ):
            flush()

        buf.append(u)
        buf_tokens += u.token_count

        # Do not flush on an anchor — headings must attach forward.
        if buf and buf[-1].anchor:
            continue

        at_boundary = i in boundary_after
        if buf_tokens >= target_tokens or (at_boundary and buf_tokens >= min_tokens):
            flush()

    flush(overlap=False)
    return chunks_text, chunks_meta


def build_chunks_from_packed(
    document: Document,
    chunks_text: list[str],
    chunks_meta: list[dict],
    chunker_name: str,
) -> list[Chunk]:
    """
    Convert (texts, metas) into Chunk objects with stable IDs, heading
    context prefix, and provenance metadata.
    """
    chunks: list[Chunk] = []
    for idx, (text, meta) in enumerate(zip(chunks_text, chunks_meta)):
        heading_path = meta.get("heading_path") or []
        prefix = ""
        if heading_path:
            # Prepending the heading path measurably improves retrieval on
            # hierarchical docs — chunks retrieved from deep sections get
            # anchored to their location for the LLM.
            prefix = "Context: " + " › ".join(heading_path) + "\n\n"
        content = prefix + text

        chunk_id = sha256(
            f"{document.document_id}:{chunker_name}:{idx}:{content}".encode()
        ).hexdigest()

        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            content=content,
            metadata={
                **document.metadata,
                "chunker": chunker_name,
                "chunk_index": idx,
                "token_count": meta["token_count"],
                "heading_path": heading_path,
                "pages": meta["pages"],
                "element_range": meta["element_range"],
            },
        ))
    return chunks


# --- the strategy ----------------------------------------------------------

@register("structural_v1")
class StructuralChunker(Chunker):
    name = "structural_v1"

    def __init__(self, **_ignored) -> None:
        # ignore embed_fn etc.
        pass

    def chunk(self, document: Document) -> list[Chunk]:
        # If the loader didn't provide structured elements
        # (legacy path, unsupported file type), fall back to character
        # chunking(v0) rather than producing zero chunks.
        if not document.elements:
            logger.info(
                "structural_v1_fallback_to_character",
                extra={"document_id": document.document_id},
            )
            fallback = chunk_document(document)
            return [
                Chunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    metadata={
                        **c.metadata,
                        "chunker": self.name,
                        "fallback": "character_v1",
                    },
                )
                for c in fallback
            ]

        units = elements_to_units(document.elements)
        if not units:
            return []

        boundaries = structural_boundaries(units)
        texts, metas = pack_units(units, boundaries)
        return build_chunks_from_packed(document, texts, metas, self.name)
