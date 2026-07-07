import re
from hashlib import sha256
from app.core.documents.models import Document, Chunk


def split_by_structure(text: str) -> list[str]:
    text = re.sub(r"\n{2,}", "\n\n", text.strip())
    blocks = text.split("\n\n")
    return [b.strip() for b in blocks if b.strip()]


def merge_blocks(blocks: list[str], max_chars: int = 700) -> list[str]:
    chunks = []
    current = ""

    for block in blocks:
        if len(current) + len(block) <= max_chars:
            current += ("\n\n" + block if current else block)
        else:
            chunks.append(current)
            current = block

    if current:
        chunks.append(current)

    return chunks


def add_overlap(chunks: list[str], overlap: int = 120) -> list[str]:
    final_chunks = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            final_chunks.append(chunk)
            continue

        prev_chunk = chunks[i - 1]
        overlap_text = prev_chunk[-overlap:]
        final_chunks.append(overlap_text + "\n\n" + chunk)

    return final_chunks


def chunk_document(document: Document):
    blocks = split_by_structure(document.content)
    merged = merge_blocks(blocks, max_chars=700)
    overlapped = add_overlap(merged, overlap=120)

    chunks = []

    for idx, chunk_text in enumerate(overlapped):
        raw_id = f"{document.document_id}:{idx}:{chunk_text}"
        chunk_id = sha256(raw_id.encode()).hexdigest()

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": idx,
                    "chunk_size": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                },
            )
        )

    return chunks