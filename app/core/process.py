import logging
import numpy as np
from typing import List
from app.core.documents.models import Document
from app.core.documents.chunkers import get_chunker
from app.core.state import state
from app.core.storage.vector_index import InMemoryVectorIndex


logger = logging.getLogger(__name__)

def _resolve_chunker():
    """
    Pick the chunking strategy from settings (default: character_v1).
    
    """
    strategy_name = getattr(state.settings, "chunker_strategy", "character_v1")
    return get_chunker(
        strategy_name,
        embed_fn=state.embedder.embed_texts,
    )


def process_document(documents: List[Document], request_id: str):
    chunker = _resolve_chunker()
    logger.info(
        "chunker_selected",
        extra = {"request_id": request_id, "chunker": chunker.name}
    )
    chunks = []
    for document in documents:
        parts = chunker.chunk(document)
        chunks.extend(parts)

        logger.info(
            "document_chunked",
            extra={
                "request_id": request_id,
                "document_id": document.document_id,
                "chunk_count": len(chunks),
            },
        )

    # Vectorization and Persistence 
    texts = []
    id_list = []
    for chunk in chunks:
        if state.embedding_cache.get(chunk.chunk_id) is None:
            texts.append(chunk.content)
            id_list.append(chunk.chunk_id)
    
    if texts:
        embeddings = state.embedder.embed_texts(texts)
        for cid, emb in zip(id_list, embeddings):
            state.embedding_cache.set(cid, emb)

        if state.vector_index is None:
            state.vector_index = InMemoryVectorIndex(dim=embeddings.shape[1])
        state.vector_index.add(np.vstack(embeddings), id_list)

    logger.info(
        "chunks_embedded_and_indexed",
        extra={
            "request_id": request_id,
            "embedded_chunks": len(chunks),
            "vector_index_size": len(state.vector_index.chunk_ids),
        },
    )

    state.chunk_store.bulk_upsert(chunks)
    logger.info(
        "chunks_stored",
        extra={
            "request_id": request_id,
            "stored_chunk_count": len(chunks),
            "total_chunks_in_store": state.chunk_store.count(),
        },
    )
    return len(id_list)