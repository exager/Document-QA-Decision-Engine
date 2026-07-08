# app/core/documents/chunkers/semantic.py
"""
semantic_v1: hybrid structural + semantic chunker.

Combines the structural constraints of structural_v1 (never split atomics,
never orphan headings) with semantic-similarity boundaries derived from
embedding cosine distances between adjacent sentences.

Boundary rule (z-score):
    distance[i]  = 1 - cos(embedding[i], embedding[i+1])
    threshold    = mean(distance) + Z_SCORE * std(distance)
    boundary at i iff distance[i] >= threshold

Fallbacks (robustness):
  - No document.elements       → delegate to character chunker.
  - <4 splittable sentences    → behave exactly like structural_v1 (semantic signal is too noisy to use).
  - embed_fn missing or errors → behave exactly like structural_v1.
"""
from __future__ import annotations

import logging

import numpy as np

from app.core.documents.chunkers.chunker_v0 import chunk_document
from app.core.documents.chunkers import register
from app.core.documents.chunkers.base import Chunker
from app.core.documents.chunkers.structural import (
    build_chunks_from_packed,
    elements_to_units,
    pack_units,
    structural_boundaries,
    _Unit,
)
from app.core.documents.models import Chunk, Document

logger = logging.getLogger(__name__)

Z_SCORE = 1.0             # Just a multiplier for us: higher -> fewer semantic breaks
MIN_SPLITTABLE_UNITS = 4  # below this, semantic signal is unreliable


@register("semantic_v1")
class SemanticChunker(Chunker):
    name = "semantic_v1"

    def __init__(self, embed_fn=None, **_ignored) -> None:
        """
        Args:
            embed_fn: callable[list[str]] -> np.ndarray of shape (N, D).
                      Embeddings are expected L2-normalized,
                      so cosine similarity == dot product. If None, this
                      strategy silently falls back to structural behavior.
        """
        self._embed_fn = embed_fn

    def chunk(self, document: Document) -> list[Chunk]:
        # --- fallback 1: no structured elements: character chunker
        if not document.elements:
            logger.info(
                "semantic_v1_fallback_to_character",
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
        semantic_hits = 0

        # try to add semantic boundaries
        splittable_idx = [
            i for i, u in enumerate(units) if not u.anchor and not u.atomic
        ]

        if self._embed_fn is None:
            logger.debug(
                "semantic_v1_no_embed_fn_using_structural_only",
                extra={"document_id": document.document_id},
            )
        elif len(splittable_idx) < MIN_SPLITTABLE_UNITS:
            logger.debug(
                "semantic_v1_too_few_units_using_structural_only",
                extra={
                    "document_id": document.document_id,
                    "splittable_units": len(splittable_idx),
                },
            )
        else:
            try:
                semantic_hits = self._add_semantic_boundaries(
                    units, splittable_idx, boundaries
                )
            except Exception as exc:
                # Robustness: never let an embedding failure break ingest.
                logger.warning(
                    "semantic_v1_embed_failed_falling_back_to_structural",
                    extra={
                        "document_id": document.document_id,
                        "error": str(exc),
                    },
                )

        texts, metas = pack_units(units, boundaries)
        # Annotate chunks with how many semantic boundaries fired — useful
        # in eval to distinguish "semantic actually mattered here" from
        # "semantic degenerated to structural on this doc".
        for m in metas:
            m["semantic_boundaries"] = semantic_hits
        return build_chunks_from_packed(document, texts, metas, self.name)


    # --- Helpers ------------------------------------------

    def _add_semantic_boundaries(
        self,
        units: list[_Unit],
        splittable_idx: list[int],
        boundaries: set[int],
    ) -> int:
        """
        Compute cosine-distance z-scores between adjacent splittable units
        and add high-distance points as soft boundaries. Returns the number
        of semantic boundaries added.
        """
        texts = [units[i].text for i in splittable_idx]
        embs = self._embed_fn(texts)                    # (N, D)
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim != 2 or embs.shape[0] != len(texts):
            raise ValueError(
                f"embed_fn returned unexpected shape {embs.shape} for "
                f"{len(texts)} inputs"
            )

        # Cosine similarity for consecutive pairs — embeddings are already
        # L2-normalized in the project, so dot product == cosine.
        # Used einstein sum (optimized for faster calculation).
        sims = np.einsum("ij,ij->i", embs[:-1], embs[1:])

        dists = 1.0 - sims                  # Now, the larger the dist, more is diff in meaning. 
        mu = float(dists.mean())
        sigma = float(dists.std()) or 1e-6
        cutoff = mu + Z_SCORE * sigma       # This is the threshold, mu is mean and sigma the st dev.

        added = 0
        for local_i, d in enumerate(dists):
            if d >= cutoff:
                # Break between splittable_idx[local_i] and splittable_idx[local_i+1]
                boundary_pos = splittable_idx[local_i]
                if boundary_pos not in boundaries:
                    boundaries.add(boundary_pos)
                    added += 1
        return added
