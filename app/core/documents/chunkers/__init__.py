"""
Chunking strategy registry.

Usage:
    from app.core.documents.chunkers import get_chunker

    chunker = get_chunker("semantic_v1", embed_fn=state.embedder.embed_texts)
    parts = chunker.chunk(document)
"""
from __future__ import annotations

import logging
from typing import Type
from app.core.documents.chunkers.base import Chunker

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Type[Chunker]] = {}


def register(name: str):
    """Decorator: register a Chunker subclass under `name`."""
    def decorator(cls: Type[Chunker]) -> Type[Chunker]:
        if name in _REGISTRY:
            raise ValueError(f"chunker already registered: {name!r}")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_chunker(name: str, **kwargs) -> Chunker:
    """
    Instantiate the chunker registered under `name`.

    Extra kwargs are forwarded to the strategy's __init__. Strategies that
    don't need a particular kwarg (e.g. `embed_fn`) should still accept and
    ignore it.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown chunker strategy: {name!r}. available: {available()}"
        )
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)



from app.core.documents.chunkers import character  # noqa: E402, F401
from app.core.documents.chunkers import structural  # noqa: E402, F401
from app.core.documents.chunkers import semantic  # noqa: E402, F401
