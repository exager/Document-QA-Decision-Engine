"""
Application settings.

Three settings classes, one for each concern:
  - AppSettings   → app-level defaults, size limits, log level.
                    Loaded from .env (no prefix).
  - LLMSettings   → generator backend + secrets + model params.
                    Loaded from .env (no prefix). Secrets live here.
  - RAGSettings   → retrieval / chunking tunables.
                    Loaded from .env.rag with prefix RAG_. No secrets.

Access via the lru_cached getters at the bottom of this file. Never
instantiate a Settings class directly at module import time — the getters
make sure there's exactly one instance per class per process.
"""
from __future__ import annotations
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


# ----- App level Config ----------------- 
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",           # tolerate unrelated keys in .env
        case_sensitive=False,
    )

    app_name: str = "applied-ai-system"
    log_level: str = "INFO"

    # Size limits — used by app/core/limits.py.
    max_json_body_bytes: int = 100_000
    max_file_upload_bytes: int = 25_000_000

    # Global request timeout applied by TimeoutMiddleware.
    request_timeout_seconds: int = 30


# ----- LLM Provider Config ----------------- 
class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # "sap" | "llamacpp"
    generator_backend: str = "llamacpp"

    # --- SAP AI Core (required only when generator_backend == "sap") -------
    aicore_auth_url: str = ""
    aicore_client_id: str = ""
    aicore_client_secret: str = ""
    aicore_resource_group: str = ""
    aicore_base_url: str = ""
    model_name: str = ""

    # --- llama.cpp (required only when generator_backend == "llamacpp") ----
    llama_model_path: str = "llama.cpp/build/models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    llama_ctx_size: int = 4096
    llama_max_tokens: int = 256
    llama_temperature: float = 0.2
    llama_timeout_sec: int = 30

    def validate_backend(self) -> None:
        if self.generator_backend == "sap":
            missing = [
                name for name in [
                    "aicore_auth_url", "aicore_client_id",
                    "aicore_client_secret", "aicore_resource_group",
                    "aicore_base_url", "model_name",
                ] if not getattr(self, name)
            ]
            if missing:
                raise RuntimeError(
                    f"generator_backend=sap requires: {missing}"
                )
        elif self.generator_backend == "llamacpp":
            if not Path(self.llama_model_path).exists():
                import logging
                logging.getLogger(__name__).warning(
                    "llama_model_path does not exist at import time: %s",
                    self.llama_model_path,
                )
        else:
            raise RuntimeError(
                f"unknown generator_backend={self.generator_backend!r}. "
                f"expected 'sap' or 'llamacpp'."
            )


# --- RAG related settings and vars--------------------------------------

class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.rag",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
        case_sensitive=False,
    )

    # Retrieval / evaluator thresholds
    min_top_score: float = 0.6
    min_chunks_above_threshold: int = 1
    max_top_k: int = 20

    # Chunker strategy — read by app/core/process.py::_resolve_chunker
    chunker_strategy: str = "character_v1"

    # Structural / semantic chunker knobs
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 150
    chunk_overlap_sentences: int = 1
    semantic_z_score: float = 1.0
    semantic_min_units: int = 4


# --- Accessors -----------------------------

@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    return AppSettings()


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    return LLMSettings()


@lru_cache(maxsize=1)
def get_rag_settings() -> RAGSettings:
    return RAGSettings()


# --- Backward-compatibility -------------------
# Anything currently importing `Settings` or `RetrievalConfig` from config
# keeps working. These names are deprecated but not removed 

Settings = AppSettings          # deprecated alias
RetrievalConfig = RAGSettings   # deprecated alias
