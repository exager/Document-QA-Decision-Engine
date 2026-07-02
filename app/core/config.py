from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "applied-ai-system"
    log_level: str = "INFO"

    # LLM config
    generator_backend: str
    aicore_auth_url: str
    aicore_client_id: str
    aicore_client_secret: str
    aicore_resource_group: str
    aicore_base_url: str
    model_name: str

    class Config:
        env_file = ".env"

class RetrievalConfig(BaseSettings):
    min_top_score: float = 0.6
    min_chunks_above_threshold: int = 1
    max_top_k: int = 20
    class Config:
        env_prefix = "RAG_"

