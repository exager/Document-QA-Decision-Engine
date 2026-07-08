from app.core.storage.chunk_store import InMemoryChunkStore
from app.core.storage.vector_index import InMemoryVectorIndex
from app.core.embeddings.embedder import Embedder, InMemoryEmbeddingCache
from app.core.retrieval.evaluator import RetrievalEvaluator
from app.core.config import get_app_settings, get_llm_settings, get_rag_settings
from app.core.metrics import MetricsStore, RetrievalMetrics, QueryEvaluationMetrics
from app.core.generation.llama_generator import LlamaCppGenerator
from app.core.generation.ai_adapter import SAPLLM

class AppState:
    def __init__(self):
        self.settings = get_app_settings()
        self.chunk_store = InMemoryChunkStore()
        self.embedder = Embedder()
        self.embedding_cache = InMemoryEmbeddingCache()
        self.vector_index = None
        self.retrieval_config = get_rag_settings()
        self.retrieval_evaluator = RetrievalEvaluator(config=self.retrieval_config)
        self.metrics_store = MetricsStore()
        self.retrieval_metrics = RetrievalMetrics()
        self.query_eval_metrics = QueryEvaluationMetrics()
        self.llm_settings = get_llm_settings()

        # Fails here on missing LLM config
        self.llm_settings.validate_backend()

        if self.settings.generator_backend == "sap":
            self.llm = SAPLLM(self.llm_settings)
        elif self.settings.generator_backend == "llamacpp":
            self.llm = LlamaCppGenerator(
                model_path=self.llm_settings.llama_model_path,
                ctx_size=self.llm_settings.llama_ctx_size,
                max_tokens=self.llm_settings.llama_max_tokens,
                temperature=self.llm_settings.llama_temperature,
                timeout_sec=self.llm_settings.llama_timeout_sec,
            )
        else:
            raise RuntimeError(
                f"unknown generator_backend={ self.settings.generator_backend!r}. "
                f"expected values: 'sap', 'llamacpp'."
            )

state = AppState()
