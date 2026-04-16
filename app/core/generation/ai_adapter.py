import os
import logging
import time

logger = logging.getLogger(__name__)


class SAPLLM:
    def __init__(self, settings):
        self._configure_env(settings)
        self._init_client()

    def _configure_env(self, settings):
        os.environ["AICORE_AUTH_URL"] = settings.aicore_auth_url
        os.environ["AICORE_CLIENT_ID"] = settings.aicore_client_id
        os.environ["AICORE_CLIENT_SECRET"] = settings.aicore_client_secret
        os.environ["AICORE_RESOURCE_GROUP"] = settings.aicore_resource_group
        os.environ["AICORE_BASE_URL"] = settings.aicore_base_url
        self.model_name = settings.model_name

    def _init_client(self):
        from gen_ai_hub.proxy.native.openai import chat
        self.chat = chat

    def preprocess_query(self, query: str) -> str:
        if not query:
            return ""

        q = query.strip().lower()               # 1. Trim + lowercase
        q = re.sub(r"\s+", " ", q)              # 2. Remove excessive whitespace
        q = re.sub(r"[^\w\s\?\.\,]", "", q)     # 3. Remove noisy characters (keep basic punctuation)
        if len(q.split()) <= 5:                 # 4. Optional: expand very short queries
            q = q + " detailed explanation"

        return q

    def generate(self, *, query: str, context_chunks: List[str],) -> Dict:
        start_time = time.time()

        context = "\n\n".join(context_chunks)

        system_prompt = (
            "You are a strict AI system that answers ONLY using the provided context.\n"
            "Rules:\n"
            "1. Use ONLY the given context.\n"
            "2. Do NOT use outside knowledge.\n"
            "3. If the answer is not present **in the context**, say: 'I cannot answer from the provided context.' **and nothing else**\n"
            "4. Keep answers concise and factual.\n"
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        prompt_chars = len(system_prompt) + len(user_prompt)

        try:
            response = self.chat.completions.create(
                model_name=self.model_name,
                messages=messages,
            )

            latency_ms = int((time.time() - start_time) * 1000)
            raw = response.to_dict()
            
            # content = response.to_dict()["choices"][0]["message"]["content"]
            content = (
                raw.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not content:
                return {
                    "response": "generation_failed",
                    "answer": "",
                    "generation_latency_ms": latency_ms,
                    "prompt_chars": prompt_chars,
                    "tokens_estimate": 0,
                    "detailed_log_out": "",
                    "detailed_log_err": "empty_response",
                }

            answer = content

            # Simple refusal detection
            refused = "cannot answer" in answer.lower()

            return {
                "response": "refused" if refused else "response_generated",
                "answer": answer,
                "generation_latency_ms": latency_ms,
                "prompt_chars": prompt_chars,
                "tokens_estimate": len(answer.split()),
                "detailed_log_out": answer,
                "detailed_log_err": "",
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)

            logger.error("sap_llm_error", extra={"error": str(e)})

            return {
                "response": "generation_error",
                "answer": "",
                "generation_latency_ms": latency_ms,
                "prompt_chars": prompt_chars,
                "tokens_estimate": 0,
                "detailed_log_out": "",
                "detailed_log_err": str(e),
            }