import subprocess
import time
from typing import List


class LlamaCppGenerator:
    def __init__(
        self,
        *,
        model_path: str,
        ctx_size: int = 4096,
        max_tokens: int = 256,
        temperature: float = 0.2,
        timeout_sec: int = 30,
    ):
        self.model_path = model_path
        self.ctx_size = ctx_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    def generate(
        self,
        *,
        query: str,
        context_chunks: List[str],
    ) -> dict:
        
        context = "\n\n".join(context_chunks)

        prompt = (
            "You are a system that answers questions using ONLY the provided context.\n"
            "If the context does not contain the answer, say you cannot answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "###RESPONSE###\nAnswer:"
        )

        prompt_chars = len(prompt)

        cmd = [
            "./llama.cpp/build/bin/llama-cli",
            "-m", str(self.model_path),
            "--ctx-size", str(self.ctx_size),
            "--n-predict", str(self.max_tokens),
            "--temp", str(self.temperature),
            "-p", prompt,
            "-e",
            "--simple-io",
            "--single-turn",
        ]

        start = time.time()
        cmd = [str(arg) for arg in cmd]

        try:
            result = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            stdout, stderr = result.communicate()
        except subprocess.TimeoutExpired:
            return {
                "error": "generation_timeout",
                "latency_ms": int((time.time() - start) * 1000),
                "prompt_chars": prompt_chars,
            }

        latency_ms = int((time.time() - start) * 1000)

        response = "generation_failed" if result.returncode != 0 or result.stdout.strip() == '' else "response_generated"

        return {
            "response": response,
            "answer": stdout.strip().split('\n\n')[-3],
            "generation_latency_ms": latency_ms,
            "prompt_chars": prompt_chars,
            "detailed_log_out": stdout.strip(),
            "detailed_log_err":stderr.strip()
        }
