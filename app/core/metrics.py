import time
from collections import defaultdict, Counter
from threading import Lock

class MetricsStore:
    def __init__(self):
        self.lock = Lock()
        self.request_count = 0
        self.error_count = 0
        self.latencies = []

    def record(self, latency_ms: float, is_error: bool):
        with self.lock:
            self.request_count += 1
            if is_error:
                self.error_count += 1
            self.latencies.append(latency_ms)

    def snapshot(self):
        with self.lock:
            latencies = sorted(self.latencies)
            count = len(latencies)

            def percentile(p):
                if count == 0:
                    return 0.0
                idx = int(count * p)
                idx = min(idx, count - 1)
                return latencies[idx]

            return {
                "requests_total": self.request_count,
                "errors_total": self.error_count,
                "p50_latency_ms": percentile(0.50),
                "p95_latency_ms": percentile(0.95),
            }

class RetrievalMetrics:
    def __init__(self):
        self.decisions = Counter()
        self.details = defaultdict(list)

    def record(self, decision: str, details: dict | None = None):
        if details is None:
            details = {}

        self.decisions[decision] += 1
        self.details[decision].append(details)

    def snapshot(self):
        return dict(self.decisions), dict(self.details)

class QueryEvaluationMetrics:
    def __init__(self):
        self.lock = Lock()
        self.total_queries = 0
        self.answered = 0
        self.refused = 0
        self.hallucinations = 0
        self.overlap_scores = []

    def record(
        self,
        *,
        decision: str,
        overlap_score: float,
        hallucination: bool,
    ):
        with self.lock:
            self.total_queries += 1

            if decision in ["answerable", "answerable_low_confidence"]:
                self.answered += 1
            else:
                self.refused += 1

            if hallucination:
                self.hallucinations += 1

            self.overlap_scores.append(overlap_score)

    def snapshot(self):
        with self.lock:
            if self.total_queries == 0:
                return {}

            avg_overlap = sum(self.overlap_scores) / len(self.overlap_scores)

            return {
                "total_queries": self.total_queries,
                "answered": self.answered,
                "refused": self.refused,
                "answer_rate": self.answered / self.total_queries,
                "refusal_rate": self.refused / self.total_queries,
                "hallucination_rate": self.hallucinations / self.total_queries,
                "avg_overlap_score": avg_overlap,
            }