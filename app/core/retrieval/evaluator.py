from typing import List, Tuple
from app.core.retrieval.decisions import RetrievalDecision


class RetrievalEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(
        self,
        results: List[Tuple[str, float]],
    ) -> tuple[RetrievalDecision, dict]:
        """
        results: List of (chunk_id, similarity_score)
        """

        if not results:
            return (
                RetrievalDecision.REFUSE_EMPTY,
                {"reason": "no_chunks_retrieved"},
            )

        scores = [score for _, score in results]
        top_score = max(scores)
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)

        strong_chunks = [s for s in scores if s >= self.config.min_top_score]
        num_strong = len(strong_chunks)
        score_deviation = top_score - avg_score

        confidence_score = 0

        # Strong top match
        if top_score >= self.config.min_top_score:
            confidence_score += 2

        # Multiple strong chunks
        if num_strong >= 2:
            confidence_score += 2

        # Stable distribution: Better picking of chunks
        if avg_score >= self.config.min_top_score * 0.8:
            confidence_score += 1

        # Penalty on sharp drop between different matching chunks (unstable context)
        if score_deviation > 0.4:
            confidence_score -= 1

        # Decision Mapping
        if confidence_score >= 3:
            decision = RetrievalDecision.ANSWERABLE

        elif confidence_score == 2:
            decision = RetrievalDecision.ANSWERABLE_LOW_CONFIDENCE

        else:
            decision = RetrievalDecision.REFUSE_WEAK

        details = {
            "top_score": top_score,
            "avg_score": avg_score,
            "min_score": min_score,
            "num_chunks": len(scores),
            "num_strong_chunks": num_strong,
            "score_deviation": score_deviation,
            "confidence_score": confidence_score,
        }

        return decision, details