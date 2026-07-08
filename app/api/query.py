from fastapi import APIRouter, Request
from app.core.state import state
from app.core.retrieval.decisions import RetrievalDecision
from app.core.retrieval.grounding import compute_overlap
from app.core.guardrails.query_guard import sanitize_query
from app.core.guardrails.query_preprocessing import preprocess_query
from app.core.guardrails.context_guard import sanitize_context
from app.core.errors import PromptInjectionError, BadRequestError
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query")
async def query(payload: dict, request: Request):
    request_id = request.state.request_id
    query_text = payload.get("query")

    if not query_text:
        raise BadRequestError("query is required")

    query_text = preprocess_query(query_text)

    query_check = sanitize_query(query_text)

    if query_check["is_malicious"]:
        raise PromptInjectionError(
            "query flagged as potential prompt injection",
            details={"reasons": query_check.get("reasons", [])},
        )

    query_text = query_check["query"]

    if state.vector_index is None:
        state.retrieval_metrics.record(RetrievalDecision.REFUSE_EMPTY.value)
        logger.info(
            "retrieval_decision",
            extra={
                "request_id": request_id,
                "decision": RetrievalDecision.REFUSE_EMPTY.value,
                "metrics_snapshot": state.retrieval_metrics.snapshot()[0],
            },
        )

        return {
            "decision": RetrievalDecision.REFUSE_EMPTY.value,
            "reason": "no_documents_indexed",
        }


    query_embedding = state.embedder.embed_texts([query_text])
    top_k = payload.get("top_k", 5)
    top_k = max(1, min(top_k, state.retrieval_config.max_top_k))

    results = state.vector_index.search(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    decision, details = state.retrieval_evaluator.evaluate(results)
    state.retrieval_metrics.record(decision.value, details)

    logger.info(
        "retrieval_decision",
        extra={
            "request_id": request_id,
            "decision": decision.value,
            "metrics_snapshot": state.retrieval_metrics.snapshot()[0],
            **details,
        },
    )

    if decision in [RetrievalDecision.REFUSE_EMPTY, RetrievalDecision.REFUSE_WEAK]:
        return {
            "query": query_text,
            "decision": decision.value,
            "details": details,
            "retrieved_chunks": [],
        }

    raw_chunks = [
        state.chunk_store.get(cid).content
        for cid, _ in results
        if state.chunk_store.get(cid) is not None
    ]

    context_chunks = sanitize_context(raw_chunks)

    gen_result = state.llm.generate(
        query=query_text,
        context_chunks=context_chunks,
    )

    logger.info(
        "generation_executed",
        extra={
            "request_id": request_id,
            "latency_ms": gen_result.get("generation_latency_ms"),
            "prompt_chars": gen_result.get("prompt_chars"),
            "error": gen_result.get("error"),
        }
    )

    overlap_score = compute_overlap(
        gen_result.get("answer", ""),
        context_chunks
    )

    hallucination = overlap_score < 0.3

    state.query_eval_metrics.record(
        decision=decision.value,
        overlap_score=overlap_score,
        hallucination=hallucination,
    )

    logger.info(
        "query_evaluation",
        extra={
            "request_id": request_id,
            "decision": decision.value,
            "overlap_score": overlap_score,
            "hallucination": hallucination,
        }
    )

    if gen_result.get("response") in ["generation_failed", "generation_error"]:
        return {
            "query": query_text,
            "decision": "generation_failed",
            "error": gen_result.get("detailed_log_err"),
            "details": gen_result,
        }

    elif gen_result.get("response") == "refused":
        return {
            "query": query_text,
            "decision": "refused",
            "error": gen_result.get("detailed_log_err"),
            "details": gen_result,
        }
    confidence_map = {
        "answerable": "high",
        "answerable_low_confidence": "medium",
    }

    response = {
        "query": query_text,
        "decision": decision.value,
        "answer": gen_result.get("answer"),
        "confidence": confidence_map.get(decision.value, "none"),
        "latency_ms": gen_result.get("generation_latency_ms"),
        "sources": [
            {
                "chunk_id": cid,
                "score": score,
            }
            for cid, score in results
        ],
        "explanation": {
            "top_score": details.get("top_score"),
            "avg_score": details.get("avg_score"),
            "num_strong_chunks": details.get("num_strong_chunks"),
            "decision": decision.value,
            "overlap_score": overlap_score,
            "hallucination_risk": hallucination,
        },
        "guardrails": {
            "query_checked": True,
            "context_filtered": True,
        },
        "trace": [
            f"retrieval_decision: {decision.value}",
            f"top_score: {details.get('top_score')}",
            f"overlap_score: {overlap_score}",
            f"hallucination: {hallucination}",
        ],
    }

    return response