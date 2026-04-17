def compute_overlap(answer: str, context_chunks: list[str]) -> float:
    if not answer or not context_chunks:
        return 0.0

    # Normalize
    answer_tokens = set(answer.lower().split())
    context_text = " ".join(context_chunks)
    context_tokens = set(context_text.lower().split())

    if not answer_tokens:
        return 0.0

    overlap = answer_tokens.intersection(context_tokens)

    return len(overlap) / len(answer_tokens)