def sanitize_context(chunks: list[str]) -> list[str]:
    cleaned_chunks = []

    blocked_patterns = [
        "ignore previous instructions",
        "ignore earlier prompt",
        "disregard above",
        "system prompt",
        "system:",
        "act as",
        "jailbreak",
        "you are now",
    ]

    for chunk in chunks:
        chunk_lower = chunk.lower()

        if any(pattern in chunk_lower for pattern in blocked_patterns):
            continue

        cleaned_chunks.append(chunk)

    return cleaned_chunks