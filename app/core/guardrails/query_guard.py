def sanitize_query(query: str) -> dict:
    original_query = query
    query = query.strip()

    lower_q = query.lower()

    blocked_phrases = [
        "ignore previous instructions",
        "ignore earlier prompt",
        "disregard above",
        "system prompt",
        "system:",
        "act as",
        "jailbreak",
        "you are now",
    ]

    flagged = False

    for phrase in blocked_phrases:
        if phrase in lower_q:
            flagged = True
            break

    # Hard reject only if clearly malicious
    if flagged:
        return {
            "query": query,
            "is_malicious": True,
            "sanitized": False,
        }

    return {
        "query": query,
        "is_malicious": False,
        "sanitized": True,
    }