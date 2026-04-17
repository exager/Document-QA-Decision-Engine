import re

def preprocess_query(query: str) -> str:
        if not query:
            return ""

        q = query.strip().lower()               # 1. Trim + lowercase
        q = re.sub(r"\s+", " ", q)              # 2. Remove excessive whitespace
        q = re.sub(r"[^\w\s\?\.\,]", "", q)     # 3. Remove noisy characters (keep basic punctuation)
        if len(q.split()) <= 5:                 # 4. Optional: expand very short queries
            q = q + " detailed explanation"

        return q