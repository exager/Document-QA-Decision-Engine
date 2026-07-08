# Document QA Decision Engine

> A RAG system that answers when it can prove the answer, and refuses when it can't.

**Resol** is a Retrieval-Augmented Generation pipeline built around one guiding principle: **refuse over hallucinate**. Where most RAG demos retrieve top-k chunks and stuff them into an LLM prompt, Resol treats retrieval as a *signal*, not a verdict - it evaluates retrieval quality, decides whether the evidence justifies an answer, generates only when it does, and validates every answer against its sources before returning it.

It's built as a FastAPI service with pluggable chunkers, pluggable LLM backends, structured error handling, and observability baked in. The whole thing runs on a laptop but is structured so that swapping a component (chunker strategy, embedding model, LLM backend, vector store) is a config change, not a rewrite.

---

## Why "not just another RAG"

Most tutorial RAG systems fail on the two things that matter in production: **knowing when to refuse**, and **explaining themselves**. Resol is designed around both.

| Concern | Typical RAG | Resol |
|---|---|---|
| Retrieval quality | "Get top-k, pass to LLM" | Multi-signal retrieval evaluator that scores confidence and can refuse |
| Chunking | Fixed-size character splitter | Three pluggable strategies (`character_v1`, `structural_v1`, `semantic_v1`) selected by config |
| Grounding | None, or LLM self-check | Token-overlap grounding score + hallucination flag on every answer |
| Prompt injection | Trust the LLM to ignore it | Query sanitization + context filtering + strict system prompt |
| Error responses | `{"detail": "..."}` | Uniform envelope with stable error codes, `request_id`, and structured `details` |s
| Configuration | Hardcoded constants | Three-layer settings (app / LLM / RAG) driven by `.env` and `.env.rag` |
| Refusal | Silent low-quality answer | Explicit `decision` field: `answerable`, `answerable_low_confidence`, `refuse_weak`, `refuse_empty` |
| Observability | `print()` | Structured JSON logs with request-ID correlation on every line |

---

## Architecture

```
┌─────────────────┐     ┌───────────────┐     ┌───────────────┐
│  /ingest        │────▶│ Loader        │────▶│ Chunker       │
│  /ingest/file   │     │ (PDF / DOCX)  │     │ (strategy-    │
│  (multipart)    │     │ structure-    │     │  driven)      │
└─────────────────┘     │ aware,        │     └──────┬────────┘
                        │ elements list │            │
                        └───────────────┘            ▼
                                               ┌──────────────┐
                                               │ Embedder     │
                                               │ (MiniLM,     │
                                               │  L2-normed)  │
                                               └──────┬───────┘
                                                      ▼
                                               ┌──────────────┐
                                               │ FAISS Index  │
                                               │ (IndexFlatIP)│
                                               └──────┬───────┘
                                                      │
┌─────────────────┐   ┌──────────────┐   ┌───────────▼────────┐
│    /query       │──▶│ Sanitize +   │──▶│ Retrieve top-k     │
│                 │   │ preprocess   │   └───────────┬────────┘
└─────────────────┘   └──────────────┘               ▼
                                           ┌──────────────────┐
                                           │ Retrieval        │
                                           │ Evaluator        │◀── refuse if
                                           │ (multi-signal    │    confidence
                                           │  heuristic)      │    too low
                                           └────────┬─────────┘
                                                    │  (answerable)
                                                    ▼
                                           ┌──────────────────┐
                                           │ Context Guard    │
                                           │ (injection strip)│
                                           └────────┬─────────┘
                                                    ▼
                                           ┌──────────────────┐
                                           │ LLM              │◀── SAP AI Core
                                           │ (pluggable)      │◀── or llama.cpp
                                           └────────┬─────────┘
                                                    ▼
                                           ┌──────────────────┐
                                           │ Grounding Check  │
                                           │ (overlap score)  │
                                           └────────┬─────────┘
                                                    ▼
                                           ┌──────────────────┐
                                           │ Structured       │
                                           │ Response         │
                                           │ (answer + trace) │
                                           └──────────────────┘
```

Every stage records structured logs tagged with the same `request_id`, so any query can be traced end-to-end in one `grep`.

---

## What makes it interesting

### 1. Retrieval evaluator - the refuse-before-hallucinate layer

Retrieval doesn't return an answer. It returns evidence, and the **evaluator** decides whether that evidence is strong enough to generate on. Signals used:

- **Top similarity score** - how good is the single best chunk?
- **Number of strong chunks** - is the signal isolated or corroborated?
- **Score distribution** - is there a sharp drop between chunk 1 and chunk 5?
- **Average score** - do we have a consistent match or one lucky hit?

Signals combine into a confidence score, which maps to one of four decisions:

| Decision | Meaning |
|---|---|
| `answerable` | Strong retrieval - call the LLM, expect a high-quality answer |
| `answerable_low_confidence` | Some evidence - answer but flag reduced confidence |
| `refuse_weak` | Retrieval too noisy - refuse rather than hallucinate |
| `refuse_empty` | Nothing indexed or nothing matched - refuse |

**Refusal is a first-class outcome**, not an error. Every refusal ships with the evidence trail that led to it.

### 2. Pluggable chunkers (Strategy pattern + registry)

Chunking is a Strategy, selectable via `RAG_CHUNKER_STRATEGY` in `.env.rag`. Three strategies ship:

- **`character_v1`** - the classic 700-char + 120-char-overlap approach. Kept as the baseline for A/B evaluation.
- **`structural_v1`** - respects document structure (headings never orphaned, tables never split, sentence-aware packing into token budgets). Zero embedding cost.
- **`semantic_v1`** - hybrid: structural constraints + z-score cosine-distance boundaries between adjacent sentences. Uses embeddings you were going to compute anyway.

Every chunk carries `metadata["chunker"]` for provenance, so retrieval outcomes can be attributed to the strategy that produced them.

**How it's wired:** each chunker is a subclass of `Chunker(ABC)` registered under a name; `chunkers.get_chunker("semantic_v1", ...)` returns an instance. Adding a strategy is a single file plus a `@register("my_v1")` decorator.

### 3. Structure-aware document extractors

PDF and DOCX loaders don't return a text blob. They return an **`ExtractedDocument`** - an ordered list of typed `DocumentElement`s (titles, headings, paragraphs, list items, tables, code, captions) with page numbers, heading paths, and style metadata.

- **PDF extraction** uses `pymupdf` with layout awareness: font-size-histogram heading detection, `page.find_tables()` for real table extraction (rendered as markdown, not scrambled cell soup), and bounding-box deduplication so table text isn't double-counted.
- **DOCX extraction** walks the docx XML body in true reading order (paragraphs *and* tables interleaved), maps Word paragraph styles to element types, and detects real list items via `<w:numPr>` rather than the fragile "List Paragraph" style name.
- Both maintain a **depth-preserving heading stack**: an `H3` appearing under an `H1` (with no `H2`) produces `["H1", "", "H3"]` - length equals level, so downstream code always knows structural depth.

This structure is what makes structural and semantic chunking possible.

### 4. Guardrails you can defend

- **Query sanitization** - pattern-matches known prompt-injection templates; malicious queries are rejected with a `400 prompt_injection_detected`, not a silent 200.
- **Context filtering** - retrieved chunks are scrubbed for embedded "ignore previous instructions" style content before being handed to the LLM.
- **Strict system prompt** - the LLM is instructed to treat context as *data*, not instructions; to refuse if the answer isn't in the context; to never draw on outside knowledge.

None of these are silver bullets - they're layered defenses, and each one is a known technique documented in the code.

### 5. Grounding validation

After generation, the answer is scored against the retrieved context via token overlap. If the overlap falls below a threshold, the response is flagged with `hallucination_risk: true` - the system tells you when it doesn't trust its own answer. This is a heuristic today, deliberately kept simple and interpretable; a cross-encoder NLI upgrade is on the roadmap.

### 6. Uniform error envelope

Every 4xx/5xx response has the same shape:

```json
{
  "error": "low_quality_document",
  "message": "extracted document is too short or low quality",
  "request_id": "5f3b1c2d-9e7a-4b8c-a1d2-1234567890ab",
  "details": {
    "extracted_length": 42,
    "filename": "note.pdf"
  }
}
```

Errors are typed (`AppError` subclasses with stable string codes and HTTP status codes), pydantic validation failures share the same envelope, unhandled exceptions get logged with the `request_id` and returned as an opaque 500. Clients only ever parse one shape.

### 7. Config as a first-class citizen

Three settings classes, two env files, one accessor per class:

| Class | File | Contents | Prefix |
|---|---|---|---|
| `AppSettings` | `.env` | log level, size limits, timeouts | none |
| `LLMSettings` | `.env` | backend choice, secrets, model params | none |
| `RAGSettings` | `.env.rag` | chunker strategy, thresholds, chunker knobs | `RAG_` |

Secrets and tunables are separated. `.env.example` documents the surface. All access goes through `@lru_cache`-backed getters so there's exactly one instance per process.

### 8. Structured, correlatable observability

- JSON logs with `request_id` propagated through every stage.
- Per-request middleware for `request_id` injection, timing, and a global timeout.
- Metrics counters for retrieval decisions, hallucination flags, and per-stage latency.
- `/health` and `/metrics` endpoints for scraping.

---

## Quickstart

### Prerequisites

- Python 3.11+
- (Optional) A local `llama.cpp` build if you want to run the LLM locally
- (Optional) SAP AI Core credentials if you want to use hosted models

### 1. Clone and set up the environment

```bash
git clone https://github.com/exager/resol.git
cd resol

python -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .\.venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Edit `.env` - at minimum set `GENERATOR_BACKEND` and the credentials for whichever backend you're using:

```dotenv
# --- App ------------------------------------------------------------
APP_NAME=applied-ai-system
LOG_LEVEL=INFO
MAX_JSON_BODY_BYTES=100000
MAX_FILE_UPLOAD_BYTES=25000000
REQUEST_TIMEOUT_SECONDS=30

# --- LLM backend selection ------------------------------------------
GENERATOR_BACKEND=llamacpp            # or "sap"

# --- SAP AI Core (needed only if GENERATOR_BACKEND=sap) -------------
AICORE_AUTH_URL=
AICORE_CLIENT_ID=
AICORE_CLIENT_SECRET=
AICORE_RESOURCE_GROUP=
AICORE_BASE_URL=
MODEL_NAME=

# --- llama.cpp (needed only if GENERATOR_BACKEND=llamacpp) ----------
LLAMA_MODEL_PATH=llama.cpp/build/models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf
LLAMA_CTX_SIZE=4096
LLAMA_MAX_TOKENS=256
LLAMA_TEMPERATURE=0.2
LLAMA_TIMEOUT_SEC=30
```

Create `.env.rag` for retrieval and chunking tunables:

```dotenv
RAG_CHUNKER_STRATEGY=character_v1     # try: structural_v1, semantic_v1
RAG_MIN_TOP_SCORE=0.6
RAG_MIN_CHUNKS_ABOVE_THRESHOLD=1
RAG_MAX_TOP_K=20

RAG_CHUNK_TARGET_TOKENS=350
RAG_CHUNK_MAX_TOKENS=512
RAG_CHUNK_MIN_TOKENS=150
RAG_CHUNK_OVERLAP_SENTENCES=1
RAG_SEMANTIC_Z_SCORE=1.0
RAG_SEMANTIC_MIN_UNITS=4
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

Open the interactive API docs at `http://127.0.0.1:8000/docs`.

### 4. Try it end-to-end

**Ingest text directly:**

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "local",
    "content": "The mitochondrion is the powerhouse of the cell. It generates ATP through oxidative phosphorylation.",
    "metadata": {"source_name": "biology_intro"}
  }'
```

**Ingest a PDF or DOCX:**

```bash
curl -X POST http://127.0.0.1:8000/ingest/file \
  -F "file=@/path/to/document.pdf"
```

**Query:**

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the mitochondrion responsible for?", "top_k": 5}'
```

---

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness / readiness |
| `POST` | `/ingest` | Ingest inline text or trigger internet search |
| `POST` | `/ingest/file` | Ingest a PDF or DOCX upload |
| `POST` | `/query` | Ask a question over ingested documents |
| `GET` | `/metrics` | Retrieval and query metrics snapshot |

Full request/response schemas are in `/docs` (Swagger UI) when the server is running.

---

## Response contract

### Successful answer

```json
{
  "query": "What is the mitochondrion responsible for?",
  "decision": "answerable",
  "answer": "The mitochondrion generates ATP through oxidative phosphorylation.",
  "confidence": "high",
  "latency_ms": 342,
  "sources": [
    {"chunk_id": "a1b2c3...", "score": 0.87},
    {"chunk_id": "d4e5f6...", "score": 0.71}
  ],
  "explanation": {
    "top_score": 0.87,
    "avg_score": 0.79,
    "num_strong_chunks": 2,
    "decision": "answerable",
    "overlap_score": 0.65,
    "hallucination_risk": false
  },
  "guardrails": {
    "query_checked": true,
    "context_filtered": true
  },
  "trace": [
    "retrieval_decision: answerable",
    "top_score: 0.87",
    "overlap_score: 0.65",
    "hallucination: false"
  ]
}
```

### Refusal (weak retrieval)

```json
{
  "query": "...",
  "decision": "refuse_weak",
  "details": {
    "top_score": 0.34,
    "num_strong_chunks": 0,
    "score_deviation": 0.05,
    "confidence_score": 0
  },
  "retrieved_chunks": []
}
```

### Error (uniform envelope)

```json
{
  "error": "prompt_injection_detected",
  "message": "query flagged as potential prompt injection",
  "request_id": "5f3b1c2d-9e7a-4b8c-a1d2-1234567890ab",
  "details": {"reasons": ["ignore_previous_instructions"]}
}
```

---

## Design principles

- **Don't trust the LLM blindly** - validate every answer against its sources.
- **Don't trust retrieval blindly** - score its quality before acting on it.
- **Prefer refusal over hallucination** - refusing correctly is a feature.
- **Every decision has evidence** - the trace is part of the API, not just internal logging.
- **Every stage is swappable** - chunker, embedder, LLM, vector store are all interfaces.
- **Fail loud at startup, gracefully at runtime** - config validation catches misconfiguration at boot; runtime failures return typed errors, not stack traces.

---

## What's honestly not there yet

Being explicit about the gaps because this is a portfolio project, not a shipped SaaS:

- **Grounding is token-overlap, not NLI.** Cheap and interpretable, but misses paraphrase. Upgrade path: cross-encoder NLI, planned.
- **Retrieval evaluator thresholds are hand-tuned, not calibrated.** The eval harness (next milestone) will produce ROC curves and set thresholds from data.
- **No reranker between retrieval and generation.** Adding a cross-encoder reranker is straightforward and will substantially improve top-k quality.
- **In-memory only.** The FAISS index and chunk store don't persist across restarts. Persistence + a real vector DB (Qdrant / pgvector) are on the roadmap.
- **No auth or rate limiting.** Localhost / demo only; production hardening is planned.
- **Guardrails are pattern-based**, not model-based. Real adversarial prompt-injection defense is a research problem; the current defenses filter the obvious cases and log the attempts.

---

## Roadmap

Grouped by tier of investment:

**Correctness & evaluation**
- [ ] Eval harness with a hand-labeled dataset, retrieval + refusal + grounding metrics, CI gate
- [ ] Calibrate retrieval evaluator thresholds from data
- [ ] NLI-based grounding (`cross-encoder/nli-deberta-v3-small`)
- [ ] Cross-encoder reranker between retrieval and generation
- [ ] Hybrid retrieval: dense + BM25 with Reciprocal Rank Fusion

**Systems**
- [ ] Persistence: FAISS-on-disk + SQLite chunk store, or Qdrant
- [ ] Concurrency: `asyncio.to_thread` for blocking calls, RW lock on the index
- [ ] Streaming responses via Server-Sent Events
- [ ] Auth (API key) + per-key rate limiting
- [ ] Multi-stage Dockerfile + docker-compose (app + Qdrant + Prometheus)
- [ ] OpenTelemetry tracing spans for each stage

**Frontier (pick one)**
- [ ] Per-citation answer synthesis with per-claim validation
- [ ] Multi-hop query decomposition
- [ ] Adversarial test suite (dedicated prompt-injection benchmark)
- [ ] Semantic answer cache with cosine-similarity lookup

---

## Repository layout

```
app/
├── main.py                          # FastAPI app + middleware + exception handlers
├── api/
│   ├── health.py
│   ├── ingest.py                    # /ingest and /ingest/file
│   ├── query.py                     # /query
│   └── metrics.py
├── core/
│   ├── config.py                    # AppSettings, LLMSettings, RAGSettings
│   ├── errors.py                    # AppError hierarchy
│   ├── limits.py                    # payload size guards
│   ├── logging.py                   # structured JSON logging
│   ├── observability.py             # request timing middleware
│   ├── request_id.py                # request ID middleware + ContextVar
│   ├── timeouts.py                  # global request timeout
│   ├── state.py                     # AppState singleton
│   ├── process.py                   # ingest orchestration
│   ├── retry.py
│   ├── metrics.py
│   ├── documents/
│   │   ├── models.py                # Document, Chunk
│   │   ├── elements.py              # DocumentElement, ExtractedDocument
│   │   └── chunkers/
│   │       ├── base.py              # Chunker ABC
│   │       ├── __init__.py          # registry
│   │       ├── character.py         # character_v1
│   │       ├── structural.py        # structural_v1
│   │       ├── semantic.py          # semantic_v1
│   │       └── chunker_v0.py        # baseline (wrapped by character_v1)
│   ├── loaders/
│   │   ├── base.py
│   │   ├── pdf_loader.py            # pymupdf, layout-aware
│   │   ├── docx_loader.py           # python-docx, structure-preserving
│   │   └── quality.py
│   ├── embeddings/
│   │   └── embedder.py              # MiniLM, L2-normalized
│   ├── retrieval/
│   │   ├── decisions.py             # RetrievalDecision enum
│   │   ├── evaluator.py             # multi-signal heuristic scorer
│   │   └── grounding.py             # answer/context overlap
│   ├── generation/
│   │   ├── base.py                  # Generator protocol
│   │   ├── ai_adapter.py            # SAP AI Core
│   │   └── llama_generator.py       # local llama.cpp
│   ├── guardrails/
│   │   ├── query_guard.py
│   │   ├── query_preprocessing.py
│   │   └── context_guard.py
│   ├── search/
│   │   ├── base.py
│   │   └── dummy_internet.py
│   └── storage/
│       ├── chunk_store.py           # in-memory chunk store
│       └── vector_index.py          # FAISS IndexFlatIP
└── schemas/
    ├── ingest.py
    └── query.py
```

---

## Contributing

This is a personal project, but issues and PRs are welcome. The rough contribution model:

1. Open an issue describing the change.
2. Fork, branch, and open a PR against `main`.
3. Every PR runs (or will run - see Roadmap) the eval harness. A PR that regresses retrieval recall by >2 points, or increases false-answer rate, won't merge.

---

## License

MIT - see `LICENSE`.