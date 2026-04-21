# Applied AI System — Document QA Engine

## Overview

This project is a **decision-driven Retrieval-Augmented Generation (RAG) system** designed to answer user queries over ingested documents while maintaining **control, explainability, and reliability**.

Unlike typical RAG pipelines that directly retrieve and generate answers, this system introduces:
- a **retrieval evaluation layer**
- a **decision engine**
- **grounding validation**
- **system-level metrics**
- **guardrails against prompt injection**

The goal is not just to generate answers, but to **decide when answering is safe and justified**.

---

## Key Features

### 1. Retrieval + Heuristic Evaluation

- Vector-based retrieval using embeddings  
- Multi-signal evaluation:
  - top similarity score  
  - average score  
  - score distribution  
  - number of strong matches  

- Converts retrieval quality into decisions:
  - `answerable`
  - `answerable_low_confidence`
  - `refuse_empty`
  - `refuse_weak`

---

### 2. Decision-Driven Answering

- The system **does not always call the LLM**  
- LLM is invoked only when retrieval confidence allows it  
- Prevents unnecessary hallucinations and token costs

---

### 3. Controlled Generation (LLM Layer)

- Pluggable LLM interface  
- SAP AI Hub integration  
- Strict system prompt:
  - answer only from context  
  - refuse if context insufficient  
  - ignore malicious instructions  

---

### 4. Grounding Validation (Hallucination Check)

- Token overlap between answer and retrieved context  
- Flags hallucination risk  
- Adds validation layer after generation  

---

### 5. Guardrails (Security)

- Query sanitization (detect prompt injection attempts)  
- Context filtering (removes malicious instructions from documents)  
- System prompt enforcement  

---

### 6. Explainable Responses

Each response includes:
- answer  
- confidence level  
- sources (chunks + scores)  
- explanation (why the system answered/refused)  
- trace (internal decision flow)  

---

### 7. Metrics & Evaluation

Tracks system performance:

- retrieval decisions  
- answer vs refusal rate  
- hallucination rate  
- average grounding score  
- latency and error tracking  

---

## System Architecture
Query
→ Query Sanitization
→ Embedding + Retrieval
→ Heuristic Evaluation
→ Decision Engine
→ Context Sanitization
→ LLM Generation
→ Grounding Validation
→ Metrics Logging
→ Structured Response

---
## Installation

### 1. Clone the repository
```
git clone <repository URL>
cd <repositony name>
```

### 2. Create virtual environment
```python -m venv venv
source venv/bin/activate # mac/linux
.\venv\Scripts\activate # windows
```

### 3. Install dependencies
``` pip install -r requirements.txt```

### Environment Variables
Create a `.env` file:
```AICORE_AUTH_URL=...
AICORE_CLIENT_ID=...
AICORE_CLIENT_SECRET=...
AICORE_RESOURCE_GROUP=default
AICORE_BASE_URL=...
MODEL_NAME=...```
```

> [!IMPORTANT]
> If needed, the model thresholds for llama.cpp model and even API based model can be tweaked in `app\core\config.py` file

## Running the Application
```uvicorn app.main:app --reload```

Then Open: http://127.0.0.1:8000/docs
---
## API Endpoints

### 1. Ingest Text
POST /ingest
```json
Payload: {
  "content": "Your document text",
  "metadata": {},
  "source": "local"
}
```
---

### 2. Query System
POST /query
```json
Payload: {
  "query": "Your question here",
  "top_k": 6,  //Optional field for Top matching chunks you want to check against
}
```

## Example Response
```json
{
  "query": "...",
  "answer": "...",
  "confidence": "high",
  "sources": [...],
  "explanation": {
    "top_score": 0.82,
    "decision": "answerable",
    "overlap_score": 0.65,
    "hallucination_risk": "<True/False>",
  },
  "guardrails": {
    "query_checked": "True",
    "context_filtered": "True",
  },
  "trace": [
    "retrieval_decision: answerable",
    "top_score: 0.82",
    "overlap_score: 0.65",
    "hallucination_risk: <True/False>",
  ]
}
```

---

## Design Principles

- Do not trust LLM blindly  
- Do not trust retrieval blindly  
- Always validate before returning  
- Prefer refusal over hallucination  
- Keep system explainable and observable  
- Separate concerns across modules  

---

## Trade-offs

- Heuristic evaluation used instead of reranking (faster, deterministic)  
- Token overlap used for grounding (simple, interpretable)  
- No heavy frameworks (LangChain, etc.) to maintain control  

---

## Failure Modes

- Weak retrieval → system refuses  
- Partial context → low confidence answer  
- Overlap heuristic may miss semantic correctness  
- Prompt injection attempts filtered but not fully eliminated  

---

## Future Work

- RAGAS-based evaluation framework for deeper quality assessment  
- Improved semantic grounding validation  
- Persistent vector storage  
- Advanced query rewriting  

---

## Summary

This project demonstrates how to build a **controlled, explainable AI system** instead of a naive RAG pipeline.  
It focuses on **decision-making, validation, and reliability**, which are critical for real-world AI applications.
