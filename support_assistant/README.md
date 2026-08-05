# /support_assistant - GenAI Support Assistant

A small RAG (Retrieval-Augmented Generation) service that answers Zepto
policy questions grounded in Zepto's own documents, using LangGraph for
orchestration and FastAPI for the API layer. Runs fully offline in mock mode
(the graded baseline) -- no signup, no API key, no network call to any LLM
provider required.

## Files
- docs/doc_01.txt ... doc_08.txt - the 8 Zepto policy documents (the corpus)
- ingest.py - loads the docs, embeds them with all-MiniLM-L6-v2, stores them in ChromaDB
- schemas.py - Pydantic request/response models (AskRequest, AskResponse)
- prompts.py - the structured prompt template (used only in the optional MOCK_LLM=0 extension)
- llm_client.py - optional real-LLM call wrapper (not used in graded mock mode)
- graph.py - the LangGraph StateGraph: 3 nodes (classify_intent, retrieve_and_answer, direct_answer) + conditional routing
- main.py - FastAPI app exposing POST /ask
- Dockerfile - builds and runs the FastAPI app in a container
- requirements.txt - Python dependencies

## How to run locally

pip install -r requirements.txt
python ingest.py
uvicorn main:app --reload

The API is then available at http://127.0.0.1:8000/ask (interactive docs at http://127.0.0.1:8000/docs).

MOCK_LLM is left unset by default, which runs the fully deterministic, offline mock baseline -- this is what gets graded and requires no API key.

## How to run with Docker
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant

Then call http://127.0.0.1:7860/ask the same way.

## Example calls (MOCK_LLM left at default -- real output from a local run)

### Example 1: a policy question (triggers retrieval)

Request:
POST /ask
{"query": "What is your return policy?"}

Response:
```json
{
    "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop",
    "sources": ["doc_02", "doc_06", "doc_05"],
    "confidence": 1.0
}
```

### Example 2: a general/unrelated question (does NOT trigger retrieval)

Request:POST /ask
{"query": "What is the capital of France?"}
Response:
```json
{
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
}
```

## Architecture: how the RAG pipeline works

**Ingestion:** `ingest.py` reads all 8 files from `docs/`, treating each whole
document as a single chunk (they are short policy paragraphs, so no further
splitting is needed).

**Embedding:** Still in `ingest.py`, each document's text is passed through
`sentence-transformers`' `all-MiniLM-L6-v2` model to produce a vector
embedding, entirely locally with no API call. These embeddings, along with
each document's id (`doc_01` ... `doc_08`) and raw text, are stored in a
persistent ChromaDB collection named `zepto_policies` on disk at
`chroma_db/`.

**Retrieval:** When a query arrives, `graph.py`'s `retrieve_and_answer` node
embeds the incoming query with the same `all-MiniLM-L6-v2` model, then calls
`collection.query(...)` to fetch the top-3 most similar chunks from ChromaDB
via cosine similarity. This retrieval step always runs for real, in both
mock and real-LLM modes, since it needs no API key and no network call.
Verified with the "What is your return policy?" example above, which
correctly retrieved doc_02 (the real Returns & Refunds document) as the
top match.

**Generation:** Still inside `retrieve_and_answer`, only the final
answer-generation step branches on the `MOCK_LLM` toggle:
- Default (mock, graded baseline): a canned templated answer is built
  directly in code -- `f"Based on the retrieved context: {top_chunk_snippet}"`
  -- using the first ~200 characters of the single most similar chunk. No
  LLM call is made.
- Optional `MOCK_LLM=0` extension: the retrieved chunks are inserted into the
  structured prompt template in `prompts.py`, which is sent to a real LLM
  (via `llm_client.py`, using Groq's free tier) to generate a grounded
  answer instead.

**Routing:** `graph.py`'s `classify_intent` node runs first for every query.
In mock mode it uses a keyword heuristic (checking for words like
"delivery", "return", "refund", etc.) to decide `policy_question` vs
`general_question` -- no LLM call. A conditional edge then routes
`policy_question` queries to `retrieve_and_answer` and `general_question`
queries to `direct_answer` (which returns a fixed canned string in mock
mode, also with no retrieval and no LLM call). Verified with the "capital
of France" example above, which correctly routed to direct_answer with
empty sources.

**Output validation:** `main.py`'s FastAPI endpoint validates the final
`answer`/`sources`/`confidence` dict against the `AskResponse` Pydantic
schema before returning it. In mock mode this always succeeds immediately,
since the values are built deterministically in our own code -- there is no
LLM output that could fail validation. The optional `MOCK_LLM=0` path
includes retry-with-correction logic (up to 2 extra attempts) in case a real
LLM's raw output ever failed to validate.

## MOCK_LLM toggle summary

| Stage | Default (MOCK_LLM unset/1) | Optional (MOCK_LLM=0) |
|---|---|---|
| classify_intent | keyword heuristic, no LLM call | LLM call to classify |
| retrieve_and_answer | retrieval always real; answer is a canned template | retrieval always real; answer generated by real LLM using the structured prompt |
| direct_answer | fixed canned string, no LLM call | LLM call, no retrieval |