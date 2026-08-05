"""
graph.py
The LangGraph StateGraph: a TypedDict state, 3 nodes, and a conditional edge
that routes each query to either retrieval-grounded answering or a direct
canned answer.

MOCK_LLM toggle:
- Unset, or MOCK_LLM=1 (default): every generation step uses deterministic,
  rule-based / templated logic. No LLM call is ever made. This is the
  required, graded baseline.
- MOCK_LLM=0 (optional, ungraded extension): generation steps call a real
  LLM via llm_client.py instead.

Retrieval itself (embedding the query + querying ChromaDB) always runs for
real in both modes -- only the final answer-generation step branches on
MOCK_LLM.
"""

import os
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "zepto_policies"

MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]

# --- Lazy-loaded singletons so the embedding model / DB connection are only
# --- created once, on first use, not at import time.
_embedder = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(COLLECTION_NAME)
    return _collection


class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    if MOCK_LLM:
        intent = _classify_intent_heuristic(state["query"])
    else:
        intent = _classify_intent_llm(state["query"])
    return {**state, "intent": intent}


def _classify_intent_heuristic(query: str) -> str:
    query_lower = query.lower()
    if any(kw in query_lower for kw in POLICY_KEYWORDS):
        return "policy_question"
    return "general_question"


def _classify_intent_llm(query: str) -> str:
    # Optional MOCK_LLM=0 extension: ask the LLM to classify instead of the
    # keyword heuristic. Falls back to the heuristic if the LLM call fails.
    try:
        from llm_client import call_llm
        prompt = (
            "Classify this customer query as exactly one word, either "
            f"'policy_question' or 'general_question': {query}"
        )
        response = call_llm(prompt).strip().lower()
        return "policy_question" if "policy" in response else "general_question"
    except Exception:
        return _classify_intent_heuristic(query)


def route_after_classify(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    collection = get_collection()
    embedder = get_embedder()

    query_embedding = embedder.encode([state["query"]]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    ids = results["ids"][0]
    documents = results["documents"][0]
    retrieved_chunks = [{"id": ids[i], "text": documents[i]} for i in range(len(ids))]

    if MOCK_LLM:
        top_chunk_snippet = documents[0][:200] if documents else ""
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence = 1.0
    else:
        answer = _generate_answer_llm(state["query"], retrieved_chunks)
        confidence = 0.9

    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer,
        "sources": ids,
        "confidence": confidence,
    }


def _generate_answer_llm(query: str, chunks: List[Dict[str, Any]]) -> str:
    # Optional MOCK_LLM=0 extension: prompt a real LLM, grounded only in the
    # retrieved chunks, using the structured template from prompts.py.
    from prompts import PROMPT_TEMPLATE
    from llm_client import call_llm

    context = "\n".join(c["text"] for c in chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    return call_llm(prompt)


# ---------------------------------------------------------------------------
# Node 3: direct_answer
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
    else:
        answer = _direct_answer_llm(state["query"])
    return {**state, "answer": answer, "sources": [], "confidence": 1.0}


def _direct_answer_llm(query: str) -> str:
    # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
    from llm_client import call_llm
    return call_llm(f"Answer this general question concisely: {query}")


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()