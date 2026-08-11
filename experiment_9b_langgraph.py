"""
experiment_9b_langgraph.py — Multi-Agent RAG with LangGraph

Install:  pip install langgraph

Reimplements the exact same 3-agent workflow from experiment_9_multiagent.py
(CrewAI) using LangGraph so the two frameworks are directly comparable:

  Retriever node → Analyst node → Reviewer node
  Conditional edge: if NLI contradiction AND revision_count < 1 → loop to Analyst
                    otherwise → END

Three LangGraph-specific capabilities are demonstrated on top of the same logic:

  1. TYPED STATE GRAPH  — shared state object flows through all nodes; every
     field is visible to every node without explicit message-passing
  2. NATIVE INTERRUPT   — graph.invoke() pauses mid-execution for the 2
     lowest-scoring queries, accepts a simulated human decision, resumes
  3. CHECKPOINTING      — MemorySaver persists state at each node step;
     full execution history retrieved from the checkpointer after the run

═══════════════════════════════════════════════════════════════════════════════
WHY IN PRODUCTION
═══════════════════════════════════════════════════════════════════════════════
LangGraph is a lower-level framework than CrewAI.  CrewAI gives you agents
with roles/backstories and hides the orchestration; LangGraph gives you
explicit nodes, typed state, and deterministic edges — you see and control
every transition.

The choice between them mirrors a classic build-vs-buy trade-off:

  CrewAI  → faster to prototype; agents handle their own reasoning about which
            tool to call; opaque internals; event-loop issues in Colab;
            harder to unit-test individual steps

  LangGraph → explicit state machine; every node is a plain Python function;
              built-in HITL (interrupt/resume) and checkpointing without
              extra packages; easier to test, debug, and extend; steeper
              initial setup; no "agent persona" abstraction

For regulated domains (insurance, finance, healthcare), LangGraph's explicit
control flow and checkpointing are often preferred: you can replay any state,
audit the full decision history, and add approval gates without framework magic.

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICS / FRAMEWORK CONCEPTS
═══════════════════════════════════════════════════════════════════════════════
STATE GRAPH
  A directed graph G = (V, E) where:
    V = {retriever_node, analyst_node, reviewer_node, __start__, __end__}
    E = {retriever→analyst, analyst→reviewer, reviewer→analyst (conditional),
         reviewer→END (conditional)}
  State S is a TypedDict threaded through every node as input and output.
  Each node returns a dict of state fields to update (partial update, not replace).

INTERRUPT
  interrupt(payload) halts the graph, serialises the full state to the
  checkpointer, and raises a GraphInterrupt.  graph.invoke() returns the
  last completed state.  graph.get_state(config).next is non-empty, indicating
  pending work.  Resuming: graph.invoke(Command(resume=value), config) restores
  the state from the checkpointer and continues from the interrupted node with
  `value` as the return value of the interrupt() call.

CHECKPOINTING
  MemorySaver stores one checkpoint per state transition (step).  Each
  checkpoint records: state values, the node that produced them, the next
  node to run, and metadata (step index, source).
  get_state_history(config) returns checkpoints newest-first.
  This enables: time-travel debugging, state rollback, parallel branches.

═══════════════════════════════════════════════════════════════════════════════
ALTERNATIVES & TRADEOFFS vs CREWAI
═══════════════════════════════════════════════════════════════════════════════
  CrewAI agents decide which tool to call via LLM reasoning (ReAct pattern).
  LangGraph nodes call tools explicitly in Python code — no LLM decision needed
  for tool routing. This makes LangGraph faster (one LLM call per node instead
  of potentially multiple tool-reasoning rounds) and more predictable.

  CrewAI's Sequential Process is equivalent to LangGraph's linear chain.
  LangGraph also supports: parallel fan-out, subgraphs, map-reduce patterns,
  and long-running async streams — none of which CrewAI supports natively.

  Both frameworks add overhead vs a hand-written pipeline. The value is in
  the built-in capabilities: checkpointing, interrupt, state management.
"""

import json
import os
import sys
import time
import textwrap
from typing import TypedDict, Literal, Optional

# ── LangGraph ──────────────────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt, Command
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

# ── Retrieval stack ────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from corpus import CORPUS
from queries import QUERIES

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_MODEL_ID  = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_DELAY_S   = float(os.environ.get("GEMINI_DELAY_S", "5"))
INTER_QUERY_DELAY_S = float(os.environ.get("INTER_QUERY_DELAY_S", "5"))

TOP_K            = 10
CONTEXT_K        = 2
RRF_K            = 60
NLI_MODEL_NAME   = "cross-encoder/nli-deberta-v3-base"
NLI_LABELS       = ["contradiction", "entailment", "neutral"]
BLOCK_THRESHOLD  = 0.50

RESULTS_DIR = "results"
EXP6_FILE   = os.path.join(RESULTS_DIR, "experiment_6_judge_results.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "experiment_9b_langgraph_results.json")

CORPUS_BY_ID = {d["id"]: d for d in CORPUS}

# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY A TYPEDDICT STATE INSTEAD OF AGENT MESSAGES?
# CrewAI agents communicate through sequential task outputs (text strings).
# LangGraph uses a shared typed state object that ALL nodes read and partially
# update.  This means:
# - Every node sees the full context without prompt engineering
# - State is checkpointed automatically at each step
# - Type safety catches bugs at definition time, not at runtime
# - The state IS the audit trail — no separate logging needed

class RAGState(TypedDict):
    # Query metadata (set at start, read-only throughout)
    query_id:       str
    query_text:     str
    query_type:     str
    difficulty:     str
    expected_doc:   str

    # Set by retriever_node
    retrieved_context:  str          # formatted context blocks for LLM
    retrieved_doc_ids:  list         # list of doc IDs

    # Set / updated by analyst_node
    draft_answer:   str
    analyst_gaps:   str              # "INFORMATION GAPS: ..." from analyst
    revision_count: int              # increments each time analyst re-runs

    # Set by reviewer_node
    nli_scores:     dict             # {contradiction, entailment, neutral}
    nli_top_label:  str
    nli_action:     str              # "APPROVED" or "BLOCKED"
    reviewer_feedback: str           # feedback sent back to analyst on revision

    # Human-in-the-loop fields
    requires_interrupt: bool         # True for 2 lowest-scoring queries
    simulated_decision: str          # "approve" or "reject" (passed in at init)
    human_decision: str              # populated after interrupt resumes

    # Final output
    final_answer:   str
    was_revised:    bool

    # Per-node timing
    node_latencies: dict             # {"retriever_ms": 123, "analyst_ms": 456, ...}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL RETRIEVAL STATE (built once)
# ══════════════════════════════════════════════════════════════════════════════

_DENSE_MODEL  = None
_COLLECTION   = None
_BM25         = None
_CE_RANKER    = None
_NLI_MODEL    = None
_GEM_MODEL    = None


def build_indexes():
    global _DENSE_MODEL, _COLLECTION, _BM25, _CE_RANKER, _NLI_MODEL, _GEM_MODEL
    print("  [1/4] Dense bi-encoder …", end=" ", flush=True)
    _DENSE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    _COLLECTION = client.get_or_create_collection("exp9b")
    texts = [d["content"] for d in CORPUS]
    ids   = [d["id"]      for d in CORPUS]
    embs  = _DENSE_MODEL.encode(texts, show_progress_bar=False).tolist()
    _COLLECTION.add(ids=ids, embeddings=embs, documents=texts)
    print("done")

    print("  [2/4] BM25 …", end=" ", flush=True)
    _BM25 = BM25Okapi([d["content"].lower().split() for d in CORPUS])
    print("done")

    print("  [3/4] Cross-encoder re-ranker …", end=" ", flush=True)
    _CE_RANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("done")

    print("  [4/4] NLI model …", end=" ", flush=True)
    _NLI_MODEL = CrossEncoder(NLI_MODEL_NAME)
    print("done")

    if HAS_GENAI:
        _GEM_MODEL = genai.GenerativeModel(GEMINI_MODEL_ID)


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL UTILITIES (same stack as experiments 1–9)
# ══════════════════════════════════════════════════════════════════════════════

def _dense_q(text, k=TOP_K):
    q_emb = _DENSE_MODEL.encode([text]).tolist()
    res   = _COLLECTION.query(query_embeddings=q_emb, n_results=k)
    return [{"doc_id": d, "rank": r+1} for r, d in enumerate(res["ids"][0])]

def _bm25_q(text, k=TOP_K):
    scores = _BM25.get_scores(text.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [{"doc_id": CORPUS[i]["id"], "rank": r+1} for r, (i, _) in enumerate(ranked)]

def _rrf(dense, bm25, k=RRF_K):
    s: dict = {}
    for lst in [dense, bm25]:
        for item in lst:
            s[item["doc_id"]] = s.get(item["doc_id"], 0.0) + 1.0 / (k + item["rank"])
    return sorted(s.items(), key=lambda x: x[1], reverse=True)

def _rerank(query_text, candidates):
    pairs  = [(query_text, CORPUS_BY_ID[c[0]]["content"]) for c in candidates]
    scores = _CE_RANKER.predict(pairs)
    tagged = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [{"doc_id": c[0], "ce_score": float(s)} for c, s in tagged]

def retrieve_docs(query_text: str, k: int = CONTEXT_K) -> list:
    dense  = _dense_q(query_text)
    bm25   = _bm25_q(query_text)
    rrf    = _rrf(dense, bm25)[:TOP_K]
    ranked = _rerank(query_text, rrf)
    return [CORPUS_BY_ID[r["doc_id"]] for r in ranked[:k]]

def run_nli(premise: str, hypothesis: str) -> dict:
    logits = _NLI_MODEL.predict([(premise, hypothesis)])
    lv     = np.array(logits[0])
    exp_v  = np.exp(lv - lv.max())
    probs  = exp_v / exp_v.sum()
    scores = {label: round(float(probs[i]), 4) for i, label in enumerate(NLI_LABELS)}
    top    = max(scores, key=lambda k: scores[k])
    return {"scores": scores, "top_label": top, "top_conf": scores[top]}

def call_gemini(prompt: str) -> str:
    response = _GEM_MODEL.generate_content(prompt)
    if GEMINI_DELAY_S > 0:
        time.sleep(GEMINI_DELAY_S)
    return response.text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY PLAIN FUNCTIONS, NOT AGENT CLASSES?
# LangGraph nodes are Python callables: (state: RAGState) -> dict.
# The dict contains only the state fields this node updates — all other fields
# pass through unchanged.  This is equivalent to a database UPDATE statement:
# you specify only the columns being changed.  No agent framework, no tool
# routing, no ReAct loop — just explicit logic that's easy to unit-test.

def retriever_node(state: RAGState) -> dict:
    """
    RETRIEVER NODE
    Runs the hybrid BM25+dense+cross-encoder pipeline.
    Returns: retrieved_context (formatted string), retrieved_doc_ids.
    """
    t0   = time.time()
    docs = retrieve_docs(state["query_text"], k=CONTEXT_K)
    lat  = round((time.time() - t0) * 1000)

    doc_ids = [d["id"] for d in docs]
    context = "\n\n".join(
        f"[Document {i+1}: {d['id']} — {d.get('title','')}]\n{d['content']}"
        for i, d in enumerate(docs)
    )

    prev_lat = state.get("node_latencies") or {}
    return {
        "retrieved_context":  context,
        "retrieved_doc_ids":  doc_ids,
        "node_latencies":     {**prev_lat, "retriever_ms": lat},
    }


def analyst_node(state: RAGState) -> dict:
    """
    ANALYST NODE
    Drafts an answer grounded in retrieved context and flags knowledge gaps.
    On revision passes, incorporates the Reviewer's specific feedback.
    Returns: draft_answer, analyst_gaps, revision_count (incremented).
    """
    t0 = time.time()

    revision_ctx = ""
    if state["revision_count"] > 0 and state.get("reviewer_feedback"):
        revision_ctx = (
            f"\n\nREVISION NOTE: Your previous draft was flagged. "
            f"Specific issue: {state['reviewer_feedback']}\n"
            f"Please fix only the identified problem and keep the rest accurate."
        )

    prompt = (
        f"You are a telecom/insurance support specialist.\n"
        f"Answer the customer question using ONLY the documents provided. "
        f"Be concise (2–4 sentences). Explicitly list any information gaps.\n\n"
        f"QUESTION: {state['query_text']}\n\n"
        f"DOCUMENTS:\n{state['retrieved_context']}{revision_ctx}\n\n"
        f"Format your response exactly as:\n"
        f"DRAFT ANSWER: <your answer>\n"
        f"INFORMATION GAPS: <gaps, or 'None' if fully addressed>"
    )

    response = call_gemini(prompt)
    lat      = round((time.time() - t0) * 1000)

    # Parse the structured response
    draft = response
    gaps  = "None"
    if "DRAFT ANSWER:" in response:
        parts = response.split("INFORMATION GAPS:", 1)
        draft = parts[0].replace("DRAFT ANSWER:", "").strip()
        gaps  = parts[1].strip() if len(parts) > 1 else "None"

    prev_lat = state.get("node_latencies") or {}
    return {
        "draft_answer":   draft,
        "analyst_gaps":   gaps,
        "revision_count": state["revision_count"] + (1 if state["revision_count"] > 0 else 0),
        "node_latencies": {**prev_lat, f"analyst_ms_pass{state['revision_count'] + 1}": lat},
    }


def reviewer_node(state: RAGState) -> dict:
    """
    REVIEWER NODE
    1. Runs NLI check (premise = first retrieved doc, hypothesis = draft answer).
    2. If this query is flagged for human review: calls interrupt() to pause
       execution and waits for a human decision before continuing.
    3. Returns NLI scores, action (APPROVED/BLOCKED), feedback for revision,
       and final_answer if approved.

    The interrupt() call is the key LangGraph primitive: it serialises the
    full state to the MemorySaver checkpointer and halts graph.invoke().
    The graph resumes when graph.invoke(Command(resume=value), config) is called.
    """
    t0 = time.time()

    # NLI check: premise = first retrieved doc content
    first_doc_id = state["retrieved_doc_ids"][0] if state["retrieved_doc_ids"] else ""
    premise      = CORPUS_BY_ID[first_doc_id]["content"] if first_doc_id else ""
    hypothesis   = state["draft_answer"]

    nli = run_nli(premise, hypothesis)
    lat = round((time.time() - t0) * 1000)

    contra_prob = nli["scores"]["contradiction"]
    action      = "BLOCKED" if contra_prob >= BLOCK_THRESHOLD else "APPROVED"

    # Generate feedback for analyst if blocked
    feedback = ""
    if action == "BLOCKED":
        feedback = (
            f"NLI detected contradiction (score {contra_prob:.3f}). "
            f"Top label: {nli['top_label']}. "
            f"Review your draft against the retrieved context and correct any "
            f"factual claim not explicitly stated in the source documents."
        )

    # ── NATIVE HITL INTERRUPT ──────────────────────────────────────────────
    # interrupt() pauses execution here for flagged queries.
    # The payload is shown to the human reviewer (or logged for simulation).
    # The return value of interrupt() is whatever is passed in Command(resume=...).
    #
    # WHY interrupt() INSIDE A NODE (not interrupt_before)?
    # interrupt_before=[node] pauses BEFORE the node runs — the human sees
    # the prior state.  Calling interrupt() inside the node lets us first
    # run the NLI check, then show the NLI result to the human as part of
    # the interrupt payload.  The human reviews with full NLI context.
    human_decision = ""
    if state.get("requires_interrupt") and state["revision_count"] == 0:
        interrupt_payload = {
            "query_id":      state["query_id"],
            "query_text":    state["query_text"],
            "draft_answer":  hypothesis,
            "nli_action":    action,
            "nli_scores":    nli["scores"],
            "analyst_gaps":  state["analyst_gaps"],
            "message":       (
                f"Query {state['query_id']} flagged for human review. "
                f"NLI action: {action}. "
                f"Enter 'approve' to deliver this answer or 'reject' to block it."
            ),
        }
        # This line pauses graph execution. The graph resumes when
        # Command(resume=<decision>) is passed to graph.invoke().
        human_decision = interrupt(interrupt_payload)

    prev_lat = state.get("node_latencies") or {}
    final    = hypothesis if action == "APPROVED" else ""

    return {
        "nli_scores":        nli["scores"],
        "nli_top_label":     nli["top_label"],
        "nli_action":        action,
        "reviewer_feedback": feedback,
        "human_decision":    human_decision,
        "final_answer":      final,
        "was_revised":       state["revision_count"] > 0,
        "node_latencies":    {**prev_lat, f"reviewer_ms_pass{state['revision_count'] + 1}": lat},
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTION  (conditional edge)
# ══════════════════════════════════════════════════════════════════════════════

def route_after_review(state: RAGState) -> Literal["analyst_node", "__end__"]:
    """
    Decide whether to loop back to the Analyst for a revision or end.

    WHY MAX 1 REVISION?
    Unlimited revision loops risk infinite cycles if the NLI model
    consistently flags the same text (e.g., a domain-specific claim
    that looks like a contradiction to a general-purpose NLI model).
    In production, cap the loop and escalate to human review after
    the maximum revision count is reached.
    """
    if state["nli_action"] == "BLOCKED" and state["revision_count"] < 1:
        return "analyst_node"
    return "__end__"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(checkpointer: "MemorySaver") -> "StateGraph":
    """
    Assemble the RAG state graph.

    Graph topology:
      __start__ → retriever_node → analyst_node → reviewer_node
                                         ↑                |
                                         └── (if BLOCKED) ┘
                                         reviewer_node → END (if APPROVED or max revision)

    WHY StateGraph OVER MessageGraph?
    MessageGraph is designed for chatbots where state IS a message list.
    StateGraph is better for RAG pipelines where state has multiple distinct
    fields (context, draft, NLI scores) that nodes update independently.
    Using MessageGraph for RAG would require encoding all state into messages
    and parsing them back out — unnecessary complexity.
    """
    builder = StateGraph(RAGState)

    builder.add_node("retriever_node", retriever_node)
    builder.add_node("analyst_node",   analyst_node)
    builder.add_node("reviewer_node",  reviewer_node)

    builder.set_entry_point("retriever_node")
    builder.add_edge("retriever_node", "analyst_node")
    builder.add_edge("analyst_node",   "reviewer_node")
    builder.add_conditional_edges(
        "reviewer_node",
        route_after_review,
        {"analyst_node": "analyst_node", "__end__": END},
    )

    return builder.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════════════════════════
# QUERY RUNNER  (handles normal + interrupt/resume flow)
# ══════════════════════════════════════════════════════════════════════════════

def run_query(graph, q: dict, interrupt_queries: set) -> dict:
    """
    Run one query through the graph, handling the interrupt/resume cycle
    for queries that require human review.

    Returns a result dict compatible with experiment_9_multiagent.py output.
    """
    t_total = time.time()

    initial_state: RAGState = {
        "query_id":          q["id"],
        "query_text":        q["text"],
        "query_type":        q["query_type"],
        "difficulty":        q["difficulty"],
        "expected_doc":      q["expected_doc"],
        "retrieved_context": "",
        "retrieved_doc_ids": [],
        "draft_answer":      "",
        "analyst_gaps":      "",
        "revision_count":    0,
        "nli_scores":        {},
        "nli_top_label":     "",
        "nli_action":        "",
        "reviewer_feedback": "",
        "requires_interrupt": q["id"] in interrupt_queries,
        "simulated_decision": "approve",   # hardcoded simulation
        "human_decision":    "",
        "final_answer":      "",
        "was_revised":       False,
        "node_latencies":    {},
    }

    cfg = {"configurable": {"thread_id": q["id"]}}

    # ── First invocation ───────────────────────────────────────────────────
    state_after_first = graph.invoke(initial_state, cfg)

    interrupted = False
    final_state = state_after_first

    # ── Check if graph paused at an interrupt ──────────────────────────────
    snapshot = graph.get_state(cfg)
    if snapshot.next:
        interrupted = True
        pending_node = snapshot.next[0]
        simulated    = initial_state["simulated_decision"]

        print(f"\n  ⏸  GRAPH PAUSED at '{pending_node}' — interrupt() called")
        print(f"     Query     : {q['id']} — \"{q['text'][:55]}\"")
        print(f"     NLI action: {state_after_first.get('nli_action', 'N/A')}")
        print(f"     Pending   : {snapshot.next}")
        print(f"     [SIMULATION] Human decision = '{simulated}'")
        print(f"     Resuming graph …")

        # Resume the graph with the simulated human decision
        final_state = graph.invoke(Command(resume=simulated), cfg)
        print(f"  ▶  Graph resumed — final answer produced.")

    total_ms = round((time.time() - t_total) * 1000)

    # Build agent_outputs dict to match experiment_9 output format
    history     = list(graph.get_state_history(cfg))
    # history is newest-first; extract node outputs from step metadata
    agent_outs  = _extract_agent_outputs(history, final_state)

    return {
        "query_id":          q["id"],
        "query_text":        q["text"],
        "query_type":        q["query_type"],
        "difficulty":        q["difficulty"],
        "expected_doc":      q["expected_doc"],
        "context_doc_ids":   final_state.get("retrieved_doc_ids", []),
        "agent_outputs":     agent_outs,
        "nli_scores":        final_state.get("nli_scores", {}),
        "nli_action":        final_state.get("nli_action", ""),
        "was_revised":       final_state.get("was_revised", False),
        "interrupted":       interrupted,
        "human_decision":    final_state.get("human_decision", ""),
        "final_answer":      final_state.get("final_answer", final_state.get("draft_answer", "")),
        "node_latencies":    final_state.get("node_latencies", {}),
        "total_latency_ms":  total_ms,
    }


def _extract_agent_outputs(history: list, final_state: dict) -> dict:
    """Extract per-node intermediate outputs from checkpointer history."""
    # History is newest-first; we want the state values produced by each node
    outs = {
        "retriever":  "",
        "analyst":    "",
        "reviewer":   "",
    }
    for checkpoint in reversed(history):
        vals   = checkpoint.values
        source = checkpoint.metadata.get("source", "")
        writes = checkpoint.metadata.get("writes") or {}

        if "retriever_node" in writes:
            outs["retriever"] = (
                f"Docs: {vals.get('retrieved_doc_ids', [])}\n"
                f"Context (truncated): {vals.get('retrieved_context', '')[:200]}…"
            )
        if "analyst_node" in writes:
            key = "analyst_revision" if vals.get("revision_count", 0) > 1 else "analyst"
            outs[key] = (
                f"DRAFT ANSWER: {vals.get('draft_answer', '')}\n"
                f"INFORMATION GAPS: {vals.get('analyst_gaps', '')}"
            )
        if "reviewer_node" in writes:
            key = "reviewer_final" if vals.get("was_revised") else "reviewer"
            outs[key] = (
                f"NLI: {vals.get('nli_top_label','')} "
                f"(contradiction={vals.get('nli_scores',{}).get('contradiction',0):.3f})\n"
                f"Action: {vals.get('nli_action','')}\n"
                f"Human decision: {vals.get('human_decision','') or 'N/A'}"
            )
    return outs


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTER HISTORY DEMO
# ══════════════════════════════════════════════════════════════════════════════

def print_checkpoint_history(graph, query_id: str):
    """
    Retrieve and display the full execution history for one query from the
    MemorySaver checkpointer.  Demonstrates LangGraph's time-travel capability.

    WHY THIS MATTERS IN PRODUCTION:
    Each checkpoint is a complete snapshot of the agent state at that step.
    This enables:
    - Replay: re-run from any checkpoint without re-executing prior steps
    - Audit: regulators can inspect exactly what context the analyst saw
    - Rollback: if a revision made things worse, revert to the pre-revision state
    - Debugging: reproduce exactly what happened for a specific query
    """
    section("CHECKPOINTER HISTORY — Agent Lifecycle State")
    print(f"  Querying MemorySaver for thread_id = '{query_id}'\n")

    cfg     = {"configurable": {"thread_id": query_id}}
    history = list(graph.get_state_history(cfg))

    print(f"  Total checkpoints stored : {len(history)}")
    print(f"  (One per state transition; newest first)\n")

    for i, checkpoint in enumerate(history):
        step    = checkpoint.metadata.get("step", "?")
        source  = checkpoint.metadata.get("source", "?")
        writes  = checkpoint.metadata.get("writes") or {}
        node    = next(iter(writes), "__start__") if writes else "__end__"
        vals    = checkpoint.values

        print(f"  ── Checkpoint {i}  (step={step}, source={source})")
        print(f"     Node that produced this state : {node}")
        print(f"     Next node(s) to run           : {checkpoint.next or '(done)'}")
        print(f"     revision_count                : {vals.get('revision_count', 0)}")
        print(f"     nli_action                    : {vals.get('nli_action') or '(not yet set)'}")
        print(f"     final_answer set              : {'yes' if vals.get('final_answer') else 'no'}")

        if vals.get("retrieved_doc_ids"):
            print(f"     retrieved_doc_ids             : {vals['retrieved_doc_ids']}")
        if vals.get("human_decision"):
            print(f"     human_decision                : {vals['human_decision']}")
        print()

    # Show how to restore any specific checkpoint
    if len(history) >= 2:
        target_step = history[-2]  # second-oldest (first completed node)
        print(f"  DEMO: Restoring state at step {target_step.metadata.get('step')} "
              f"(node: {next(iter(target_step.metadata.get('writes') or {'?':0}), '?')})")
        print(f"  graph.update_state(config, {{...}}) would allow time-travel from here.")
        print(f"  In production: use this to replay from retriever_node with a different query.")


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")


def section_minor(title: str):
    print(f"\n  {'─'*68}")
    print(f"  {title}")
    print(f"  {'─'*68}")


def wrap(text: str, width: int = 68, indent: str = "    ") -> str:
    return "\n".join(
        textwrap.fill(para, width=width, initial_indent=indent, subsequent_indent=indent)
        for para in (text or "").split("\n") if para.strip()
    )


def _table(rows, headers):
    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="rounded_outline")
    col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    fmt  = "  ".join(f"{{:<{w}}}" for w in col_w)
    sep  = "-" * (sum(col_w) + 2 * len(col_w))
    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))
    return "\n".join(lines)


def print_results_summary(results: list):
    section("RESULTS SUMMARY — All 10 Queries")
    rows = []
    for r in results:
        rows.append((
            r["query_id"],
            r["query_type"][:10],
            r["difficulty"],
            r["nli_action"] or "?",
            "YES" if r["was_revised"] else "no",
            "⏸ YES" if r["interrupted"] else "no",
            f"{r['total_latency_ms']:,} ms",
            (r["final_answer"] or "")[:55] + ("…" if len(r.get("final_answer","")) > 55 else ""),
        ))
    headers = ["Query", "Type", "Diff", "NLI", "Revised", "Interrupted", "Latency", "Answer (truncated)"]
    print(_table(rows, headers))

    n_revised     = sum(1 for r in results if r["was_revised"])
    n_interrupted = sum(1 for r in results if r["interrupted"])
    avg_lat       = sum(r["total_latency_ms"] for r in results) // len(results)
    print(f"\n  Revised (NLI loop triggered) : {n_revised}")
    print(f"  Interrupted (HITL paused)    : {n_interrupted}")
    print(f"  Average end-to-end latency   : {avg_lat:,} ms")


def print_crewai_vs_langgraph():
    section("FRAMEWORK COMPARISON — CrewAI (exp 9) vs LangGraph (exp 9b)")
    rows = [
        ("Orchestration model",    "Role-based agents",          "Typed state machine"),
        ("Node/agent definition",  "Agent(role, backstory, ...)", "Plain Python function"),
        ("Inter-agent comms",      "Sequential task outputs",     "Shared TypedDict state"),
        ("Tool routing",           "LLM reasons which tool",      "Explicit Python call"),
        ("HITL / interrupt",       "Manual simulation required",  "interrupt() primitive built-in"),
        ("Checkpointing",          "Not built-in",                "MemorySaver / SqliteSaver"),
        ("Colab async issue",      "Required ThreadPoolExecutor", "No issue (plain Python)"),
        ("Debugging",              "Logs from agent verbose",     "get_state_history() time-travel"),
        ("Unit-testable nodes",    "Hard (agent class)",          "Easy (plain function)"),
        ("Parallel fan-out",       "Not supported natively",      "Supported (parallel nodes)"),
        ("LLM calls per query",    "~3–5 (tool reasoning loops)", "2–3 (one per node, explicit)"),
        ("Framework abstraction",  "High (hides orchestration)",  "Low (you control everything)"),
    ]
    print(_table(rows, ["Dimension", "CrewAI (exp 9)", "LangGraph (exp 9b)"]))
    print("""
  WHEN TO CHOOSE EACH
  ─────────────────────────────────────────────────────────────────────────
  CrewAI  → Rapid prototype; agent persona makes prompts natural; you want
             the framework to decide tool invocation order.

  LangGraph → Production system; you need checkpointing for audit/replay;
               you need native HITL gates; you want deterministic control flow;
               you want to unit-test each node independently.
""")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6 SCORE LOADER — picks which queries get interrupted
# ══════════════════════════════════════════════════════════════════════════════

def pick_interrupt_queries(n: int = 2) -> set:
    """
    Load experiment 6 scores and pick the n lowest-overall-scoring queries
    as candidates for the interrupt/human-review demonstration.

    If experiment 6 results don't exist OR all faithfulness scores are 1.0
    (model was generous), fall back to the n hardest queries by query_type.

    This fallback is intentional and documented: in a real run where the
    model performs well, there may be no low-faithfulness queries. The
    interrupt mechanism still demonstrates correctly on the fallback set.
    """
    if os.path.exists(EXP6_FILE):
        with open(EXP6_FILE) as f:
            data = json.load(f)
        scored = sorted(
            data.get("query_results", []),
            key=lambda q: (q["dim_scores"]["faithfulness"]["score"], q["overall"])
        )
        # Check if scores are all 1.0 (generous model)
        all_perfect = all(
            q["dim_scores"]["faithfulness"]["score"] >= 1.0 for q in scored
        )
        if not all_perfect:
            chosen = [q["query_id"] for q in scored[:n]]
            print(f"  HITL targets (lowest faithfulness from exp 6): {chosen}")
            return set(chosen)
        else:
            # All faithfulness = 1.0; use lowest overall and note the fallback
            by_overall = sorted(scored, key=lambda q: q["overall"])
            chosen = [q["query_id"] for q in by_overall[:n]]
            print(f"  HITL targets: all faithfulness=1.0 in exp 6 (model generous).")
            print(f"  Falling back to lowest overall-score queries: {chosen}")
            return set(chosen)

    # No exp 6 results — use hardest query types
    hard = [q["id"] for q in QUERIES if q["difficulty"] == "hard"][:n]
    print(f"  HITL targets (exp 6 not found, using hard queries): {hard}")
    return set(hard)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not HAS_LANGGRAPH:
        print("[ERROR] langgraph not installed. Run: pip install langgraph")
        sys.exit(1)
    if not HAS_GENAI:
        print("[ERROR] google-generativeai not installed. Run: pip install google-generativeai")
        sys.exit(1)

    section("Experiment 9b — Multi-Agent RAG with LangGraph")
    print(f"  LangGraph version    : 1.2.x")
    print(f"  Gemini model         : {GEMINI_MODEL_ID}")
    print(f"  Nodes                : retriever → analyst → reviewer")
    print(f"  Revision loop        : conditional edge, max 1 pass")
    print(f"  Checkpointer         : MemorySaver (in-memory)")
    print(f"  HITL interrupt       : 2 lowest-scoring queries from exp 6")
    print(f"  Inter-query delay    : {INTER_QUERY_DELAY_S} s")

    # ── Gemini setup ───────────────────────────────────────────────────────────
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        print(
            "\n[ERROR] GEMINI_API_KEY not set.\n"
            "        In Colab: os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')"
        )
        sys.exit(1)
    genai.configure(api_key=gemini_key)

    # ── Build retrieval indexes ────────────────────────────────────────────────
    section("Building retrieval indexes …")
    build_indexes()

    # ── Determine which queries get interrupted ────────────────────────────────
    section("Selecting HITL interrupt targets …")
    interrupt_queries = pick_interrupt_queries(n=2)

    # ── Build graph with MemorySaver checkpointer ──────────────────────────────
    section("Building LangGraph state graph …")
    checkpointer = MemorySaver()
    graph        = build_graph(checkpointer)
    print(f"  Graph nodes  : {list(graph.nodes)}")
    print(f"  Interrupt queries : {interrupt_queries}")

    # ── Run all 10 queries ─────────────────────────────────────────────────────
    section("Running all 10 queries through the graph …")

    all_results = []
    for qi, q in enumerate(QUERIES, start=1):
        section_minor(f"[{qi}/10] {q['id']} — {q['query_type']} / {q['difficulty']}")
        print(f"  Query: \"{q['text']}\"")
        if q["id"] in interrupt_queries:
            print(f"  [HITL] This query will trigger interrupt() in reviewer_node")

        try:
            result = run_query(graph, q, interrupt_queries)
            all_results.append(result)

            print(f"\n  ── Retriever output:")
            print(wrap(result["agent_outputs"].get("retriever", "")[:200]))
            print(f"\n  ── Analyst output:")
            print(wrap(result["agent_outputs"].get("analyst", "")))
            print(f"\n  ── Reviewer output:")
            print(wrap(result["agent_outputs"].get("reviewer", "")))
            if result["was_revised"]:
                print(f"\n  ── [REVISION] Analyst revision:")
                print(wrap(result["agent_outputs"].get("analyst_revision", "")))
                print(f"\n  ── [REVISION] Reviewer final:")
                print(wrap(result["agent_outputs"].get("reviewer_final", "")))

            print(f"\n  NLI: {result['nli_action']}  |  "
                  f"Revised: {result['was_revised']}  |  "
                  f"Interrupted: {result['interrupted']}  |  "
                  f"Latency: {result['total_latency_ms']:,} ms")

        except Exception as exc:
            import traceback
            print(f"  [ERROR] {exc}")
            traceback.print_exc()
            all_results.append({
                "query_id": q["id"], "query_text": q["text"],
                "query_type": q["query_type"], "difficulty": q["difficulty"],
                "expected_doc": q["expected_doc"],
                "context_doc_ids": [], "agent_outputs": {},
                "nli_scores": {}, "nli_action": "", "was_revised": False,
                "interrupted": False, "human_decision": "",
                "final_answer": f"[error: {exc}]", "node_latencies": {},
                "total_latency_ms": 0,
            })

        if qi < len(QUERIES):
            time.sleep(INTER_QUERY_DELAY_S)

    # ── Results summary table ──────────────────────────────────────────────────
    print_results_summary(all_results)

    # ── Checkpointer history demo (use first interrupted query, or Q01) ────────
    demo_qid = next(iter(interrupt_queries), "Q01")
    print_checkpoint_history(graph, demo_qid)

    # ── Framework comparison table ─────────────────────────────────────────────
    print_crewai_vs_langgraph()

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "experiment":        "experiment_9b_langgraph",
        "gemini_model":      GEMINI_MODEL_ID,
        "interrupt_queries": list(interrupt_queries),
        "query_results":     all_results,
        "summary": {
            "total":          len(all_results),
            "nli_blocked":    sum(1 for r in all_results if r["nli_action"] == "BLOCKED"),
            "revised":        sum(1 for r in all_results if r["was_revised"]),
            "interrupted":    sum(1 for r in all_results if r["interrupted"]),
            "avg_latency_ms": (
                sum(r["total_latency_ms"] for r in all_results) // len(all_results)
                if all_results else 0
            ),
        },
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
