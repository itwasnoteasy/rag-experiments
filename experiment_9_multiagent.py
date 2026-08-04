"""
experiment_9_multiagent.py — Multi-Agent RAG Orchestration with CrewAI

Install:  pip install crewai

Central question: "When does multi-agent orchestration actually help
versus just adding latency?"

Three specialized agents cooperate on a sequential pipeline:
  1. RETRIEVER AGENT — runs hybrid BM25 + dense + cross-encoder retrieval
  2. ANALYST AGENT  — drafts an answer AND explicitly flags knowledge gaps
  3. REVIEWER AGENT — runs the NLI guardrail from experiment 8 and either
                       approves the draft or sends it back for one revision

For the 3 hardest queries (Q04, Q06, Q08) the multi-agent crew is compared
side-by-side against the single-agent baseline (retrieve + synthesise in one
step). Latency is measured for both paths.

═══════════════════════════════════════════════════════════════════════════════
WHY IN PRODUCTION
═══════════════════════════════════════════════════════════════════════════════
Single-agent RAG (retrieve → synthesise) works well when:
  • The query is unambiguous and retrieval reliably surfaces the right doc
  • The answer is short and fully contained in one chunk
  • Latency is the primary constraint

Multi-agent orchestration earns its cost when:
  • Answers require synthesising across multiple documents or steps
  • A quality-checking loop is mandated (regulated domains: insurance, finance)
  • Queries are ambiguous and benefit from explicit gap-flagging before answer

The Retriever / Analyst / Reviewer separation mirrors the architecture at
companies like Glean (separate retrieval and synthesis services), Cohere
(Rerank API as an independent service), and enterprise LLM platforms that
enforce answer validation before customer delivery.

The honest finding this experiment targets: multi-agent adds latency regardless
of whether it improves quality. The question is whether the quality delta
justifies the latency cost — and the answer depends on the query type.

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICS / FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════
CrewAI uses a sequential Process where each Task receives the output of all
prior Tasks as context (passed in the `context` list). Agents communicate
exclusively through structured text — no shared memory or direct function calls
between agents. The orchestrator is a thin wrapper that:
  1. Feeds Task[i].output as context to Task[i+1]
  2. Calls the LLM for each agent's reasoning step
  3. Invokes tools when the agent's output contains a tool-use request

Revision loop: if the Reviewer's output contains "REVISION NEEDED", a
second mini-crew (Analyst + Reviewer only) runs with the feedback injected
into the task description. Maximum 1 revision regardless of second outcome.

Latency model:
  Single-agent: T_retrieval + T_synthesis
  Multi-agent:  T_retrieval_agent + T_analyst + T_nli_tool + T_reviewer
                + (T_analyst_revision + T_nli_tool + T_reviewer_final) if revised

═══════════════════════════════════════════════════════════════════════════════
ALTERNATIVES & TRADEOFFS
═══════════════════════════════════════════════════════════════════════════════
• LangGraph: graph-based orchestration with explicit state machine and
  conditional edges. More expressive than CrewAI for non-linear workflows
  (loops, branching, parallel subgraphs). Higher setup complexity.

• AutoGen (Microsoft): conversation-based multi-agent where agents literally
  send messages to each other in a chat loop. Natural for dialogue-style
  tasks but harder to control termination conditions.

• LlamaIndex Workflows: event-driven orchestration; steps emit events that
  trigger other steps. Good for async/streaming pipelines.

• Custom orchestrator (no framework): explicit Python function calls between
  retrieval, analysis, and review steps. Maximum control, zero magic. The
  single-agent baseline in this experiment IS this pattern.

Key tradeoff: frameworks (CrewAI, LangGraph) add abstraction cost (learning
curve, opaque internals, version churn) in exchange for built-in agent
memory, tool routing, and observability. For a production team < 5 engineers,
a custom orchestrator is often the better starting point.
"""

import json
import os
import re
import sys
import time
import textwrap
from typing import Any

# ── CrewAI ─────────────────────────────────────────────────────────────────────
try:
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.tools import tool as crewai_tool
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False

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

# CrewAI with Gemini via LiteLLM: model string must be "gemini/<model-id>"
GEMINI_MODEL_ID  = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
CREWAI_MODEL     = f"gemini/{GEMINI_MODEL_ID}"

TOP_K            = 10
CONTEXT_K        = 2
RRF_K            = 60
NLI_MODEL_NAME   = "cross-encoder/nli-deberta-v3-base"
NLI_LABELS       = ["contradiction", "entailment", "neutral"]
BLOCK_THRESHOLD  = 0.50

# Rate limiting: CrewAI manages its own LLM calls internally.
# INTER_QUERY_DELAY_S sleeps between queries so the per-minute budget recovers.
# Each query makes ~3 LLM calls (one per agent step). At 15 RPM free-tier,
# ~20 s between queries keeps burst rate safe.
INTER_QUERY_DELAY_S = float(os.environ.get("INTER_QUERY_DELAY_S", "20"))

# Gemini baseline delay for single-agent calls (same as exp 5-7)
GEMINI_DELAY_S = float(os.environ.get("GEMINI_DELAY_S", "5"))

# The 3 hardest queries for single-agent vs multi-agent comparison
HARD_QUERY_IDS  = ["Q04", "Q06", "Q08"]

RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "experiment_9_multiagent_results.json")

CORPUS_BY_ID = {d["id"]: d for d in CORPUS}

# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL STATE  (built once, shared by all tool calls)
# ══════════════════════════════════════════════════════════════════════════════
# CrewAI tool functions are module-level callables; they cannot receive class
# instances as arguments. We store the retrieval indexes and NLI model in
# module-level variables and initialise them in main() before the crew runs.

_DENSE_MODEL   = None
_COLLECTION    = None
_BM25          = None
_CE_RANKER     = None
_NLI_MODEL     = None


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL STACK  (self-contained, consistent with experiments 1-8)
# ══════════════════════════════════════════════════════════════════════════════

def build_indexes():
    """Build all retrieval indexes and load NLI model. Called once in main()."""
    global _DENSE_MODEL, _COLLECTION, _BM25, _CE_RANKER, _NLI_MODEL

    print("  [1/4] Dense bi-encoder …", end=" ", flush=True)
    _DENSE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    client       = chromadb.Client()
    _COLLECTION  = client.get_or_create_collection("exp9")
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


def _dense_query(query_text, top_k=TOP_K):
    q_emb = _DENSE_MODEL.encode([query_text]).tolist()
    res   = _COLLECTION.query(query_embeddings=q_emb, n_results=top_k)
    return [{"doc_id": d, "rank": r + 1} for r, d in enumerate(res["ids"][0])]


def _bm25_query(query_text, top_k=TOP_K):
    tokens = query_text.lower().split()
    scores = _BM25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"doc_id": CORPUS[i]["id"], "rank": r + 1} for r, (i, _) in enumerate(ranked)]


def _rrf(dense, bm25, k=RRF_K):
    s: dict[str, float] = {}
    for lst in [dense, bm25]:
        for item in lst:
            s[item["doc_id"]] = s.get(item["doc_id"], 0.0) + 1.0 / (k + item["rank"])
    return sorted(s.items(), key=lambda x: x[1], reverse=True)


def _rerank(query_text, candidates):
    pairs  = [(query_text, CORPUS_BY_ID[c[0]]["content"]) for c in candidates]
    scores = _CE_RANKER.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [{"doc_id": c[0], "ce_score": float(s)} for c, s in ranked]


def retrieve_top_docs(query_text: str, k: int = CONTEXT_K) -> list[dict]:
    """Full retrieval pipeline: RRF → cross-encoder → top-k docs."""
    dense   = _dense_query(query_text)
    bm25    = _bm25_query(query_text)
    rrf     = _rrf(dense, bm25)[:TOP_K]
    ranked  = _rerank(query_text, rrf)
    return [CORPUS_BY_ID[r["doc_id"]] for r in ranked[:k]]


def _nli(premise: str, hypothesis: str) -> dict:
    logits  = _NLI_MODEL.predict([(premise, hypothesis)])
    lv      = np.array(logits[0])
    exp_v   = np.exp(lv - lv.max())
    probs   = exp_v / exp_v.sum()
    scores  = {label: round(float(probs[i]), 4) for i, label in enumerate(NLI_LABELS)}
    top     = max(scores, key=lambda k: scores[k])
    return {"scores": scores, "top_label": top, "top_conf": scores[top]}


# ══════════════════════════════════════════════════════════════════════════════
# CREWAI TOOLS
# ══════════════════════════════════════════════════════════════════════════════

# WHY TOOLS AS MODULE-LEVEL FUNCTIONS?
# CrewAI's @tool decorator wraps a plain function into a StructuredTool that the
# agent's LLM can call via tool-use syntax. The decorator reads the function's
# type hints and docstring to generate the tool schema shown to the LLM.
# Tools must be module-level callables — they cannot be methods or closures over
# local variables, because CrewAI serialises the tool description at class
# instantiation time. Module-level _GLOBAL state is the standard pattern for
# injecting pre-built indexes into CrewAI tools.

@crewai_tool("hybrid_retrieval")
def hybrid_retrieval_tool(query: str) -> str:
    """
    Retrieve the most relevant documents for a query using the hybrid pipeline
    (BM25 + dense embedding + cross-encoder re-ranking). Returns formatted
    context blocks ready for answer drafting.

    Args:
        query: The customer question to retrieve documents for.

    Returns:
        Formatted string with document ID, title, and full content for each
        top-ranked document.
    """
    docs   = retrieve_top_docs(query, k=CONTEXT_K)
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(
            f"[Document {i}: {d['id']} — {d.get('title', '')}]\n{d['content']}"
        )
    return "\n\n".join(blocks) if blocks else "No relevant documents found."


@crewai_tool("nli_contradiction_check")
def nli_contradiction_check_tool(premise_and_hypothesis: str) -> str:
    """
    Check whether an answer contradicts its source context using a Natural
    Language Inference (NLI) model. Returns a verdict with confidence scores
    and a recommended action (APPROVED or REVISION NEEDED).

    Args:
        premise_and_hypothesis: A string in the exact format:
            PREMISE: <the retrieved context text>
            HYPOTHESIS: <the drafted answer to check>

    Returns:
        NLI verdict string including label, confidence scores, and action.
    """
    # Parse the formatted input
    premise    = ""
    hypothesis = ""
    for line in premise_and_hypothesis.splitlines():
        if line.startswith("PREMISE:"):
            premise    = line[len("PREMISE:"):].strip()
        elif line.startswith("HYPOTHESIS:"):
            hypothesis = line[len("HYPOTHESIS:"):].strip()

    # Fallback: if parse fails, split on double newline
    if not premise or not hypothesis:
        parts = premise_and_hypothesis.split("\n\n", 1)
        if len(parts) == 2:
            premise, hypothesis = parts[0].strip(), parts[1].strip()
        else:
            return "ERROR: Could not parse premise and hypothesis. Use format: PREMISE: ... HYPOTHESIS: ..."

    result = _nli(premise, hypothesis)
    s      = result["scores"]
    action = "BLOCKED — REVISION NEEDED" if s["contradiction"] >= BLOCK_THRESHOLD else "APPROVED"

    return (
        f"NLI Result:\n"
        f"  contradiction : {s['contradiction']:.3f}\n"
        f"  entailment    : {s['entailment']:.3f}\n"
        f"  neutral       : {s['neutral']:.3f}\n"
        f"  Top label     : {result['top_label']} ({result['top_conf']:.3f})\n"
        f"  Action        : {action}\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT FACTORY
# ══════════════════════════════════════════════════════════════════════════════

# WHY AGENTS ARE CREATED PER-EXPERIMENT NOT PER-QUERY?
# CrewAI Agents hold their LLM reference and tool list. They are stateless with
# respect to individual queries — all query-specific context is injected via the
# Task description and context mechanism. Creating agents once and reusing them
# across queries is the correct pattern; it mirrors how a real microservice
# architecture would deploy fixed-capability agents with variable inputs.

def build_agents(llm: "LLM") -> tuple:
    """Create and return (retriever, analyst, reviewer) agents."""

    retriever = Agent(
        role="Document Retriever",
        goal=(
            "Given a customer query, use the hybrid_retrieval tool to fetch the "
            "most relevant documents from the knowledge base. Return the retrieved "
            "context clearly formatted with document IDs and full content."
        ),
        backstory=(
            "You are a specialist in information retrieval for a telecom and insurance "
            "company. You have access to a hybrid BM25 + semantic search pipeline with "
            "cross-encoder re-ranking. Your only job is to retrieve; you do not answer "
            "questions — you provide raw context for the Analyst."
        ),
        tools=[hybrid_retrieval_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    analyst = Agent(
        role="Answer Analyst",
        goal=(
            "Given a customer query and retrieved context, draft a concise, accurate "
            "answer grounded strictly in the provided documents. Also explicitly list "
            "any information gaps — things the customer asked about that the retrieved "
            "context does not fully address."
        ),
        backstory=(
            "You are a senior customer support specialist for a telecom and insurance "
            "company. You are meticulous: you only state what the documents say, and "
            "you always flag when the context is incomplete. You format your output as:\n"
            "DRAFT ANSWER: <your answer>\n"
            "INFORMATION GAPS: <gaps, or 'None' if fully addressed>"
        ),
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="Quality Reviewer",
        goal=(
            "Review the Analyst's draft answer for factual consistency with the "
            "retrieved context. Use the nli_contradiction_check tool to run an NLI "
            "check. If the NLI result is APPROVED, output 'APPROVED: <final answer>'. "
            "If BLOCKED, output 'REVISION NEEDED: <specific feedback on what to fix>' "
            "so the Analyst can correct the draft."
        ),
        backstory=(
            "You are a quality assurance reviewer at a regulated telecom and insurance "
            "company. Before any AI-generated response reaches a customer, you verify it "
            "does not contradict the source documents. You use an NLI model as your "
            "primary tool. You are the last gate before customer delivery."
        ),
        tools=[nli_contradiction_check_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return retriever, analyst, reviewer


# ══════════════════════════════════════════════════════════════════════════════
# CREW RUNNER
# ══════════════════════════════════════════════════════════════════════════════

# WHY TASKS ARE CREATED PER QUERY (NOT PER AGENT)?
# CrewAI Tasks hold query-specific descriptions and context references. The
# `context` field of a Task is a list of other Task objects whose .output will
# be injected into this task's prompt at runtime. This is how agent-to-agent
# communication happens: the orchestrator wires outputs, not the agents themselves.
# Creating new Task objects per query ensures no context bleeds between queries.

def run_crew(query: dict, retriever, analyst, reviewer) -> dict:
    """
    Run the full 3-agent crew for one query. Returns a result dict with
    intermediate outputs, final answer, revision flag, and latency.
    """
    t_start = time.time()

    # ── Step 1: Retriever Task ──────────────────────────────────────────────
    retriever_task = Task(
        description=(
            f"Retrieve the most relevant documents for the following customer query "
            f"using the hybrid_retrieval tool. Return the full document content.\n\n"
            f"QUERY: {query['text']}"
        ),
        expected_output=(
            "Retrieved documents formatted as numbered blocks with document ID, "
            "title, and full content for each document."
        ),
        agent=retriever,
    )

    # ── Step 2: Analyst Task ────────────────────────────────────────────────
    analyst_task = Task(
        description=(
            f"Draft an answer for the following customer query using ONLY the "
            f"retrieved context provided by the Retriever. "
            f"Be concise (2-4 sentences). Explicitly flag any information gaps.\n\n"
            f"QUERY: {query['text']}\n\n"
            f"Format your response as:\n"
            f"DRAFT ANSWER: <your answer>\n"
            f"INFORMATION GAPS: <list gaps, or 'None' if fully addressed>"
        ),
        expected_output=(
            "A structured response with DRAFT ANSWER and INFORMATION GAPS sections."
        ),
        agent=analyst,
        context=[retriever_task],
    )

    # ── Step 3: Reviewer Task ───────────────────────────────────────────────
    reviewer_task = Task(
        description=(
            f"Review the Analyst's draft answer for factual consistency with the "
            f"retrieved context. Use the nli_contradiction_check tool with:\n"
            f"  PREMISE: <first retrieved document content, truncated to 500 chars>\n"
            f"  HYPOTHESIS: <the DRAFT ANSWER text only>\n\n"
            f"Based on the NLI result:\n"
            f"- If APPROVED: output 'APPROVED: <the analyst's draft answer>'\n"
            f"- If BLOCKED:  output 'REVISION NEEDED: <specific sentence to fix and why>'"
        ),
        expected_output=(
            "Either 'APPROVED: <answer>' or 'REVISION NEEDED: <specific feedback>'"
        ),
        agent=reviewer,
        context=[retriever_task, analyst_task],
    )

    crew = Crew(
        agents=[retriever, analyst, reviewer],
        tasks=[retriever_task, analyst_task, reviewer_task],
        process=Process.sequential,
        verbose=False,
    )

    crew.kickoff()

    retriever_out = retriever_task.output.raw if retriever_task.output else ""
    analyst_out   = analyst_task.output.raw   if analyst_task.output   else ""
    reviewer_out  = reviewer_task.output.raw  if reviewer_task.output  else ""

    # ── Revision loop (max 1) ───────────────────────────────────────────────
    revision_out   = None
    revision_final = None
    revised        = False

    if reviewer_out and "REVISION NEEDED" in reviewer_out.upper():
        revised  = True
        feedback = reviewer_out

        analyst_rev_task = Task(
            description=(
                f"The Reviewer flagged your previous draft. Revise it based on "
                f"the specific feedback below.\n\n"
                f"ORIGINAL QUERY: {query['text']}\n\n"
                f"REVIEWER FEEDBACK:\n{feedback}\n\n"
                f"Use the same retrieved context. Produce a corrected answer.\n"
                f"Format:\n"
                f"REVISED ANSWER: <corrected answer>\n"
                f"INFORMATION GAPS: <updated gaps or 'None'>"
            ),
            expected_output="REVISED ANSWER and INFORMATION GAPS sections.",
            agent=analyst,
            context=[retriever_task],
        )

        reviewer_final_task = Task(
            description=(
                f"Run a final NLI check on the revised answer using the "
                f"nli_contradiction_check tool. Output 'APPROVED: <answer>' or "
                f"'STILL PROBLEMATIC: <reason>' — this is the final verdict."
            ),
            expected_output="'APPROVED: <answer>' or 'STILL PROBLEMATIC: <reason>'",
            agent=reviewer,
            context=[retriever_task, analyst_rev_task],
        )

        revision_crew = Crew(
            agents=[analyst, reviewer],
            tasks=[analyst_rev_task, reviewer_final_task],
            process=Process.sequential,
            verbose=False,
        )
        revision_crew.kickoff()

        revision_out   = analyst_rev_task.output.raw  if analyst_rev_task.output  else ""
        revision_final = reviewer_final_task.output.raw if reviewer_final_task.output else ""

    # ── Extract final answer ────────────────────────────────────────────────
    final_text = revision_final if revision_final else reviewer_out
    # Strip the APPROVED: prefix for clean display
    clean = re.sub(r"^APPROVED:\s*", "", final_text, flags=re.IGNORECASE).strip()

    total_ms = round((time.time() - t_start) * 1000)

    return {
        "query_id":        query["id"],
        "query_text":      query["text"],
        "query_type":      query["query_type"],
        "difficulty":      query["difficulty"],
        "expected_doc":    query["expected_doc"],
        "agent_outputs": {
            "retriever":       retriever_out[:600] + ("…" if len(retriever_out) > 600 else ""),
            "analyst":         analyst_out,
            "reviewer":        reviewer_out,
            "analyst_revision": revision_out,
            "reviewer_final":  revision_final,
        },
        "revised":         revised,
        "final_answer":    clean,
        "total_latency_ms": total_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-AGENT BASELINE
# ══════════════════════════════════════════════════════════════════════════════

# WHY MEASURE THE BASELINE?
# The multi-agent crew always adds latency from sequential agent calls and
# tool invocations. Whether it adds quality is empirical — it depends on the
# query type. By running the same queries through a single retrieve+synthesise
# call, we can isolate the quality delta from the latency cost and answer
# "was the extra complexity worth it?"

def run_single_agent(query: dict) -> dict:
    """
    Baseline: retrieve top docs then call Gemini once to synthesise.
    No agent framework — plain function calls. Same retrieval stack.
    """
    if not HAS_GENAI:
        return {"query_id": query["id"], "final_answer": "[google-generativeai not installed]",
                "total_latency_ms": 0}

    t_start = time.time()

    docs    = retrieve_top_docs(query["text"], k=CONTEXT_K)
    context = "\n\n".join(
        f"[Document {i+1}: {d['id']} — {d.get('title','')}]\n{d['content']}"
        for i, d in enumerate(docs)
    )
    prompt  = (
        "You are a helpful telecom support assistant.\n\n"
        "Answer the following customer question using ONLY the documents provided. "
        "Be concise (2–4 sentences). If information is missing, say so.\n\n"
        f"Question: {query['text']}\n\nDocuments:\n{context}\n\nAnswer:"
    )

    gem_model = genai.GenerativeModel(GEMINI_MODEL_ID)
    response  = gem_model.generate_content(prompt)
    if GEMINI_DELAY_S > 0:
        time.sleep(GEMINI_DELAY_S)

    answer   = response.text.strip()
    total_ms = round((time.time() - t_start) * 1000)

    return {
        "query_id":         query["id"],
        "query_text":       query["text"],
        "context_doc_ids":  [d["id"] for d in docs],
        "final_answer":     answer,
        "total_latency_ms": total_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")


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


def wrap(text: str, width: int = 68, indent: str = "    ") -> str:
    return "\n".join(
        textwrap.fill(para, width=width, initial_indent=indent,
                      subsequent_indent=indent)
        for para in text.split("\n") if para.strip()
    )


def print_intermediate_outputs(result: dict):
    """Print each agent's output for one query."""
    ao = result["agent_outputs"]
    print(f"\n  ── Retriever output (truncated):")
    print(wrap(ao["retriever"][:300] + "…", width=70))
    print(f"\n  ── Analyst output:")
    print(wrap(ao["analyst"], width=70))
    print(f"\n  ── Reviewer output:")
    print(wrap(ao["reviewer"], width=70))
    if result["revised"]:
        print(f"\n  ── [REVISION] Analyst revision:")
        print(wrap(ao.get("analyst_revision", ""), width=70))
        print(f"\n  ── [REVISION] Reviewer final:")
        print(wrap(ao.get("reviewer_final", ""), width=70))


def print_crew_summary(crew_results: list[dict]):
    section("CREW RESULTS — All 10 Queries")
    rows = []
    for r in crew_results:
        rows.append((
            r["query_id"],
            r["query_type"][:10],
            r["difficulty"],
            "YES" if r["revised"] else "no",
            f"{r['total_latency_ms']:,} ms",
            r["final_answer"][:55] + ("…" if len(r["final_answer"]) > 55 else ""),
        ))
    headers = ["Query", "Type", "Difficulty", "Revised", "Latency", "Final Answer (truncated)"]
    print(_table(rows, headers))

    revised_count = sum(1 for r in crew_results if r["revised"])
    avg_lat = sum(r["total_latency_ms"] for r in crew_results) // len(crew_results)
    print(f"\n  Queries that triggered a revision loop : {revised_count} / {len(crew_results)}")
    print(f"  Average end-to-end latency (crew)     : {avg_lat:,} ms")


def print_comparison(hard_results_crew: list[dict], hard_results_single: list[dict]):
    section("SIDE-BY-SIDE COMPARISON — 3 Hardest Queries")
    print(f"  Queries: {', '.join(HARD_QUERY_IDS)}")
    print(f"  These were selected because they are the most likely to expose")
    print(f"  quality differences: semantic paraphrase gap (Q04), ambiguity")
    print(f"  (Q06), and near-empty context-dependent query (Q08).\n")

    crew_by_id   = {r["query_id"]: r for r in hard_results_crew}
    single_by_id = {r["query_id"]: r for r in hard_results_single}

    lat_rows = []
    for qid in HARD_QUERY_IDS:
        cr = crew_by_id.get(qid, {})
        sr = single_by_id.get(qid, {})

        print(f"  {'─'*68}")
        q_obj = next(q for q in QUERIES if q["id"] == qid)
        print(f"  {qid}  [{q_obj['query_type']} / {q_obj['difficulty']}]")
        print(f"  Query: \"{q_obj['text']}\"")
        print()

        sa = sr.get("final_answer", "N/A")
        ca = cr.get("final_answer", "N/A")

        print(f"  SINGLE-AGENT ({sr.get('total_latency_ms', 0):,} ms):")
        print(wrap(sa[:350] + ("…" if len(sa) > 350 else ""), width=70))

        print(f"\n  3-AGENT CREW ({cr.get('total_latency_ms', 0):,} ms)"
              f"{'  [revision triggered]' if cr.get('revised') else ''}:")
        print(wrap(ca[:350] + ("…" if len(ca) > 350 else ""), width=70))

        if cr.get("revised"):
            print(f"\n  NOTE: The Reviewer flagged the Analyst's first draft and requested")
            print(f"        a revision. The final answer above is post-revision.")

        print()

        lat_rows.append((
            qid,
            q_obj["query_type"],
            f"{sr.get('total_latency_ms', 0):,} ms",
            f"{cr.get('total_latency_ms', 0):,} ms",
            f"{cr.get('total_latency_ms', 0) - sr.get('total_latency_ms', 0):+,} ms",
            "YES" if cr.get("revised") else "no",
        ))

    section("LATENCY COST OF MULTI-AGENT ORCHESTRATION")
    headers = ["Query", "Type", "Single-Agent", "3-Agent Crew", "Overhead", "Revised"]
    print(_table(lat_rows, headers))

    total_single = sum(r.get("total_latency_ms", 0) for r in hard_results_single)
    total_crew   = sum(r.get("total_latency_ms", 0) for r in hard_results_crew)
    overhead     = total_crew - total_single
    mult         = round(total_crew / total_single, 1) if total_single else 0

    print(f"\n  Total across 3 queries:")
    print(f"    Single-agent : {total_single:,} ms")
    print(f"    3-agent crew : {total_crew:,} ms")
    print(f"    Overhead     : +{overhead:,} ms  ({mult}× slower)")

    print(f"""
  HONEST ASSESSMENT
  ─────────────────────────────────────────────────────────────────────────
  Q04 (semantic/hard — paraphrase gap):
    The hardest retrieval case. Multi-agent may help if the Reviewer catches
    the Analyst overstating certainty when context is tangentially relevant.
    If retrieval itself failed (wrong doc at rank 1), neither pipeline can fix
    it — multi-agent does not improve retrieval quality, only response quality.

  Q06 (ambiguous — plan change vs. billing):
    The Analyst's explicit INFORMATION GAPS flag is the key differentiator here.
    A single-agent pipeline commits to one interpretation silently; the Analyst
    step surfaces ambiguity the customer can resolve with a follow-up question.

  Q08 (context-dependent — "how long does that usually take?"):
    Near-empty query with no context history. Both pipelines are handicapped.
    The multi-agent crew may produce a longer hedged response that admits
    uncertainty; the single agent may hallucinate a specific timeline. The
    Reviewer / NLI layer should catch numeric fabrications here.

  WHEN MULTI-AGENT IS WORTH IT:
  • Response quality is a hard constraint (regulated domain, customer-facing)
  • Queries frequently require surfacing knowledge gaps, not just answering
  • You have a revision SLA (e.g., async review before customer delivery)
  • The additional latency is within your UX budget (async response, email)

  WHEN SINGLE-AGENT WINS:
  • Sub-200 ms latency is required (chat, autocomplete)
  • Queries are structured and retrieval is high-precision
  • The added complexity of agent orchestration outweighs the marginal quality gain
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not HAS_CREWAI:
        print("[ERROR] crewai not installed. Run: pip install crewai")
        sys.exit(1)

    section("Experiment 9 — Multi-Agent RAG with CrewAI")
    print(f"  CrewAI model         : {CREWAI_MODEL}")
    print(f"  Agents               : Retriever → Analyst → Reviewer")
    print(f"  Revision loop        : max 1 pass")
    print(f"  Hard queries (compare): {', '.join(HARD_QUERY_IDS)}")
    print(f"  Inter-query delay    : {INTER_QUERY_DELAY_S} s  (rate-limit guard)")

    # ── Gemini / CrewAI API key ────────────────────────────────────────────────
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        print(
            "\n[ERROR] GEMINI_API_KEY not set.\n"
            "        In Colab: os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')"
        )
        sys.exit(1)

    # CrewAI uses LiteLLM which reads GEMINI_API_KEY automatically
    os.environ["GEMINI_API_KEY"] = gemini_key

    # Also wire direct Gemini SDK for single-agent baseline
    if HAS_GENAI:
        genai.configure(api_key=gemini_key)

    # ── Build retrieval indexes ────────────────────────────────────────────────
    section("Building retrieval indexes and loading models …")
    build_indexes()

    # ── Build CrewAI LLM and agents ────────────────────────────────────────────
    llm = LLM(
        model=CREWAI_MODEL,
        api_key=gemini_key,
        temperature=0.1,    # low temperature for factual accuracy
        timeout=120,
    )
    retriever_agent, analyst_agent, reviewer_agent = build_agents(llm)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: Run all 10 queries through the crew
    # ══════════════════════════════════════════════════════════════════════════
    section("PART 1 — Running all 10 queries through 3-agent crew")

    crew_all_results = []

    for qi, q in enumerate(QUERIES, start=1):
        section_minor(f"[{qi}/10] {q['id']} — {q['query_type']} / {q['difficulty']}")
        print(f"  Query: \"{q['text']}\"")

        try:
            result = run_crew(q, retriever_agent, analyst_agent, reviewer_agent)
            crew_all_results.append(result)
            print(f"  Revised: {result['revised']}  |  Latency: {result['total_latency_ms']:,} ms")
            print(f"  Final  : {result['final_answer'][:100]}{'…' if len(result['final_answer']) > 100 else ''}")

            # Show intermediate outputs
            print_intermediate_outputs(result)

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            crew_all_results.append({
                "query_id": q["id"], "query_text": q["text"],
                "query_type": q["query_type"], "difficulty": q["difficulty"],
                "expected_doc": q["expected_doc"],
                "agent_outputs": {}, "revised": False,
                "final_answer": f"[error: {exc}]", "total_latency_ms": 0,
            })

        if qi < len(QUERIES):
            print(f"\n  [rate-limit guard] sleeping {INTER_QUERY_DELAY_S:.0f} s …")
            time.sleep(INTER_QUERY_DELAY_S)

    print_crew_summary(crew_all_results)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: Single-agent baseline for the 3 hard queries
    # ══════════════════════════════════════════════════════════════════════════
    section("PART 2 — Single-agent baseline for 3 hard queries")

    hard_queries    = [q for q in QUERIES if q["id"] in HARD_QUERY_IDS]
    single_results  = []
    hard_crew       = [r for r in crew_all_results if r["query_id"] in HARD_QUERY_IDS]

    for qi, q in enumerate(hard_queries, start=1):
        print(f"  [{qi}/{len(hard_queries)}] {q['id']}: \"{q['text'][:60]}…\"  ", end="", flush=True)
        result = run_single_agent(q)
        single_results.append(result)
        print(f"{result['total_latency_ms']:,} ms")
        if qi < len(hard_queries):
            time.sleep(INTER_QUERY_DELAY_S)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: Side-by-side comparison + latency analysis
    # ══════════════════════════════════════════════════════════════════════════
    print_comparison(hard_crew, single_results)

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "experiment":       "experiment_9_multiagent",
        "crewai_model":     CREWAI_MODEL,
        "hard_query_ids":   HARD_QUERY_IDS,
        "crew_results":     crew_all_results,
        "single_agent_results": single_results,
        "summary": {
            "total_queries":       len(crew_all_results),
            "queries_revised":     sum(1 for r in crew_all_results if r["revised"]),
            "avg_crew_latency_ms": (
                sum(r["total_latency_ms"] for r in crew_all_results)
                // len(crew_all_results) if crew_all_results else 0
            ),
            "hard_query_latency_comparison": [
                {
                    "query_id":    r["query_id"],
                    "single_ms":   next((s["total_latency_ms"] for s in single_results
                                        if s["query_id"] == r["query_id"]), 0),
                    "crew_ms":     r["total_latency_ms"],
                    "revised":     r["revised"],
                }
                for r in hard_crew
            ],
        },
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {OUTPUT_FILE}")


def section_minor(title: str):
    print(f"\n  {'─'*68}")
    print(f"  {title}")
    print(f"  {'─'*68}")


if __name__ == "__main__":
    main()
