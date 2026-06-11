"""
Experiment 4: End-to-End Pipeline with Langfuse Observability Tracing
======================================================================

WHY THIS EXPERIMENT EXISTS IN PRODUCTION
-----------------------------------------
A RAG pipeline is a chain of 5–8 sequential steps (classify → embed →
retrieve → re-rank → synthesise). Each step has its own latency, failure
modes, and quality characteristics. Without instrumentation you are flying
blind: you cannot answer "why did this query return a bad answer?", "which
step is causing our P95 latency to spike?", or "did the cross-encoder actually
improve anything last week?" Langfuse is an open-source LLM observability
platform that records every step as a hierarchical trace, lets you attach
inputs/outputs/metadata to each span, and surfaces dashboards, cost tracking,
and evaluation scores.

In production this tracing layer is non-negotiable for any RAG system in
customer-facing use:
  - Support engineers use traces to debug why a specific query got the wrong
    answer (which span returned the wrong document).
  - ML engineers use span latencies to identify bottlenecks and plan
    infrastructure scaling.
  - Product managers use token counts and Gemini call counts to track LLM
    cost per query and per user.
  - Evaluators attach human feedback or automated scores (e.g. G-Eval,
    RAGAS) to traces in Langfuse's evaluation framework.

Run:
    # Set environment variables first:
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export GOOGLE_API_KEY=...          # for Gemini Flash synthesis
    python experiment_4_langfuse.py

    # Or inline on one line:
    LANGFUSE_PUBLIC_KEY=pk-... LANGFUSE_SECRET_KEY=sk-... GOOGLE_API_KEY=... python experiment_4_langfuse.py

    # In Colab, set secrets via:
    # from google.colab import userdata
    # os.environ["LANGFUSE_PUBLIC_KEY"] = userdata.get("LANGFUSE_PUBLIC_KEY")

Outputs:
    - Per-query trace confirmation to stdout
    - Latency breakdown table to stdout
    - results/experiment_4_latency_report.json
    - Traces visible in Langfuse dashboard within seconds of each run
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import google.generativeai as genai
from langfuse import Langfuse
from langfuse.types import TraceContext
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import pipeline as hf_pipeline
from tabulate import tabulate
import chromadb

from corpus import CORPUS
from queries import QUERIES
from experiment_3_intent import (
    build_classifier,
    classify_query,
    routing_decision,
    INTENT_CLASSES,
    PRIMARY_THRESHOLD,
)

RESULTS_DIR  = "results"
OUTPUT_FILE  = os.path.join(RESULTS_DIR, "experiment_4_latency_report.json")

TOP_K        = 5
DISPLAY_K    = 3
RRF_K        = 60
GEMINI_MODEL = "gemini-2.0-flash"

DOC_CONTENT = {doc["id"]: doc["content"] for doc in CORPUS}
DOC_TITLE   = {doc["id"]: doc["title"]   for doc in CORPUS}


# =============================================================================
# SECTION 1 — LANGFUSE CLIENT SETUP
# =============================================================================
#
# WHY IN PRODUCTION
# -----------------
# Langfuse uses an async, buffered background thread to ship trace data to its
# ingestion API (cloud or self-hosted). This means tracing adds negligible
# latency to your application — the spans are written to an in-memory queue
# and flushed in background batches. The only hard requirement is calling
# langfuse.flush() before process exit to drain the queue; without it, the
# last few spans may be lost. In production you typically call flush() in a
# shutdown hook or use the Python atexit module.
#
# SELF-HOSTED vs CLOUD
# --------------------
# Langfuse Cloud (langfuse.com): free tier, no infrastructure to manage.
#   Use LANGFUSE_HOST = "https://cloud.langfuse.com" (default).
# Self-hosted (Docker/Kubernetes): full data residency, required for PII-heavy
#   workloads (healthcare, finance). Set LANGFUSE_HOST to your own endpoint.
#   The self-hosted stack is Docker Compose or Helm chart — documented at
#   langfuse.com/docs/deployment/self-host.
#
# ALTERNATIVES & TRADEOFFS
# -------------------------
# • LangSmith (LangChain's observability product)
#     Pro: native integration with LangChain/LangGraph; excellent UI for
#          chain debugging; strong eval framework.
#     Con: closed-source, LangChain-centric; free tier limited; vendor lock-in
#          if you later move off LangChain.
#
# • Arize Phoenix
#     Pro: open-source, local-first, excellent for ML teams; built on
#          OpenTelemetry so spans are vendor-neutral; strong embedding
#          visualisation (UMAP plots of retrieved vs. expected docs).
#     Con: less polished UI than Langfuse; smaller community.
#
# • OpenTelemetry + Jaeger / Zipkin
#     Pro: vendor-neutral standard; works for all services not just LLMs;
#          integrates with existing APM infrastructure (Datadog, Grafana).
#     Con: no LLM-specific concepts (token counts, prompts, model params)
#          out of the box; requires custom instrumentation for each step.
#     Note: Langfuse v4 is built ON TOP of OpenTelemetry, so spans created
#          by Langfuse are valid OTEL spans and can be exported to any
#          OTEL-compatible backend simultaneously.
#
# • Helicone
#     Pro: plug-and-play proxy for OpenAI/Anthropic; zero code changes needed;
#          automatic cost tracking.
#     Con: only instruments LLM calls, not retrieval steps; less useful for
#          full RAG pipeline visibility.
#
# • Custom logging to a data warehouse (BigQuery, Snowflake)
#     Pro: full control over schema; integrates with existing BI tooling.
#     Con: you build the UI, alerting, and query tooling yourself; high
#          engineering cost to replicate what Langfuse provides out of the box.

def setup_langfuse() -> Langfuse:
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        print("ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment")
        print("       variables must be set before running this experiment.")
        print()
        print("  In Colab:")
        print("    from google.colab import userdata")
        print("    import os")
        print('    os.environ["LANGFUSE_PUBLIC_KEY"] = userdata.get("LANGFUSE_PUBLIC_KEY")')
        print('    os.environ["LANGFUSE_SECRET_KEY"] = userdata.get("LANGFUSE_SECRET_KEY")')
        print('    os.environ["GOOGLE_API_KEY"]      = userdata.get("GOOGLE_API_KEY")')
        sys.exit(1)

    lf = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        # host defaults to https://cloud.langfuse.com
        # For self-hosted: host="https://your-langfuse.internal"
    )

    ok = lf.auth_check()
    if not ok:
        print("ERROR: Langfuse auth_check() failed. Verify your keys.")
        sys.exit(1)

    print("=== Langfuse client initialised ===")
    print(f"  Project authenticated: OK")
    print(f"  Trace dashboard: https://cloud.langfuse.com")
    print()
    return lf


# =============================================================================
# SECTION 2 — MODEL LOADING (done once, reused across all queries)
# =============================================================================
#
# WHY LOAD ONCE IN PRODUCTION
# ---------------------------
# Model loading is the dominant startup cost: a 400M-parameter classifier
# like BART takes 2–5 seconds on CPU; sentence-transformers and cross-encoders
# take 1–3 seconds each. In a production service these are loaded at server
# startup and kept warm in memory for the lifetime of the process. Reloading
# per query would make the service unusable. This is why production RAG
# systems are deployed as long-running API servers (FastAPI, Flask, Ray Serve)
# rather than scripts — the models stay resident in RAM/GPU memory.
#
# GEMINI SETUP
# ------------
# google-generativeai is Google's Python SDK for Gemini models. It uses an
# API key for authentication (no OAuth required for developer use). In
# production you would use Vertex AI instead of the API key approach:
#   - Vertex AI uses IAM service accounts (no keys to rotate or leak).
#   - Vertex AI is required for VPC Service Controls and data residency.
#   - The generativeai SDK supports both; switch by calling
#     vertexai.init(project=..., location=...) instead of genai.configure().

def load_all_models():
    print("=== Loading all pipeline models ===")

    print("  [1/4] Intent classifier (facebook/bart-large-mnli)...")
    clf = build_classifier()

    print("  [2/4] Bi-encoder for dense retrieval (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("  [3/4] Cross-encoder for re-ranking (ms-marco-MiniLM-L-6-v2)...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("  [4/4] Gemini Flash synthesis model...")
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        print("WARNING: GOOGLE_API_KEY not set. Synthesis span will be skipped.")
        gemini = None
    else:
        genai.configure(api_key=google_api_key)
        gemini = genai.GenerativeModel(GEMINI_MODEL)
        print(f"  Gemini model: {GEMINI_MODEL}")

    print()
    return clf, embedder, reranker, gemini


def build_indexes(embedder):
    print("=== Building retrieval indexes ===")

    # Dense index
    client = chromadb.Client()
    try:
        client.delete_collection("rag_exp4")
    except Exception:
        pass
    collection = client.create_collection(
        "rag_exp4", metadata={"hnsw:space": "cosine"}
    )
    doc_ids    = [doc["id"]      for doc in CORPUS]
    doc_texts  = [doc["content"] for doc in CORPUS]
    embeddings = embedder.encode(doc_texts, show_progress_bar=True).tolist()
    collection.add(
        ids=doc_ids,
        embeddings=embeddings,
        documents=doc_texts,
        metadatas=[{"title": doc["title"]} for doc in CORPUS],
    )
    print(f"  Dense index: {len(doc_ids)} documents in ChromaDB.")

    # BM25 index
    bm25 = BM25Okapi([doc["content"].lower().split() for doc in CORPUS])
    print(f"  BM25 index: {len(CORPUS)} documents.")
    print()

    return collection, bm25


# =============================================================================
# SECTION 3 — SPAN HELPERS
# =============================================================================
#
# WHY STRUCTURED SPANS IN PRODUCTION
# ------------------------------------
# A span is the unit of observability in a distributed trace. Each span
# records: name, start time, end time, input, output, and metadata.
# Structuring spans hierarchically (trace → span → child span) lets you:
#   - See the full query lifecycle as a waterfall diagram in the Langfuse UI
#   - Identify which step in the pipeline was slow or wrong for a specific query
#   - Aggregate latencies across thousands of traces to find systemic bottlenecks
#   - Attach evaluation scores (e.g. RAGAS faithfulness) at the trace level
#
# SPAN TYPES IN LANGFUSE v4
# --------------------------
# Langfuse v4 uses OpenTelemetry span types with semantic conventions:
#   "span"       : generic step (default)
#   "retriever"  : document retrieval step — shows retrieved docs in UI
#   "embedding"  : embedding generation — shows model name and vector size
#   "generation" : LLM call — shows prompt, completion, token counts, cost
#   "chain"      : a sequence of steps (wraps the whole pipeline)
#
# Using the right type is important: the Langfuse UI renders retrievers
# with their document list, and generations with token usage and cost
# estimation. Generic "span" types get plain input/output display.

def ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


# =============================================================================
# SECTION 4 — INSTRUMENTED PIPELINE
# =============================================================================
#
# WHY INSTRUMENT EVERY STEP INDIVIDUALLY
# ----------------------------------------
# Coarse-grained tracing (one span for the entire pipeline) tells you the
# total latency but not where the time went. Fine-grained span-per-step
# tracing is what makes observability actionable. When a query takes 4 seconds
# you need to know: was it the embedding (GPU contention?), the ChromaDB query
# (index fragmentation?), the cross-encoder (too many candidates?), or Gemini
# (rate limit / long response?)? Each answer leads to a different fix.
#
# SYNTHESIS WITH GEMINI FLASH
# ----------------------------
# This experiment uses Gemini 2.0 Flash as the generation model. Flash is
# Google's fastest production-grade model — designed for high-throughput
# applications where latency matters more than maximum reasoning depth.
#
# MATHEMATICS — WHY TOP-2 DOCUMENTS AS CONTEXT
# ---------------------------------------------
# We pass only the top-2 re-ranked documents to Gemini rather than all 5.
# This is a deliberate context window management decision. Each document is
# ~150 words (~200 tokens). Passing 5 documents would use ~1000 prompt tokens;
# passing 2 uses ~400. For a customer support bot where answers typically
# require one specific policy clause, top-2 is sufficient and reduces:
#   - Input token cost (billed per token with Gemini)
#   - Risk of the "lost in the middle" attention dropout effect
#   - Hallucination risk from conflicting context passages
# In production, the optimal k (documents in context) is determined by
# offline evaluation: try k=1,2,3,5 and measure answer correctness.
#
# ALTERNATIVES FOR SYNTHESIS LLM
# --------------------------------
# • Gemini 2.0 Flash (this experiment)
#     Pro: very fast (~300–600ms), low cost (~$0.075/1M input tokens),
#          generous free tier, 1M token context window.
#     Con: slightly lower accuracy than Gemini Ultra or GPT-4o on complex
#          multi-hop reasoning; Google Cloud dependency.
#
# • Claude Haiku (Anthropic)
#     Pro: extremely fast, cheap, excellent instruction following; low
#          hallucination rate on RAG tasks.
#     Con: Anthropic API key required; no self-hosted option.
#
# • GPT-4o-mini (OpenAI)
#     Pro: very cost-effective, strong performance; widely supported in
#          LangChain/LlamaIndex integrations.
#     Con: data goes to OpenAI; GDPR considerations for EU users.
#
# • Ollama + Llama 3.1 8B (local)
#     Pro: fully offline, no data leaves your machine, free.
#     Con: 3–10× slower than API models on CPU; requires local GPU for
#          production-grade throughput; you own the hosting and updates.
#
# • vLLM + Mistral 7B (self-hosted GPU)
#     Pro: production-grade serving with continuous batching; no per-token cost
#          once GPU is provisioned; full data residency.
#     Con: significant infrastructure investment; model quality below frontier
#          API models for complex reasoning tasks.

SYNTHESIS_PROMPT_TEMPLATE = """\
You are a helpful telecom and insurance support assistant.
Answer the customer's question using ONLY the context documents provided.
Be concise (2–3 sentences). If the answer is not in the context, say so.

Context document 1:
{doc1_title}
{doc1_content}

Context document 2:
{doc2_title}
{doc2_content}

Customer question: {query}

Answer:"""


def run_instrumented_query(
    query: dict,
    langfuse: Langfuse,
    clf,
    embedder,
    collection,
    bm25,
    reranker,
    gemini,
) -> dict:
    """
    Runs the full RAG pipeline for a single query with Langfuse tracing.
    Each step is wrapped in a span that records input, output, and latency.
    Returns a dict of per-span latency measurements for the local report.
    """
    query_text = query["text"]
    latencies  = {}

    # ── Create a trace ID and open a top-level chain span ────────────────────
    trace_id   = Langfuse.create_trace_id(seed=query["id"])
    trace_ctx  = TraceContext(trace_id=trace_id, name=f"rag-pipeline: {query['id']}")

    # Set trace-level metadata (visible in Langfuse as the top row)
    langfuse.set_current_trace_io(
        input={"query": query_text, "query_type": query["query_type"]},
    )

    pipeline_t0 = time.perf_counter()

    # ── Span 1: Intent Classification ────────────────────────────────────────
    t0 = time.perf_counter()
    with langfuse.start_as_current_observation(
        trace_context=trace_ctx,
        name="intent-classification",
        as_type="span",
        input={"query": query_text, "candidate_labels": [ic["label"] for ic in INTENT_CLASSES]},
    ) as span:
        ranked_classes = classify_query(clf, query_text)
        route          = routing_decision(ranked_classes, threshold=PRIMARY_THRESHOLD)
        lat = ms_since(t0)
        span.update(
            output={
                "predicted_intent": ranked_classes[0]["label"],
                "confidence":       ranked_classes[0]["score"],
                "all_scores":       ranked_classes,
                "routing_decision": route["decision"],
                "routed_namespace": route["namespace"],
            },
            metadata={"latency_ms": lat},
        )
    latencies["intent-classification"] = lat
    print(f"    intent-classification:   {lat:7.1f} ms  → {ranked_classes[0]['label']} ({ranked_classes[0]['score']:.2f})")

    # ── Span 2: Embedding ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with langfuse.start_as_current_observation(
        trace_context=trace_ctx,
        name="embedding",
        as_type="embedding",
        input={"query": query_text, "model": "all-MiniLM-L6-v2"},
    ) as span:
        q_embedding = embedder.encode(query_text).tolist()
        lat = ms_since(t0)
        span.update(
            output={"vector_dim": len(q_embedding), "model": "all-MiniLM-L6-v2"},
            metadata={"latency_ms": lat},
        )
    latencies["embedding"] = lat
    print(f"    embedding:               {lat:7.1f} ms  → dim={len(q_embedding)}")

    # ── Span 3: Dense Retrieval ───────────────────────────────────────────────
    t0 = time.perf_counter()
    with langfuse.start_as_current_observation(
        trace_context=trace_ctx,
        name="retrieval-dense",
        as_type="retriever",
        input={"query": query_text, "top_k": TOP_K},
    ) as span:
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=TOP_K,
            include=["distances", "metadatas"],
        )
        dense_results = [
            {"doc_id": did, "score": round(1 - dist, 4), "rank": i + 1}
            for i, (did, dist) in enumerate(
                zip(results["ids"][0], results["distances"][0])
            )
        ]
        lat = ms_since(t0)
        span.update(
            output={"top_docs": [r["doc_id"] for r in dense_results[:DISPLAY_K]],
                    "scores":   [r["score"]  for r in dense_results[:DISPLAY_K]]},
            metadata={"latency_ms": lat},
        )
    latencies["retrieval-dense"] = lat
    print(f"    retrieval-dense:         {lat:7.1f} ms  → {[r['doc_id'] for r in dense_results[:3]]}")

    # ── Span 4: BM25 Retrieval ────────────────────────────────────────────────
    t0 = time.perf_counter()
    with langfuse.start_as_current_observation(
        trace_context=trace_ctx,
        name="retrieval-bm25",
        as_type="retriever",
        input={"query": query_text, "top_k": TOP_K},
    ) as span:
        tokens     = query_text.lower().split()
        bm25_scores = bm25.get_scores(tokens)
        ranked_idx  = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:TOP_K]
        bm25_results = [
            {"doc_id": CORPUS[idx]["id"], "score": round(score, 4), "rank": i + 1}
            for i, (idx, score) in enumerate(ranked_idx)
        ]
        lat = ms_since(t0)
        span.update(
            output={"top_docs": [r["doc_id"] for r in bm25_results[:DISPLAY_K]],
                    "scores":   [r["score"]  for r in bm25_results[:DISPLAY_K]]},
            metadata={"latency_ms": lat},
        )
    latencies["retrieval-bm25"] = lat
    print(f"    retrieval-bm25:          {lat:7.1f} ms  → {[r['doc_id'] for r in bm25_results[:3]]}")

    # ── Span 5: RRF Fusion ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with langfuse.start_as_current_observation(
        trace_context=trace_ctx,
        name="rrf-fusion",
        as_type="span",
        input={"dense_top_k": TOP_K, "bm25_top_k": TOP_K, "rrf_k": RRF_K},
    ) as span:
        from collections import defaultdict as _dd
        rrf_scores = _dd(float)
        for item in dense_results:
            rrf_scores[item["doc_id"]] += 1.0 / (RRF_K + item["rank"])
        for item in bm25_results:
            rrf_scores[item["doc_id"]] += 1.0 / (RRF_K + item["rank"])
        rrf_results = [
            {"doc_id": doc_id, "score": round(score, 6), "rank": i + 1}
            for i, (doc_id, score) in enumerate(
                sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            )
        ]
        lat = ms_since(t0)
        span.update(
            output={"top5": [{"doc_id": r["doc_id"], "score": r["score"]}
                             for r in rrf_results[:TOP_K]]},
            metadata={"latency_ms": lat},
        )
    latencies["rrf-fusion"] = lat
    print(f"    rrf-fusion:              {lat:7.1f} ms  → {[r['doc_id'] for r in rrf_results[:3]]}")

    # ── Span 6: Cross-Encoder Re-ranking ─────────────────────────────────────
    t0 = time.perf_counter()
    with langfuse.start_as_current_observation(
        trace_context=trace_ctx,
        name="cross-encoder-rerank",
        as_type="span",
        input={"query": query_text,
               "candidates": [r["doc_id"] for r in rrf_results[:TOP_K]],
               "model": "ms-marco-MiniLM-L-6-v2"},
    ) as span:
        ce_candidates = rrf_results[:TOP_K]
        pairs  = [(query_text, DOC_CONTENT[r["doc_id"]]) for r in ce_candidates]
        scores = reranker.predict(pairs).tolist()
        reranked = sorted(
            [
                {**r,
                 "rrf_rank": r["rank"],
                 "ce_score": round(float(s), 4)}
                for r, s in zip(ce_candidates, scores)
            ],
            key=lambda x: x["ce_score"],
            reverse=True,
        )
        for new_rank, c in enumerate(reranked, start=1):
            c["ce_rank"] = new_rank
        lat = ms_since(t0)
        span.update(
            output={"final_ranking": [
                {"doc_id": c["doc_id"], "ce_score": c["ce_score"], "ce_rank": c["ce_rank"]}
                for c in reranked
            ]},
            metadata={"latency_ms": lat},
        )
    latencies["cross-encoder-rerank"] = lat
    print(f"    cross-encoder-rerank:    {lat:7.1f} ms  → {[c['doc_id'] for c in reranked[:3]]}")

    # ── Span 7: Synthesis (Gemini Flash) ─────────────────────────────────────
    #
    # WHY SYNTHESIS IS THE HIGHEST-RISK SPAN
    # ----------------------------------------
    # The synthesis step is where retrieval quality translates (or fails to
    # translate) into answer quality. If the wrong documents are in the top-2
    # context, the model will either hallucinate or correctly say "I don't
    # know." Logging the full prompt input and the response output in this
    # span is essential for debugging answer quality issues. In Langfuse, the
    # "generation" span type automatically links to the prompt template version
    # if you use Langfuse's prompt management system, enabling you to A/B test
    # prompt changes against production traffic.
    #
    # TOKEN COUNTING
    # ---------------
    # Gemini returns usage metadata in the response object:
    #   response.usage_metadata.prompt_token_count    (input tokens)
    #   response.usage_metadata.candidates_token_count (output tokens)
    # These are logged to Langfuse's usage_details field, which powers the
    # cost dashboard. At $0.075/1M input tokens (Gemini Flash pricing), a
    # 400-token prompt costs $0.00003 — negligible per query, but at 1M
    # queries/day it becomes $30/day just in input costs.

    synthesis_response = None
    t0 = time.perf_counter()

    if gemini is None:
        latencies["synthesis"] = 0.0
        print(f"    synthesis:               SKIPPED (no GOOGLE_API_KEY)")
    else:
        top2 = reranked[:2]
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            doc1_title   = DOC_TITLE.get(top2[0]["doc_id"], ""),
            doc1_content = DOC_CONTENT.get(top2[0]["doc_id"], ""),
            doc2_title   = DOC_TITLE.get(top2[1]["doc_id"], "") if len(top2) > 1 else "",
            doc2_content = DOC_CONTENT.get(top2[1]["doc_id"], "") if len(top2) > 1 else "",
            query        = query_text,
        )
        with langfuse.start_as_current_observation(
            trace_context=trace_ctx,
            name="synthesis",
            as_type="generation",
            model=GEMINI_MODEL,
            input={"prompt": prompt,
                   "context_docs": [c["doc_id"] for c in top2]},
        ) as span:
            response = gemini.generate_content(prompt)
            synthesis_response = response.text
            lat = ms_since(t0)

            usage = response.usage_metadata
            span.update(
                output={"response": synthesis_response},
                usage_details={
                    "input":  usage.prompt_token_count,
                    "output": usage.candidates_token_count,
                    "total":  usage.total_token_count,
                },
                metadata={"latency_ms": lat, "model": GEMINI_MODEL},
            )
        latencies["synthesis"] = lat
        print(f"    synthesis:               {lat:7.1f} ms  → {usage.total_token_count} tokens")

    latencies["_total"] = ms_since(pipeline_t0)

    # Update trace-level output
    langfuse.set_current_trace_io(
        output={
            "final_doc_ranking": [c["doc_id"] for c in reranked[:3]],
            "expected_doc":      query["expected_doc"],
            "hit":               query["expected_doc"] in [c["doc_id"] for c in reranked[:3]],
            "synthesis":         synthesis_response,
        }
    )

    return latencies


# =============================================================================
# SECTION 5 — LATENCY REPORT
# =============================================================================
#
# WHY LATENCY ANALYSIS DRIVES PRODUCTION ARCHITECTURE DECISIONS
# --------------------------------------------------------------
# The latency report answers the question every engineering manager asks:
# "Where does the time go?" It maps directly to infrastructure decisions:
#
#   If synthesis (LLM call) dominates:
#     → Evaluate a faster/cheaper model (Flash over Pro, Haiku over Sonnet)
#     → Cache responses for repeated queries (semantic caching with Redis)
#     → Use streaming responses to improve perceived latency even if total
#       time is the same
#
#   If cross-encoder re-ranking dominates:
#     → Reduce candidates from top-10 to top-5
#     → Switch to a smaller cross-encoder (L-6 instead of L-12)
#     → Apply selective re-ranking: only re-rank when intent confidence
#       is below 0.85 (high confidence queries don't need re-ranking)
#     → Move to Cohere Rerank API (offloads GPU cost, predictable latency)
#
#   If embedding dominates:
#     → Pre-warm the model (avoid cold-start penalty)
#     → Batch queries when processing logs or offline evaluation
#     → Switch to an API-based embedding service for consistent latency
#
#   If BM25 or retrieval dominates:
#     → Unlikely — BM25 and ChromaDB queries on 20 docs take <5ms each
#     → At scale (1M+ docs), this would indicate index fragmentation or
#       missing approximate nearest-neighbour tuning (HNSW ef parameter)
#
# TARGET LATENCY BUDGET (1.5 second total)
# -----------------------------------------
# A 1.5s end-to-end target is realistic for a production support bot:
#   Intent classification : ~50ms  (batched on GPU, or cached for top intents)
#   Embedding             : ~20ms  (MiniLM on GPU)
#   Dense retrieval       : ~5ms   (ChromaDB in-process)
#   BM25 retrieval        : ~2ms
#   RRF fusion            : <1ms
#   Cross-encoder rerank  : ~30ms  (MiniLM-L6 on GPU, top-10)
#   Synthesis (Gemini Flash): ~400ms (P50); ~800ms (P95)
#   Total P50             : ~510ms ✓
#   Total P95             : ~910ms ✓
# The bottleneck on CPU (as in this experiment) will be the classifier and
# cross-encoder, each taking 500ms–2s without GPU acceleration.

def compute_latency_report(all_latencies: list) -> dict:
    span_names = [
        "intent-classification",
        "embedding",
        "retrieval-dense",
        "retrieval-bm25",
        "rrf-fusion",
        "cross-encoder-rerank",
        "synthesis",
    ]

    stats = {}
    total_pipeline_time = sum(l.get("_total", 0) for l in all_latencies)

    for span in span_names:
        values = [l[span] for l in all_latencies if span in l and l[span] > 0]
        if not values:
            continue
        span_total = sum(values)
        stats[span] = {
            "min_ms":    round(min(values), 1),
            "max_ms":    round(max(values), 1),
            "avg_ms":    round(span_total / len(values), 1),
            "total_ms":  round(span_total, 1),
            "variance":  round(max(values) - min(values), 1),
            "pct_total": round(span_total / total_pipeline_time * 100, 1) if total_pipeline_time > 0 else 0,
        }

    return stats


def print_latency_report(stats: dict):
    print("=" * 80)
    print("  LATENCY BREAKDOWN (across all 10 queries)")
    print("=" * 80)
    print()

    rows = [
        [
            span,
            f"{s['min_ms']:.1f}",
            f"{s['max_ms']:.1f}",
            f"{s['avg_ms']:.1f}",
            f"{s['pct_total']:.1f}%",
        ]
        for span, s in stats.items()
    ]
    print(tabulate(
        rows,
        headers=["Span", "Min ms", "Max ms", "Avg ms", "% of Total"],
        tablefmt="rounded_outline",
        colalign=("left", "right", "right", "right", "right"),
    ))
    print()

    # Most time-consuming span
    most_time = max(stats.items(), key=lambda x: x[1]["total_ms"])
    print(f"  Span consuming most total time: {most_time[0]}")
    print(f"    Total across 10 queries: {most_time[1]['total_ms']:.1f} ms")
    print(f"    Share of pipeline: {most_time[1]['pct_total']:.1f}%")
    print()

    # Highest variance span
    most_variance = max(stats.items(), key=lambda x: x[1]["variance"])
    print(f"  Span with highest variance (max - min): {most_variance[0]}")
    print(f"    Range: {most_variance[1]['min_ms']:.1f} ms → {most_variance[1]['max_ms']:.1f} ms")
    print(f"    Variance: {most_variance[1]['variance']:.1f} ms")
    print()

    # Optimization target for 1.5s target
    print("  Optimisation target for 1.5s total pipeline budget:")
    print("  " + "─" * 60)
    target = most_time[0]
    advice = {
        "synthesis": (
            f"  '{target}' dominates because every Gemini API call is a network round-trip.\n"
            "  To hit 1.5s: switch to Gemini Flash (already fastest), enable streaming,\n"
            "  or implement semantic response caching (Redis + cosine similarity on query\n"
            "  embeddings to serve cached answers for near-duplicate queries)."
        ),
        "cross-encoder-rerank": (
            f"  '{target}' dominates because it runs a full transformer forward pass per\n"
            "  candidate without GPU. To hit 1.5s: deploy on GPU (10× speedup), reduce\n"
            "  candidates from 5 to 3, or apply selective re-ranking (skip when intent\n"
            "  confidence > 0.90 — those queries don't need re-ranking)."
        ),
        "intent-classification": (
            f"  '{target}' dominates because BART-large is a 400M-parameter model on CPU.\n"
            "  To hit 1.5s: switch to a fine-tuned bert-base classifier (6× faster),\n"
            "  use DeBERTa-small for zero-shot (3× faster), or cache top-1000 frequent\n"
            "  queries with their pre-computed intent labels."
        ),
        "embedding": (
            f"  '{target}' dominates — likely a cold-start issue on first query.\n"
            "  To hit 1.5s: move to GPU (20× speedup), or use an API-based embedding\n"
            "  service (OpenAI, Cohere) for consistent low-latency without GPU management."
        ),
    }
    print(advice.get(target,
        f"  '{target}' dominates. Profile with line-level timing to find the hot path."))
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    langfuse = setup_langfuse()
    clf, embedder, reranker, gemini = load_all_models()
    collection, bm25 = build_indexes(embedder)

    print("=" * 80)
    print("  RUNNING INSTRUMENTED PIPELINE FOR ALL 10 QUERIES")
    print("=" * 80)
    print()
    print("  Each span's latency is printed as it completes.")
    print("  Traces will appear in Langfuse within ~5–10 seconds.")
    print()

    all_latencies = []

    for q in QUERIES:
        print(f"  ── {q['id']} [{q['query_type']}] ──────────────────────────────")
        print(f"  {q['text']}")
        lats = run_instrumented_query(
            query=q,
            langfuse=langfuse,
            clf=clf,
            embedder=embedder,
            collection=collection,
            bm25=bm25,
            reranker=reranker,
            gemini=gemini,
        )
        print(f"    ── total: {lats['_total']:.1f} ms")
        print()
        all_latencies.append(lats)

    # Flush all buffered spans to Langfuse before computing report
    print("  Flushing traces to Langfuse...")
    langfuse.flush()
    print("  Done. Check https://cloud.langfuse.com for your traces.")
    print()

    stats = compute_latency_report(all_latencies)
    print_latency_report(stats)

    report = {
        "model_config": {
            "classifier":    "facebook/bart-large-mnli",
            "embedder":      "all-MiniLM-L6-v2",
            "reranker":      "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "synthesis_llm": GEMINI_MODEL,
        },
        "per_query_latencies": [
            {"query_id": q["id"], **lats}
            for q, lats in zip(QUERIES, all_latencies)
        ],
        "span_stats": stats,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Latency report saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
