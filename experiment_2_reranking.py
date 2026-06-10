"""
Experiment 2: Cross-Encoder Re-ranking of RRF Candidates
=========================================================

WHY THIS EXPERIMENT EXISTS IN PRODUCTION
-----------------------------------------
Retrieval and ranking are two separate problems with different accuracy-latency
tradeoffs. The first stage (BM25, dense, RRF) must be fast enough to scan
millions of documents in milliseconds, so it uses approximations: bag-of-words
counts or pre-computed embedding lookups. These approximations are good at
finding a plausible candidate set but imprecise at ranking within it. The
second stage — re-ranking — runs a much more expensive model over only the
top-5 to top-50 candidates returned by stage one. It can afford to be slow
because it processes a tiny set. This two-stage architecture is the standard
pattern at companies like Google (for search), Cohere (their Rerank API),
and any enterprise RAG system where answer quality is a hard requirement.
Without a re-ranker, you are trusting a fast approximate model to give the
LLM its context in the right order — and the LLM is sensitive to position.

Run:
    python experiment_2_reranking.py
    (requires results/experiment_1_results.json — run experiment 1 first)

Outputs:
    - Per-query displacement tables printed to stdout
    - Summary analysis printed to stdout
    - results/experiment_2_results.json
"""

import json
import os

from sentence_transformers import CrossEncoder
from tabulate import tabulate

from corpus import CORPUS

RESULTS_DIR = "results"
INPUT_FILE  = os.path.join(RESULTS_DIR, "experiment_1_results.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "experiment_2_results.json")

DOC_CONTENT = {doc["id"]: doc["content"] for doc in CORPUS}
DOC_TITLE   = {doc["id"]: doc["title"]   for doc in CORPUS}


# =============================================================================
# SECTION 1 — LOAD STAGE-1 CANDIDATES
# =============================================================================
#
# WHY STAGE SEPARATION IN PRODUCTION
# ------------------------------------
# Persisting stage-1 results to JSON and loading them here models the actual
# production architecture: retrieval and re-ranking are separate services,
# often deployed on different hardware (retrieval on CPU-optimised instances
# for throughput, re-ranking on GPU for quality). Decoupling them lets you
# upgrade the re-ranker without touching retrieval infrastructure and vice
# versa. It also enables offline batch re-ranking experiments without
# re-running the expensive embedding step every time.

def load_experiment_1(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path}.\n"
            "Run experiment_1_retrieval.py first to generate this file."
        )
    with open(path) as f:
        return json.load(f)


# =============================================================================
# SECTION 2 — CROSS-ENCODER RE-RANKING
# =============================================================================
#
# WHY IN PRODUCTION
# -----------------
# Re-ranking with a cross-encoder measurably improves answer quality in RAG
# because the LLM's answer is heavily influenced by what appears at the top of
# its context window. Studies on "lost in the middle" (Liu et al., 2023) show
# that LLMs preferentially use information from the beginning and end of their
# context, largely ignoring the middle. A re-ranker ensures the most relevant
# passage is at position 1, not buried at position 4. In production, Cohere
# Rerank, Amazon Bedrock Rerank, and Azure AI Search all expose re-ranking as
# a managed API because the accuracy gains justify the latency cost for most
# enterprise use cases.
#
# MATHEMATICS — WHY CROSS-ENCODERS ARE MORE ACCURATE THAN BI-ENCODERS
# ---------------------------------------------------------------------
# A bi-encoder (used in stage 1) encodes query and document independently:
#
#   e_q = Encoder(query)          → fixed vector
#   e_d = Encoder(document)       → fixed vector (pre-computed offline)
#   score = cosine(e_q, e_d)
#
# The query and document tokens never "see" each other during encoding.
# A cross-encoder concatenates them as a single input:
#
#   score = Encoder([query] [SEP] [document])  → single scalar
#
# With full self-attention across all tokens in both texts simultaneously,
# every query token can attend to every document token in the same forward
# pass. This captures fine-grained relevance signals that are invisible to
# independent encoders — for example, the exact phrase "non-return fee" in
# the document matching "fee if I don't send back" in the query.
#
# The cost: you cannot pre-compute document representations. Every query
# requires a fresh forward pass over each (query, document) pair, making
# latency O(n × sequence_length²) vs. O(1) lookup for bi-encoders. That is
# why cross-encoders are only applied to the small candidate set (top-5 to
# top-100) rather than the full corpus.
#
# ALTERNATIVES & TRADEOFFS
# -------------------------
# • Cohere Rerank API (rerank-english-v3.0, rerank-multilingual-v3.0)
#     Pro: state-of-the-art accuracy; managed, no GPU needed; multilingual;
#          simple single API call; supported natively in LangChain/LlamaIndex.
#     Con: ~$2/1000 queries (adds up at scale); data leaves your environment
#          (PII risk); adds 200–600ms latency; vendor dependency.
#
# • Jina Reranker (jina-reranker-v2-base-multilingual)
#     Pro: open-source, 8192-token context (good for long documents);
#          strong multilingual support; can run locally for free.
#     Con: requires GPU for production throughput; less adoption than Cohere.
#
# • ms-marco-MiniLM-L-6-v2 (this experiment)
#     Pro: free, fast (6-layer MiniLM), runs on CPU, good on short passages.
#     Con: 512-token limit (truncates long documents); lower accuracy than
#          larger cross-encoders or Cohere on complex queries.
#
# • ms-marco-MiniLM-L-12-v2 / ms-marco-electra-base
#     Pro: higher accuracy than L-6 at the cost of 2× slower inference.
#     Con: still limited to 512 tokens; not multilingual.
#
# • FlashRank (lightweight cross-encoder framework)
#     Pro: optimised for CPU inference, very low latency, good for on-prem.
#     Con: smaller model selection than HuggingFace ecosystem.
#
# • LLM-as-reranker (prompt GPT-4 to rate relevance 1–10 for each candidate)
#     Pro: can understand nuanced relevance criteria; no separate model.
#     Con: extremely expensive (one LLM call per candidate), 10–50× slower
#          than a cross-encoder; not viable for latency-sensitive applications.
#
# PRODUCTION LATENCY GUIDANCE
# ---------------------------
# Typical p95 latencies for re-ranking top-10 candidates:
#   MiniLM-L-6 on CPU    : ~30–80ms
#   MiniLM-L-6 on GPU    : ~5–15ms
#   MiniLM-L-12 on GPU   : ~10–30ms
#   Cohere Rerank API     : ~150–400ms (network + model)
# Add this to your stage-1 retrieval latency to get end-to-end budget.

def build_cross_encoder():
    print("=== Loading cross-encoder (ms-marco-MiniLM-L-6-v2) ===")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  Model loaded.\n")
    return model


def rerank(cross_encoder, query_text, rrf_top10):
    # ── The cross-encoder scores each (query, document) pair jointly ──────────
    # Both texts are concatenated as "[query] [SEP] [document]" and fed through
    # the transformer in a single forward pass. Full self-attention means every
    # query token can attend to every document token, capturing precise
    # relevance signals that the bi-encoder's independent embeddings miss.
    # This is more accurate but cannot be pre-computed — each pair requires
    # its own inference call, so this only runs on the small candidate set.
    pairs  = [(query_text, DOC_CONTENT[r["doc_id"]]) for r in rrf_top10]
    scores = cross_encoder.predict(pairs).tolist()

    # Normalise the RRF rank field name (stored as "rank" in experiment 1 JSON)
    candidates = [
        {
            "doc_id":    r["doc_id"],
            "rrf_rank":  r.get("rrf_rank", r.get("rank", i + 1)),
            "rrf_score": r.get("score", 0),
            "ce_score":  round(float(s), 4),
        }
        for i, (r, s) in enumerate(zip(rrf_top10, scores))
    ]
    candidates.sort(key=lambda x: x["ce_score"], reverse=True)

    for new_rank, c in enumerate(candidates, start=1):
        c["ce_rank"] = new_rank

    return candidates


# =============================================================================
# SECTION 3 — DISPLACEMENT ANALYSIS & OUTPUT
# =============================================================================
#
# WHY MEASURE DISPLACEMENT IN PRODUCTION
# ----------------------------------------
# Displacement (how many positions a document moves between stage-1 and
# stage-2 ranking) is the key diagnostic for deciding whether re-ranking is
# worth its latency cost for a given query type.
#
#   High displacement → stage-1 had the right docs but wrong order;
#                        re-ranking earns its latency.
#   Low displacement  → stage-1 ordering was already correct;
#                        re-ranking adds latency with no accuracy benefit;
#                        consider skipping it for this query category.
#   Zero displacement → your first-stage retrieval is well-calibrated;
#                        you may not need a re-ranker at all.
#
# In a production system you would track this metric per query intent class
# and use it to decide which query paths to route through re-ranking. This
# is called selective re-ranking and reduces overall pipeline latency by
# 30–60% without sacrificing accuracy on the queries where re-ranking helps.

def displacement_symbol(rrf_rank, ce_rank):
    delta = rrf_rank - ce_rank   # positive = moved up, negative = moved down
    if delta > 0:
        return f"▲ +{delta}"
    elif delta < 0:
        return f"▼ {delta}"
    else:
        return "  ="


def print_displacement_table(q_record, reranked):
    print("─" * 80)
    print(f"  {q_record['query_id']} [{q_record['query_type']}] [{q_record['difficulty']}]")
    print(f"  Query   : {q_record['query_text']}")
    print(f"  Expected: {q_record['expected_doc']}")
    print()

    expected = q_record["expected_doc"]
    rows = []
    for c in reranked:
        marker = " ✓" if c["doc_id"] == expected else ""
        rows.append([
            f"{c['doc_id']}{marker}",
            c["rrf_rank"],
            c["ce_rank"],
            f"{c['ce_score']:.4f}",
            displacement_symbol(c["rrf_rank"], c["ce_rank"]),
        ])

    print(tabulate(
        rows,
        headers=["Doc ID", "RRF Rank", "CE Rank", "CE Score", "Movement"],
        tablefmt="rounded_outline",
        colalign=("left", "center", "center", "right", "center"),
    ))
    print()


# =============================================================================
# SECTION 4 — SUMMARY ANALYSIS
# =============================================================================
#
# WHY SUMMARY METRICS MATTER FOR PRODUCTION DECISIONS
# ----------------------------------------------------
# Individual query results tell you what happened; summary metrics tell you
# what to do. The three summary outputs map directly to production decisions:
#
#   1. "How many docs moved >2 positions?"
#      → Quantifies overall re-ranking churn. High churn means stage-1 is
#        noisy and re-ranking is adding real value. Low churn means you
#        might be able to remove the re-ranker and save the latency.
#
#   2. "Which query type had the most displacement?"
#      → Tells you which intent class benefits most from re-ranking. Use
#        this to build selective re-ranking routing: apply the cross-encoder
#        only to query types with historically high displacement.
#
#   3. "Promotion example: rank-4+ document moved to rank 1–2"
#      → The most compelling evidence that re-ranking found something the
#        first stage missed. This is the case you show to stakeholders when
#        justifying the latency budget for a re-ranker.

def compute_summary(all_query_data):
    total_displaced_docs = 0
    max_displacement     = 0
    max_displacement_qid = None
    promotion_example    = None
    per_query_displacement = {}

    for qd in all_query_data:
        qid            = qd["query_id"]
        reranked       = qd["reranked"]
        query_max_delta = 0

        for c in reranked:
            delta = abs(c["rrf_rank"] - c["ce_rank"])
            if delta > 2:
                total_displaced_docs += 1
            if delta > query_max_delta:
                query_max_delta = delta

            if (
                promotion_example is None
                and c["rrf_rank"] >= 4
                and c["ce_rank"] <= 2
            ):
                promotion_example = {
                    "query_id":   qid,
                    "query_text": qd["query_text"],
                    "doc_id":     c["doc_id"],
                    "doc_title":  DOC_TITLE[c["doc_id"]],
                    "rrf_rank":   c["rrf_rank"],
                    "ce_rank":    c["ce_rank"],
                    "ce_score":   c["ce_score"],
                }

        per_query_displacement[qid] = query_max_delta
        if query_max_delta > max_displacement:
            max_displacement     = query_max_delta
            max_displacement_qid = qid

    return {
        "total_displaced_docs":   total_displaced_docs,
        "max_displacement_qid":   max_displacement_qid,
        "max_displacement_value": max_displacement,
        "per_query_max_delta":    per_query_displacement,
        "promotion_example":      promotion_example,
    }


def print_summary(summary, all_query_data):
    print("=" * 80)
    print("  SUMMARY ANALYSIS")
    print("=" * 80)

    print(f"\n  Documents that moved more than 2 positions: {summary['total_displaced_docs']}")
    print(f"\n  Query with most re-ranking displacement: {summary['max_displacement_qid']} "
          f"(max single-doc shift = {summary['max_displacement_value']} positions)")

    rows = []
    for qd in all_query_data:
        qid   = qd["query_id"]
        delta = summary["per_query_max_delta"][qid]
        rows.append([qid, qd["query_type"], delta, "█" * delta])
    print()
    print(tabulate(rows, headers=["Query", "Type", "Max Δ", ""], tablefmt="rounded_outline",
                   colalign=("left", "left", "center", "left")))

    ex = summary["promotion_example"]
    print()
    if ex:
        print(f"  Promotion Example")
        print(f"  {'─' * 60}")
        print(f"  Query  : {ex['query_text']}")
        print(f"  Doc    : {ex['doc_id']} — \"{ex['doc_title']}\"")
        print(f"  RRF rank {ex['rrf_rank']}  →  Cross-encoder rank {ex['ce_rank']} "
              f"(score {ex['ce_score']:.4f})")
        print()
        qd = next(x for x in all_query_data if x["query_id"] == ex["query_id"])
        reasons = {
            "exact_match": (
                "The cross-encoder confirmed a precise factual match by attending "
                "simultaneously to the exact token in the query and its occurrence "
                "in the document body — a signal RRF's rank aggregation had diluted."
            ),
            "semantic": (
                "Joint attention over query and document tokens revealed a semantic "
                "alignment that the bi-encoder's independently computed embeddings "
                "had underweighted when comparing separate vectors."
            ),
            "ambiguous": (
                "Reading both texts together let the cross-encoder resolve the "
                "ambiguity and identify which category the query required — something "
                "neither BM25 nor the bi-encoder could do independently."
            ),
        }
        reason = reasons.get(
            qd["query_type"],
            "Full cross-attention between query and document tokens captured relevance "
            "signals that were invisible to retrievers processing them separately."
        )
        print(f"  Why: {reason}")
    else:
        print("  No document was promoted from rank ≥4 to rank ≤2 in this run.")
        print("  This means first-stage retrieval ordering was already high quality —")
        print("  the re-ranker is confirming rather than correcting stage-1.")

    print()
    print("  Key Takeaways")
    print("  " + "─" * 60)
    print("  • Cross-encoders re-order candidates, they do not retrieve —")
    print("    they only improve ordering of what stage-1 already found.")
    print("  • High displacement = first stage had right docs, wrong order;")
    print("    re-ranking earns its latency cost on these query types.")
    print("  • Low displacement = first stage was already well ordered;")
    print("    consider selective re-ranking (skip it for these queries).")
    print("  • The 'lost in the middle' problem makes rank-1 position critical:")
    print("    LLMs use information at the start/end of context more than middle.")
    print("  • In production: retrieve top-50 fast, re-rank top-10 precisely.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    exp1_data     = load_experiment_1(INPUT_FILE)
    cross_encoder = build_cross_encoder()

    print("=" * 80)
    print("  PER-QUERY DISPLACEMENT: RRF rank → Cross-encoder rank")
    print("=" * 80)
    print()

    all_query_data = []

    for record in exp1_data:
        rrf_candidates = record["rrf_top5"]
        reranked       = rerank(cross_encoder, record["query_text"], rrf_candidates)

        print_displacement_table(record, reranked)

        all_query_data.append({
            "query_id":    record["query_id"],
            "query_text":  record["query_text"],
            "query_type":  record["query_type"],
            "difficulty":  record["difficulty"],
            "expected_doc": record["expected_doc"],
            "reranked":    reranked,
            "rrf_hit":     record["rrf_hit"],
            "ce_hit":      any(
                c["doc_id"] == record["expected_doc"] and c["ce_rank"] <= 3
                for c in reranked
            ),
        })

    summary = compute_summary(all_query_data)
    print_summary(summary, all_query_data)

    rrf_correct = sum(1 for qd in all_query_data if qd["rrf_hit"])
    ce_correct  = sum(1 for qd in all_query_data if qd["ce_hit"])
    total       = len(all_query_data)
    print(f"  Accuracy (top-3 hit rate)")
    print(f"    RRF alone       : {rrf_correct}/{total}")
    print(f"    After re-ranking: {ce_correct}/{total}")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"summary": summary, "queries": all_query_data}, f, indent=2)
    print(f"  Results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
