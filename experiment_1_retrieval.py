"""
Experiment 1: Dense vs BM25 vs RRF Hybrid Retrieval Comparison
===============================================================

WHY THIS EXPERIMENT EXISTS IN PRODUCTION
-----------------------------------------
Every production RAG system must answer one question before anything else:
"How do I find the right documents fast enough to put in front of the LLM?"
The retrieval stage is the ceiling on answer quality — if the correct document
is never retrieved, the LLM cannot generate a correct answer no matter how
capable it is. This experiment benchmarks the two dominant retrieval paradigms
(sparse keyword and dense semantic) and their fusion, so engineers can make an
evidence-based choice for their specific corpus and query distribution before
committing to infrastructure. Skipping this evaluation is one of the most
common mistakes in RAG projects — teams pick dense retrieval because it sounds
modern, then discover their domain is full of product codes and serial numbers
that BM25 handles better.

Run:
    python experiment_1_retrieval.py

Outputs:
    - Per-query comparison tables printed to stdout
    - results/experiment_1_results.json
"""

import json
import os
import math
import textwrap
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
from tabulate import tabulate

from corpus import CORPUS
from queries import QUERIES

TOP_K = 5          # candidates retrieved per method before fusion
DISPLAY_K = 3      # columns in the comparison table
RRF_K = 60         # standard RRF constant (see reciprocal_rank_fusion below)

RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "experiment_1_results.json")


# =============================================================================
# SECTION 1 — DENSE RETRIEVAL  (sentence-transformers + ChromaDB)
# =============================================================================
#
# WHY IN PRODUCTION
# -----------------
# Dense retrieval solves the vocabulary mismatch problem. A customer who types
# "my phone got wet" and a KB article that says "liquid immersion damage" share
# zero keywords, so any keyword-based system returns nothing. Dense retrieval
# maps both texts into the same high-dimensional vector space, where semantically
# similar phrases cluster together regardless of surface wording. This is
# critical for consumer-facing support bots, where users describe problems in
# their own words rather than the terminology of the knowledge base.
#
# MATHEMATICS
# -----------
# A transformer-based bi-encoder (all-MiniLM-L6-v2 here) passes each text
# through the same BERT-like network and applies mean pooling over the final
# hidden states to produce a fixed-size dense vector (384 dimensions for
# MiniLM). Relevance between a query q and document d is measured by cosine
# similarity:
#
#   sim(q, d) = (q · d) / (||q|| × ||d||)
#
# Values range from -1 to 1; in practice well-trained models yield 0.3–0.9 for
# relevant pairs. ChromaDB stores these vectors and answers nearest-neighbour
# queries using the HNSW (Hierarchical Navigable Small World) graph algorithm,
# which finds approximate nearest neighbours in O(log n) rather than O(n),
# making it usable at scale.
#
# ALTERNATIVES & TRADEOFFS
# -------------------------
# • OpenAI text-embedding-3-small / text-embedding-3-large
#     Pro: state-of-the-art quality, no GPU needed, easy API.
#     Con: cost per token adds up at scale; data leaves your environment
#          (compliance risk for PII-heavy corpora); latency depends on
#          OpenAI's availability.
#
# • Cohere Embed v3 (with Cohere Rerank)
#     Pro: self-managed deployment option via Azure/AWS; multilingual;
#          includes input_type parameter to optimise query vs. doc embeddings
#          separately, which measurably improves retrieval quality.
#     Con: more expensive than open-source models; vendor lock-in; the
#          separate query/doc embedding types add complexity to your pipeline.
#
# • Google Vertex AI Embeddings (text-embedding-gecko)
#     Pro: tightly integrated with GCP, low latency within GCP infra.
#     Con: GCP-only; less flexible for on-prem/hybrid deployments.
#
# • Local open-source: all-MiniLM-L6-v2 (this experiment), BGE-M3, E5-large
#     Pro: free, runs on CPU, data stays in your environment, no per-call cost.
#     Con: quality below frontier API models; requires GPU for production
#          throughput; you own the hosting and upgrade cycle.
#
# • Vector store alternatives to ChromaDB:
#     - Pinecone: managed, production-grade, expensive at scale.
#     - Weaviate: open-source, supports hybrid search natively.
#     - pgvector: adds vector column to Postgres; great if you already use PG
#       and want to avoid a separate service.
#     - Qdrant: Rust-based, fast, good filtering support.
#     ChromaDB is ideal for experiments and small-scale production (<1M docs).
#     Move to Pinecone/Weaviate/pgvector once you need filtering, replication,
#     or >1M documents.

def build_dense_index(corpus):
    print("=== Building dense index (all-MiniLM-L6-v2) ===")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    # Delete and recreate for a clean run each time
    try:
        client.delete_collection("rag_corpus")
    except Exception:
        pass
    collection = client.create_collection(
        "rag_corpus",
        metadata={"hnsw:space": "cosine"},
    )

    doc_ids   = [doc["id"]      for doc in corpus]
    doc_texts = [doc["content"] for doc in corpus]
    embeddings = model.encode(doc_texts, show_progress_bar=True).tolist()

    collection.add(
        ids=doc_ids,
        embeddings=embeddings,
        documents=doc_texts,
        metadatas=[{"title": doc["title"], "category": doc["category"]} for doc in corpus],
    )
    print(f"  Indexed {len(doc_ids)} documents into ChromaDB.\n")
    return model, collection


def dense_query(model, collection, query_text, top_k=TOP_K):
    q_emb = model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["distances", "metadatas"],
    )
    # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
    # Convert to similarity so higher = better, consistent with intuition.
    doc_ids   = results["ids"][0]
    distances = results["distances"][0]
    return [
        {"doc_id": did, "score": round(1 - dist, 4), "rank": i + 1}
        for i, (did, dist) in enumerate(zip(doc_ids, distances))
    ]


# =============================================================================
# SECTION 2 — BM25 SPARSE RETRIEVAL
# =============================================================================
#
# WHY IN PRODUCTION
# -----------------
# BM25 remains the baseline every retrieval system is measured against because
# it is fast (no GPU), explainable (you can see exactly which terms drove the
# score), and surprisingly hard to beat on domains with precise terminology:
# legal citations, medical codes, product SKUs, policy document numbers. In
# telecom and insurance — this experiment's domain — agents often search by
# exact claim numbers or policy codes. For these queries, a dense model that
# spreads attention across the whole sentence can actually dilute the signal
# from a rare, high-value token.
#
# MATHEMATICS
# -----------
# BM25 (Best Match 25) is a probabilistic retrieval function from the BM family.
# For a query Q with terms q1 … qn and a document D, the score is:
#
#   BM25(D, Q) = Σ IDF(qi) × [ f(qi,D) × (k1+1) ] / [ f(qi,D) + k1×(1 - b + b×|D|/avgdl) ]
#
# Where:
#   f(qi, D)  = term frequency of qi in D
#   |D|       = document length in tokens
#   avgdl     = average document length in the corpus
#   k1 ≈ 1.5  = term frequency saturation (diminishing returns on repetition)
#   b  ≈ 0.75 = length normalisation strength
#   IDF(qi)   = log( (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 )
#               where N = corpus size, n(qi) = docs containing qi
#
# The IDF component makes rare tokens (e.g. "INS-POL-2024") score very highly —
# this is the key property that makes BM25 excellent for exact-match queries.
#
# ALTERNATIVES & TRADEOFFS
# -------------------------
# • Elasticsearch / OpenSearch BM25
#     Pro: production-grade, distributed, supports complex filters and
#          aggregations alongside BM25; widely understood by ops teams.
#     Con: heavy infrastructure; overkill for <100k documents; Elasticsearch
#          recently changed licensing (SSPL), so OpenSearch is the open fork.
#
# • Elasticsearch with ELSER (Elastic Learned Sparse EncoderR)
#     Pro: sparse learned representations that understand synonyms, unlike
#          classical BM25; native Elasticsearch integration.
#     Con: proprietary; requires Elastic Cloud or self-managed Elastic stack.
#
# • BM25s (Python, Rust-backed)
#     Pro: 10–100× faster than rank-bm25 for large corpora; drop-in interface.
#     Con: less adoption, fewer examples online; rank-bm25 is fine up to ~1M docs.
#
# • TF-IDF (scikit-learn TfidfVectorizer)
#     Pro: even simpler, familiar to data scientists.
#     Con: no length normalisation or saturation — BM25 is strictly better
#          on most benchmarks; use TF-IDF only for quick prototypes.
#
# • Hybrid native in Weaviate / Qdrant
#     Both support BM25 + dense as first-class features, so you don't need
#     a separate BM25 library at all if you're already using those stores.

def build_bm25_index(corpus):
    print("=== Building BM25 index ===")
    tokenized = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    print(f"  Indexed {len(corpus)} documents.\n")
    return bm25


def bm25_query(bm25, corpus, query_text, top_k=TOP_K):
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"doc_id": corpus[idx]["id"], "score": round(score, 4), "rank": i + 1}
        for i, (idx, score) in enumerate(ranked)
    ]


# =============================================================================
# SECTION 3 — RRF HYBRID FUSION
# =============================================================================
#
# WHY IN PRODUCTION
# -----------------
# No retriever is universally better. Dense wins on paraphrase queries; BM25
# wins on exact-match queries. A production support bot gets both kinds of
# traffic simultaneously. Rather than choosing one approach and accepting its
# failure modes, RRF lets you run both in parallel and merge their ranked lists
# into one result. It is the single most reliable improvement you can make to
# a retrieval pipeline with minimal added complexity — no training required,
# no labelled data needed, no hyperparameter sensitivity. Major AI search
# providers (Google, Cohere, Weaviate) all expose hybrid search with RRF or
# a variant as the recommended default.
#
# MATHEMATICS
# -----------
# Reciprocal Rank Fusion (Cormack et al., 2009) combines ranked lists by
# assigning each document a fused score:
#
#   RRF_score(d) = Σ_{r ∈ rankers} 1 / (k + rank_r(d))
#
# Where:
#   rank_r(d) = position of document d in ranker r's list (1-indexed)
#   k = 60     = a constant chosen empirically to prevent very high-ranked docs
#                from dominating; documents ranked 1st contribute 1/61 ≈ 0.016
#                rather than 1/1, smoothing the advantage of top position.
#
# Documents not present in a ranker's top-k list contribute 0 to that ranker's
# term. The final ranked list is sorted by descending RRF score. The elegance
# of RRF is that it never requires score normalisation — BM25 scores are
# unbounded floats while cosine similarities are 0–1, yet RRF is immune to
# this mismatch because it only uses *ranks*, not raw scores.
#
# ALTERNATIVES & TRADEOFFS
# -------------------------
# • Weighted RRF (dense_weight × 1/(k+rank) + bm25_weight × 1/(k+rank))
#     Pro: lets you favour semantic or keyword retrieval based on query type.
#     Con: requires labelled eval data to tune weights; without it you're
#          guessing and may make things worse than equal weighting.
#
# • Linear score combination (normalise each ranker's scores to [0,1] then add)
#     Pro: preserves magnitude information (a score of 0.95 vs 0.51 matters).
#     Con: BM25 scores vary wildly by corpus; normalisation is fragile across
#          query types; much more sensitive to outliers than RRF.
#
# • Cohere Rerank API as a fusion layer
#     Pro: uses a cross-encoder to re-score the merged candidate list; often
#          better accuracy than RRF because it understands semantics.
#     Con: adds ~200–500ms latency and per-call cost; becomes the new
#          bottleneck at high query volume.
#
# • Native hybrid in vector stores (Weaviate, Qdrant, OpenSearch)
#     Pro: single query to one service; operationally simpler.
#     Con: less control over the fusion logic; harder to A/B test individual
#          retriever contributions.
#
# • Learn-to-Rank (LambdaMART, LightGBM Ranker)
#     Pro: can learn optimal fusion weights per query type from click data.
#     Con: requires thousands of labelled query-document pairs to train;
#          adds a serving dependency on a feature store and ML model.

def reciprocal_rank_fusion(dense_results, bm25_results, k=RRF_K):
    rrf_scores = defaultdict(float)

    for item in dense_results:
        rrf_scores[item["doc_id"]] += 1.0 / (k + item["rank"])

    for item in bm25_results:
        rrf_scores[item["doc_id"]] += 1.0 / (k + item["rank"])

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"doc_id": doc_id, "score": round(score, 6), "rank": i + 1}
        for i, (doc_id, score) in enumerate(ranked)
    ]


# =============================================================================
# SECTION 4 — OUTPUT & EVALUATION HELPERS
# =============================================================================
#
# WHY OFFLINE EVALUATION MATTERS IN PRODUCTION
# ---------------------------------------------
# Before deploying a retrieval change you need a number that tells you whether
# it is better or worse. The standard metric used here is Hit Rate @ k (HR@k):
# "What fraction of queries had the correct document appear in the top-k
# results?" It is simple to compute, easy to explain to non-engineers, and
# correlates well with end-to-end answer quality in RAG systems.
# More sophisticated metrics used in production include:
#   - MRR (Mean Reciprocal Rank): rewards finding the right doc at rank 1 more
#     than rank 3; useful when you only pass rank-1 to the LLM.
#   - NDCG@k (Normalised Discounted Cumulative Gain): handles graded relevance
#     (a "perfect" doc vs. a "partially relevant" doc); standard in IR research.
#   - Recall@k: fraction of ALL relevant docs retrieved; matters when there are
#     multiple valid answers (e.g. insurance covers both DI-001 and DI-007).
# For this experiment HR@3 is sufficient because we have a single ground-truth
# document per query.

def top_ids(results, k=DISPLAY_K):
    return [r["doc_id"] for r in results[:k]]


def hit(results, expected, k=DISPLAY_K):
    """True if expected doc appears in top-k."""
    return expected in top_ids(results, k)


def format_cell(r, expected):
    marker = " ✓" if r["doc_id"] == expected else "  "
    return f"#{r['rank']} {r['doc_id']} ({r['score']:.4f}){marker}"


def print_query_block(q, dense_res, bm25_res, rrf_res):
    print("─" * 80)
    print(f"  {q['id']} [{q['query_type']}] [{q['difficulty']}]")
    print(f"  Query   : {q['text']}")
    print(f"  Expected: {q['expected_doc']}")
    print()

    expected = q["expected_doc"]
    # One row per rank position — avoids multi-line cell rendering issues in Colab
    rows = [
        [
            format_cell(dense_res[i], expected),
            format_cell(bm25_res[i],  expected),
            format_cell(rrf_res[i],   expected),
        ]
        for i in range(DISPLAY_K)
    ]
    print(tabulate(rows, headers=["Dense Top-3", "BM25 Top-3", "RRF Top-3"], tablefmt="rounded_outline"))

    dense_hit = hit(dense_res, expected)
    bm25_hit  = hit(bm25_res,  expected)
    rrf_hit   = hit(rrf_res,   expected)

    verdict = []
    if dense_hit and not bm25_hit:
        verdict.append("Dense WINS over BM25")
    elif bm25_hit and not dense_hit:
        verdict.append("BM25 WINS over Dense")
    elif dense_hit and bm25_hit:
        verdict.append("Both Dense and BM25 correct")
    else:
        verdict.append("Both Dense and BM25 MISS")

    verdict.append("RRF correct ✓" if rrf_hit else "RRF also misses ✗")
    print("  " + " | ".join(verdict))
    print()


def print_summary(query_results):
    print("=" * 80)
    print("  SUMMARY ANALYSIS")
    print("=" * 80)

    dense_only_wins = []
    bm25_only_wins  = []
    both_correct    = []
    both_miss       = []
    rrf_matches_best = 0
    rrf_beats_both   = 0

    for qr in query_results:
        q        = qr["query"]
        d_hit    = qr["dense_hit"]
        b_hit    = qr["bm25_hit"]
        r_hit    = qr["rrf_hit"]
        best_hit = d_hit or b_hit

        if d_hit and not b_hit:
            dense_only_wins.append(q["id"])
        elif b_hit and not d_hit:
            bm25_only_wins.append(q["id"])
        elif d_hit and b_hit:
            both_correct.append(q["id"])
        else:
            both_miss.append(q["id"])

        if r_hit and best_hit:
            rrf_matches_best += 1
        if r_hit and not best_hit:
            rrf_beats_both += 1

    total = len(query_results)

    print(f"\n  Queries where Dense outperformed BM25  : {len(dense_only_wins)}/{total}")
    for qid in dense_only_wins:
        qr = next(x for x in query_results if x["query"]["id"] == qid)
        print(f"    {qid} [{qr['query']['query_type']}] — {qr['query']['text'][:65]}")

    print(f"\n  Queries where BM25 outperformed Dense  : {len(bm25_only_wins)}/{total}")
    for qid in bm25_only_wins:
        qr = next(x for x in query_results if x["query"]["id"] == qid)
        print(f"    {qid} [{qr['query']['query_type']}] — {qr['query']['text'][:65]}")

    print(f"\n  Both methods correct                    : {len(both_correct)}/{total}")
    for qid in both_correct:
        print(f"    {qid}")

    print(f"\n  Both methods missed                     : {len(both_miss)}/{total}")
    for qid in both_miss:
        qr = next(x for x in query_results if x["query"]["id"] == qid)
        print(f"    {qid} [{qr['query']['query_type']}] — {qr['query']['text'][:65]}")

    rrf_correct = sum(1 for qr in query_results if qr["rrf_hit"])
    print(f"\n  RRF matched or beat the better method  : {rrf_matches_best}/{total}")
    print(f"  RRF rescued a query both others missed : {rrf_beats_both}/{total}")
    print(f"  RRF total correct (top-{DISPLAY_K})            : {rrf_correct}/{total}")

    print()
    print("  Key Takeaways")
    print("  " + "─" * 60)

    if len(dense_only_wins) > len(bm25_only_wins):
        print("  • Dense retrieval was stronger overall on this query set,")
        print("    especially on paraphrase/semantic queries.")
    elif len(bm25_only_wins) > len(dense_only_wins):
        print("  • BM25 was stronger overall — exact token matches dominated.")
    else:
        print("  • Dense and BM25 were evenly matched on this query set.")

    if rrf_matches_best >= math.ceil(total * 0.7):
        print("  • RRF reliably matched the stronger individual method (>=70%).")
        print("    In production, hybrid is the safe default.")
    else:
        print("  • RRF did not consistently outperform individual methods here.")
        print("    Consider tuning RRF k or the weighting between methods.")

    if both_miss:
        print(f"  • {len(both_miss)} queries defeated all methods — these are the")
        print("    context-dependent queries. They require query rewriting with")
        print("    conversation history before retrieval (see experiment 3).")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, collection = build_dense_index(CORPUS)
    bm25              = build_bm25_index(CORPUS)

    print("=" * 80)
    print("  PER-QUERY COMPARISON: Dense | BM25 | RRF")
    print("=" * 80)
    print()

    query_results = []

    for q in QUERIES:
        dense_res = dense_query(model, collection, q["text"])
        bm25_res  = bm25_query(bm25, CORPUS, q["text"])
        rrf_res   = reciprocal_rank_fusion(dense_res, bm25_res)

        expected = q["expected_doc"]
        d_hit    = hit(dense_res, expected)
        b_hit    = hit(bm25_res,  expected)
        r_hit    = hit(rrf_res,   expected)

        print_query_block(q, dense_res, bm25_res, rrf_res)

        query_results.append({
            "query":         q,
            "dense_results": dense_res,
            "bm25_results":  bm25_res,
            "rrf_results":   rrf_res[:TOP_K],
            "dense_hit":     d_hit,
            "bm25_hit":      b_hit,
            "rrf_hit":       r_hit,
        })

    print_summary(query_results)

    with open(RESULTS_FILE, "w") as f:
        json.dump(
            [
                {
                    "query_id":     qr["query"]["id"],
                    "query_text":   qr["query"]["text"],
                    "query_type":   qr["query"]["query_type"],
                    "difficulty":   qr["query"]["difficulty"],
                    "expected_doc": qr["query"]["expected_doc"],
                    "dense_top5":   qr["dense_results"],
                    "bm25_top5":    qr["bm25_results"],
                    "rrf_top5":     qr["rrf_results"],
                    "dense_hit":    qr["dense_hit"],
                    "bm25_hit":     qr["bm25_hit"],
                    "rrf_hit":      qr["rrf_hit"],
                }
                for qr in query_results
            ],
            f,
            indent=2,
        )
    print(f"  Results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
