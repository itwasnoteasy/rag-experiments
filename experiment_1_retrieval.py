"""
Experiment 1: Dense vs BM25 vs RRF Hybrid Retrieval Comparison

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

# ── Dependencies ──────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
from tabulate import tabulate

from corpus import CORPUS
from queries import QUERIES

TOP_K = 5          # candidates retrieved per method before fusion
DISPLAY_K = 3      # columns in the comparison table
RRF_K = 60         # standard RRF constant

RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "experiment_1_results.json")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DENSE RETRIEVAL  (sentence-transformers + ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────

def build_dense_index(corpus):
    print("=== Building dense index (all-MiniLM-L6-v2) ===")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    # fresh collection each run
    try:
        client.delete_collection("rag_corpus")
    except Exception:
        pass
    collection = client.create_collection(
        "rag_corpus",
        metadata={"hnsw:space": "cosine"},
    )

    doc_ids = [doc["id"] for doc in corpus]
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
    # ChromaDB cosine distance → similarity = 1 - distance
    doc_ids = results["ids"][0]
    distances = results["distances"][0]
    return [
        {"doc_id": did, "score": round(1 - dist, 4), "rank": i + 1}
        for i, (did, dist) in enumerate(zip(doc_ids, distances))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. BM25 SPARSE RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def build_bm25_index(corpus):
    print("=== Building BM25 index ===")
    tokenized = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    print(f"  Indexed {len(corpus)} documents.\n")
    return bm25


def bm25_query(bm25, corpus, query_text, top_k=TOP_K):
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]
    return [
        {"doc_id": corpus[idx]["id"], "score": round(score, 4), "rank": i + 1}
        for i, (idx, score) in enumerate(ranked)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. RRF HYBRID FUSION
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(dense_results, bm25_results, k=RRF_K):
    """
    RRF score = sum over methods of 1 / (k + rank)
    Documents not in a method's result list get rank = infinity (score = 0).
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. OUTPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def top_ids(results, k=DISPLAY_K):
    return [r["doc_id"] for r in results[:k]]


def hit(results, expected, k=DISPLAY_K):
    """True if expected doc appears in top-k."""
    return expected in top_ids(results, k)


def format_result_col(results, expected, k=DISPLAY_K):
    rows = []
    for r in results[:k]:
        marker = " ✓" if r["doc_id"] == expected else ""
        rows.append(f"#{r['rank']} {r['doc_id']} ({r['score']:.4f}){marker}")
    return "\n".join(rows)


def print_query_block(q, dense_res, bm25_res, rrf_res):
    print("─" * 80)
    print(f"  {q['id']} [{q['query_type']}] [{q['difficulty']}]")
    print(f"  Query   : {q['text']}")
    print(f"  Expected: {q['expected_doc']}")
    print()

    expected = q["expected_doc"]
    table = [
        [
            format_result_col(dense_res, expected),
            format_result_col(bm25_res, expected),
            format_result_col(rrf_res, expected),
        ]
    ]
    print(
        tabulate(
            table,
            headers=["Dense Top-3", "BM25 Top-3", "RRF Top-3"],
            tablefmt="rounded_outline",
        )
    )

    dense_hit = hit(dense_res, expected)
    bm25_hit  = hit(bm25_res, expected)
    rrf_hit   = hit(rrf_res, expected)

    verdict = []
    if dense_hit and not bm25_hit:
        verdict.append("Dense WINS over BM25")
    elif bm25_hit and not dense_hit:
        verdict.append("BM25 WINS over Dense")
    elif dense_hit and bm25_hit:
        verdict.append("Both Dense and BM25 correct")
    else:
        verdict.append("Both Dense and BM25 MISS")

    if rrf_hit:
        verdict.append("RRF correct ✓")
    else:
        verdict.append("RRF also misses ✗")

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
        q         = qr["query"]
        d_hit     = qr["dense_hit"]
        b_hit     = qr["bm25_hit"]
        r_hit     = qr["rrf_hit"]
        best_hit  = d_hit or b_hit  # best individual method got it right

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
        print("    context-dependent queries (Q07, Q08). They require query")
        print("    rewriting with conversation history before retrieval.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build indexes
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

        expected  = q["expected_doc"]
        d_hit     = hit(dense_res, expected)
        b_hit     = hit(bm25_res, expected)
        r_hit     = hit(rrf_res, expected)

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

    # Persist results
    with open(RESULTS_FILE, "w") as f:
        json.dump(
            [
                {
                    "query_id":      qr["query"]["id"],
                    "query_text":    qr["query"]["text"],
                    "query_type":    qr["query"]["query_type"],
                    "difficulty":    qr["query"]["difficulty"],
                    "expected_doc":  qr["query"]["expected_doc"],
                    "dense_top5":    qr["dense_results"],
                    "bm25_top5":     qr["bm25_results"],
                    "rrf_top5":      qr["rrf_results"],
                    "dense_hit":     qr["dense_hit"],
                    "bm25_hit":      qr["bm25_hit"],
                    "rrf_hit":       qr["rrf_hit"],
                }
                for qr in query_results
            ],
            f,
            indent=2,
        )
    print(f"  Results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
