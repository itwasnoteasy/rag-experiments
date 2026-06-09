"""
Experiment 2: Cross-Encoder Re-ranking of RRF Results

Loads RRF top-10 from experiment_1_results.json, re-ranks each query's
candidates with a cross-encoder, then reports how much the ranking changed.

Run:
    python experiment_2_reranking.py

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

# Build a doc_id → content lookup once
DOC_CONTENT = {doc["id"]: doc["content"] for doc in CORPUS}
DOC_TITLE   = {doc["id"]: doc["title"]   for doc in CORPUS}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD RRF TOP-10 FROM EXPERIMENT 1
# ─────────────────────────────────────────────────────────────────────────────

def load_experiment_1(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path}.\n"
            "Run experiment_1_retrieval.py first to generate this file."
        )
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CROSS-ENCODER RE-RANKING
# ─────────────────────────────────────────────────────────────────────────────

def build_cross_encoder():
    print("=== Loading cross-encoder (ms-marco-MiniLM-L-6-v2) ===")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  Model loaded.\n")
    return model


def rerank(cross_encoder, query_text, rrf_top10):
    """
    Cross-encoder scoring: unlike the bi-encoder (sentence-transformers) which
    embeds the query and each document independently and then compares vectors,
    the cross-encoder receives BOTH the query and the document concatenated as a
    single input — "[query] [SEP] [document]" — and runs full self-attention
    across all tokens of both texts simultaneously. This lets every query token
    attend to every document token in one forward pass, capturing fine-grained
    relevance signals (e.g., the exact phrase "non-return fee" in the document
    matching the specific dollar amount in the query). The cost is that you
    cannot pre-compute document embeddings: every query-document pair requires
    its own inference call, making it O(n) per query vs. O(1) lookup for
    bi-encoders. That's why cross-encoders are used only on a small candidate
    set (top-10 to top-100) after a fast first-stage retriever has narrowed
    the field.
    """
    pairs = [(query_text, DOC_CONTENT[r["doc_id"]]) for r in rrf_top10]
    scores = cross_encoder.predict(pairs).tolist()

    # Attach cross-encoder scores and sort descending
    candidates = [
        {**r, "ce_score": round(float(s), 4)}
        for r, s in zip(rrf_top10, scores)
    ]
    candidates.sort(key=lambda x: x["ce_score"], reverse=True)

    # Assign new ranks after re-ranking
    for new_rank, c in enumerate(candidates, start=1):
        c["ce_rank"] = new_rank

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# 3. OUTPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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
    qid   = q_record["query_id"]
    qtype = q_record["query_type"]
    diff  = q_record["difficulty"]
    print(f"  {qid} [{qtype}] [{diff}]")
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

    print(
        tabulate(
            rows,
            headers=["Doc ID", "RRF Rank", "CE Rank", "CE Score", "Movement"],
            tablefmt="rounded_outline",
            colalign=("left", "center", "center", "right", "center"),
        )
    )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. SUMMARY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(all_query_data):
    total_displaced_docs = 0    # docs that moved more than 2 positions
    max_displacement     = 0
    max_displacement_qid = None
    promotion_example    = None  # first doc promoted from rank ≥4 to rank ≤2

    per_query_displacement = {}

    for qd in all_query_data:
        qid      = qd["query_id"]
        reranked = qd["reranked"]

        query_max_delta = 0
        for c in reranked:
            delta = abs(c["rrf_rank"] - c["ce_rank"])
            if delta > 2:
                total_displaced_docs += 1
            if delta > query_max_delta:
                query_max_delta = delta

            # Look for promotion: RRF rank ≥4, CE rank ≤2
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
        "total_displaced_docs":     total_displaced_docs,
        "max_displacement_qid":     max_displacement_qid,
        "max_displacement_value":   max_displacement,
        "per_query_max_delta":      per_query_displacement,
        "promotion_example":        promotion_example,
    }


def print_summary(summary, all_query_data):
    print("=" * 80)
    print("  SUMMARY ANALYSIS")
    print("=" * 80)

    print(f"\n  Documents that moved more than 2 positions: "
          f"{summary['total_displaced_docs']}")

    print(f"\n  Query with most re-ranking displacement: "
          f"{summary['max_displacement_qid']} "
          f"(max single-doc shift = {summary['max_displacement_value']} positions)")

    # Print per-query max delta as a small table
    rows = []
    for qd in all_query_data:
        qid   = qd["query_id"]
        delta = summary["per_query_max_delta"][qid]
        bar   = "█" * delta
        rows.append([qid, qd["query_type"], delta, bar])
    print()
    print(
        tabulate(
            rows,
            headers=["Query", "Type", "Max Δ", ""],
            tablefmt="rounded_outline",
            colalign=("left", "left", "center", "left"),
        )
    )

    # Promotion example
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
        # Generate a one-sentence explanation based on query type
        qd = next(x for x in all_query_data if x["query_id"] == ex["query_id"])
        if qd["query_type"] == "exact_match":
            reason = (
                "The cross-encoder likely preferred this document because it could "
                "attend simultaneously to the exact token in the query and its "
                "occurrence in the document body, confirming a precise factual match "
                "that RRF's rank aggregation had diluted."
            )
        elif qd["query_type"] == "semantic":
            reason = (
                "The cross-encoder likely preferred this document because joint "
                "attention over query and document tokens revealed semantic alignment "
                "that the bi-encoder's independent embeddings had underweighted."
            )
        elif qd["query_type"] == "ambiguous":
            reason = (
                "The cross-encoder likely preferred this document because reading "
                "both texts together let it resolve the ambiguity and identify which "
                "category the query actually required, something neither BM25 nor the "
                "bi-encoder could do independently."
            )
        else:
            reason = (
                "The cross-encoder likely preferred this document because full "
                "cross-attention between query and document tokens captured relevance "
                "signals invisible to retrievers that process them separately."
            )
        print(f"  Why: {reason}")
    else:
        print("  No document was promoted from rank ≥4 to rank ≤2 in this run.")
        print("  This can happen when the first-stage retrieval is already high quality.")

    print()
    print("  Key Takeaways")
    print("  " + "─" * 60)
    print("  • Cross-encoders re-order candidates, they do not retrieve —")
    print("    they only see what the first stage passed them.")
    print("  • High displacement = the bi-encoder and BM25 were ranking by")
    print("    surface similarity; the cross-encoder found deeper relevance.")
    print("  • Low displacement = first-stage retrieval was already well-ordered;")
    print("    re-ranking adds latency with little accuracy gain — skip it.")
    print("  • In production: retrieve top-50 to top-100 fast, re-rank top-10.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    exp1_data    = load_experiment_1(INPUT_FILE)
    cross_encoder = build_cross_encoder()

    print("=" * 80)
    print("  PER-QUERY DISPLACEMENT: RRF rank → Cross-encoder rank")
    print("=" * 80)
    print()

    all_query_data = []

    for record in exp1_data:
        rrf_top10 = record["rrf_top5"]   # experiment 1 stored top-5; use what's available

        reranked = rerank(cross_encoder, record["query_text"], rrf_top10)

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

    # Accuracy comparison
    rrf_correct = sum(1 for qd in all_query_data if qd["rrf_hit"])
    ce_correct  = sum(1 for qd in all_query_data if qd["ce_hit"])
    total       = len(all_query_data)
    print(f"  Accuracy (top-3 hit rate)")
    print(f"    RRF alone       : {rrf_correct}/{total}")
    print(f"    After re-ranking: {ce_correct}/{total}")
    print()

    # Persist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(
            {
                "summary": summary,
                "queries": all_query_data,
            },
            f,
            indent=2,
        )
    print(f"  Results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
