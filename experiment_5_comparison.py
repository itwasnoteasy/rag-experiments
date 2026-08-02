"""
experiment_5_comparison.py — Does retrieval method choice affect synthesis quality?

This experiment ties together all four prior experiments by running three retrieval
strategies on the same queries, synthesising answers with Gemini, and collecting
human evaluation scores to answer the central question.

═══════════════════════════════════════════════════════════════════════════════════
WHY IN PRODUCTION
═══════════════════════════════════════════════════════════════════════════════════
Teams often choose a retrieval stack once and never revisit it.  But retrieval
quality directly affects LLM output quality through a mechanism called
"context poisoning": if the top-2 documents are wrong or partially wrong, the
model either hallucinates, hedges excessively, or silently blends irrelevant
facts into the answer.  This experiment makes that cost VISIBLE by asking a
human rater to score three responses side-by-side — enabling data-driven
retrieval stack selection rather than gut-feel.

WHY HUMAN EVAL INSTEAD OF AUTOMATIC METRICS?
BLEU / ROUGE measure surface overlap; BERTScore measures semantic similarity
to a reference.  For open-domain QA with no gold answer, the only reliable
judge is a human (or a calibrated LLM-as-judge).  This experiment uses human
scoring as a deliberately low-cost proxy for a more expensive LLM-judge pass.

═══════════════════════════════════════════════════════════════════════════════════
MATHEMATICS
═══════════════════════════════════════════════════════════════════════════════════
Run A — Dense top-2
  bi-encoder cosine similarity; top-2 by score fed as context

Run B — BM25 top-2
  Okapi BM25 score; top-2 by score fed as context

Run C — RRF + cross-encoder top-2
  RRF(dense_top_10, bm25_top_10), then cross-encoder re-scores the top-10 pool,
  top-2 of re-ranked list fed as context

Human eval score: mean of (Accuracy + Completeness + Grounding) / 3, each 1–3.
Final ranking: mean score across queries per retrieval method.

═══════════════════════════════════════════════════════════════════════════════════
ALTERNATIVES & TRADEOFFS
═══════════════════════════════════════════════════════════════════════════════════
• LLM-as-judge (GPT-4 / Gemini Ultra): removes human bottleneck, but introduces
  position bias and self-preference when evaluating the same model's output.
• G-Eval / RAGAS: automated metrics purpose-built for RAG evaluation.  RAGAS
  measures faithfulness (does the answer contradict the context?), answer relevancy,
  and context precision/recall.  Trade-off: requires a reference corpus or LLM judge.
• A/B testing in production: route 5 % of traffic to each retrieval strategy,
  measure downstream satisfaction signals (thumbs up/down, session length).
  Most reliable but requires production traffic to get signal.
"""

import os
import sys
import json
import time
import textwrap
from typing import Any

# ── Optional tabulate ──────────────────────────────────────────────────────────
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ── Google Generative AI ───────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ── Sentence Transformers / CrossEncoder ──────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi

# ── Local modules ──────────────────────────────────────────────────────────────
from corpus import CORPUS
from queries import QUERIES

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# NOTE: "gemini-3.1-flash-lite" (as originally requested) does not exist as a
# real model ID.  The closest current model is "gemini-2.0-flash-lite".
# Override at runtime: GEMINI_MODEL=gemini-2.5-flash python experiment_5_comparison.py
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

TOP_K      = 10   # pool size for RRF + cross-encoder
CONTEXT_K  = 2    # docs fed to Gemini per run
RRF_K      = 60   # RRF smoothing constant

# Queries chosen to maximise contrast:
#   Q03 — semantic/hard:    paraphrase gap, no keyword overlap
#   Q07 — context_dep/hard: near-empty query, stress-tests all retrievers
#   Q09 — clear_intent/easy: control; all methods should agree
SELECTED_QUERY_IDS = ["Q03", "Q07", "Q09"]

RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "experiment_5_synthesis_comparison.json")

EVAL_DIMS = ["Accuracy", "Completeness", "Grounding"]
SCORE_RANGE = (1, 3)
WORD_LIMIT  = 100   # truncate synthesis responses for side-by-side display


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# WHY IN PRODUCTION: bi-encoder models are deployed as a microservice backed by
# a vector database (Pinecone, Weaviate, pgvector).  At query time only the query
# is encoded; document vectors are pre-computed and indexed.  Query latency is
# O(1) against index size because HNSW approximate search bounds the search.
#
# MATHEMATICS: cosine(q, d) = (q · d) / (||q|| × ||d||).  all-MiniLM-L6-v2 maps
# text to 384-dimensional unit sphere; cosine similarity equals dot product on the
# unit sphere.
#
# ALTERNATIVES: OpenAI text-embedding-3-small (1536-d), Cohere Embed v3 (1024-d),
# BGE-large (1024-d).  Trade-off: larger models cost more and are slower, but score
# higher on MTEB retrieval benchmarks.
def build_dense_index(corpus):
    """Embed all documents into a ChromaDB in-memory collection."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()  # ephemeral, in-memory
    collection = client.create_collection("exp5")
    texts = [d["content"] for d in corpus]
    ids   = [d["id"]      for d in corpus]
    embs  = model.encode(texts, show_progress_bar=False).tolist()
    collection.add(ids=ids, embeddings=embs, documents=texts)
    return model, collection


def dense_query(model, collection, query_text, top_k=TOP_K):
    """Return top-k results from dense retrieval sorted by cosine similarity."""
    q_emb = model.encode([query_text]).tolist()
    res   = collection.query(query_embeddings=q_emb, n_results=top_k)
    out   = []
    for rank, (doc_id, dist) in enumerate(
        zip(res["ids"][0], res["distances"][0]), start=1
    ):
        # ChromaDB returns L2 distance by default; convert to similarity for display
        # (doesn't affect ranking since both are monotone in the same direction)
        out.append({"doc_id": doc_id, "score": round(1 - dist / 2, 4), "rank": rank})
    return out


# WHY IN PRODUCTION: BM25 is the default baseline in Elasticsearch and OpenSearch.
# It excels on enterprise document search where exact product codes, policy IDs,
# and dollar amounts must be matched precisely.
#
# MATHEMATICS: score(D,Q) = Σ IDF(qi) × [f(qi,D)×(k1+1)] / [f(qi,D) + k1×(1-b+b×|D|/avgdl)]
# k1=1.5, b=0.75 by default.  High-IDF rare tokens (e.g. "INS-POL-2024") dominate.
#
# ALTERNATIVES: TF-IDF (no length normalisation), BM25+, BM25F (field-weighted).
def build_bm25_index(corpus):
    """Tokenise corpus and build BM25 index."""
    tokenised = [d["content"].lower().split() for d in corpus]
    return BM25Okapi(tokenised)


def bm25_query(bm25, corpus, query_text, top_k=TOP_K):
    """Return top-k results from BM25 sorted by BM25 score."""
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"doc_id": corpus[i]["id"], "score": round(s, 4), "rank": r + 1}
        for r, (i, s) in enumerate(ranked)
    ]


# WHY IN PRODUCTION: RRF is preferred over score-normalised fusion because it is
# immune to scale differences (cosine ∈ [0,1] vs BM25 ∈ [0, ∞)).  Used by MSMARCO
# winners and many production search stacks as a zero-parameter combiner.
#
# MATHEMATICS: RRF(d) = Σ 1/(k + rank_i(d)), k=60.  k=60 was empirically derived
# by Cormack et al. (2009) to down-weight rank-1 dominance.
#
# ALTERNATIVES: CombMNZ, Borda count, learned fusion (LambdaMART).
def reciprocal_rank_fusion(dense_results, bm25_results, k=RRF_K):
    """Combine dense and BM25 ranked lists with RRF."""
    scores: dict[str, float] = {}
    for result_list in [dense_results, bm25_results]:
        for item in result_list:
            doc_id = item["doc_id"]
            rank   = item["rank"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": d, "rrf_score": round(s, 6), "rank": r + 1}
            for r, (d, s) in enumerate(ranked)]


# WHY IN PRODUCTION: bi-encoders encode query and document independently, so they
# miss fine-grained query-document interaction.  Cross-encoders process the pair
# jointly (full self-attention over "[query][SEP][document]") and capture subtle
# relevance signals at the cost of O(n) inference per query.  In production they
# are applied only to the top-20–50 candidates from a cheaper first stage.
#
# MATHEMATICS: the cross-encoder produces a scalar relevance score per (q,d) pair
# using the [CLS] token representation after full joint attention.
#
# ALTERNATIVES: ColBERT (late interaction), MonoT5, RankLLM (prompting an LLM to
# rank).  ColBERT offers a middle ground: pre-computes per-token doc embeddings,
# then scores at query time with MaxSim — faster than cross-encoder, better than
# bi-encoder.
def build_cross_encoder():
    """Load cross-encoder/ms-marco-MiniLM-L-6-v2 for candidate re-ranking."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(cross_encoder, query_text, candidates, corpus_lookup):
    """
    Score (query, doc_content) pairs with cross-encoder and return
    re-ranked list with both rrf_rank and ce_rank.
    """
    pairs   = [(query_text, corpus_lookup[c["doc_id"]]) for c in candidates]
    ce_scores = cross_encoder.predict(pairs)
    annotated = [
        {**c, "ce_score": float(round(ce_scores[i], 4))}
        for i, c in enumerate(candidates)
    ]
    annotated.sort(key=lambda x: x["ce_score"], reverse=True)
    for new_rank, item in enumerate(annotated, start=1):
        item["ce_rank"] = new_rank
    return annotated


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════════

# WHY IN PRODUCTION: synthesis quality depends critically on what appears in the
# context window.  "Lost in the middle" (Liu et al., 2023) shows LLMs preferentially
# use context at position 1 and position n, ignoring the middle.  This is why
# Run C (cross-encoder) is expected to win: it places the single most relevant
# document at position 1.
#
# MATHEMATICS: for a two-document context, the model attends over O(n²) token pairs
# where n = |query| + |doc1| + |doc2|.  The attention pattern determines which
# tokens influence the generated answer.
#
# ALTERNATIVES: RAG-Fusion (generate multiple sub-queries, retrieve per sub-query,
# fuse before synthesis), HyDE (generate a hypothetical document first, embed it,
# then retrieve).  Both expand the effective retrieval surface at the cost of extra
# LLM calls.
def setup_gemini(api_key: str):
    """Configure Gemini client; exits with diagnostic if key is missing."""
    if not HAS_GENAI:
        print(
            "[ERROR] google-generativeai is not installed.\n"
            "        Run:  pip install google-generativeai"
        )
        sys.exit(1)
    if not api_key:
        print(
            "[ERROR] GEMINI_API_KEY environment variable is not set.\n"
            "        In Colab: from google.colab import userdata; "
            "os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')"
        )
        sys.exit(1)
    genai.configure(api_key=api_key)


def synthesise(model_name: str, query_text: str, docs: list[dict]) -> dict:
    """
    Synthesise an answer from top-k documents using Gemini.
    Returns dict with response_text, input_tokens, output_tokens, latency_ms.
    """
    context_block = "\n\n".join(
        f"[Document {i+1}: {d['id']} — {d.get('title','')}]\n{d['content']}"
        for i, d in enumerate(docs)
    )
    prompt = (
        f"You are a helpful telecom support assistant.\n\n"
        f"Answer the following customer question using ONLY the documents provided below. "
        f"Be concise (2–4 sentences). If the documents do not contain enough information "
        f"to answer fully, say so.\n\n"
        f"Question: {query_text}\n\n"
        f"Documents:\n{context_block}\n\n"
        f"Answer:"
    )

    gem_model = genai.GenerativeModel(model_name)
    t0 = time.time()
    response = gem_model.generate_content(prompt)
    latency_ms = round((time.time() - t0) * 1000, 1)

    text = response.text.strip()
    usage = response.usage_metadata
    return {
        "response_text": text,
        "input_tokens":  getattr(usage, "prompt_token_count",     0),
        "output_tokens": getattr(usage, "candidates_token_count", 0),
        "latency_ms":    latency_ms,
    }


def truncate_to_words(text: str, max_words: int = WORD_LIMIT) -> str:
    """Truncate response to max_words words, appending '…' if cut."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def _table(rows, headers):
    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="rounded_outline")
    # Fallback: simple fixed-width
    col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
    lines = [fmt.format(*headers), "-" * (sum(col_w) + 2 * len(col_w))]
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))
    return "\n".join(lines)


def print_context_docs(run_label: str, docs: list[dict]):
    """Show which documents are being fed to the LLM."""
    print(f"\n  {run_label} — context documents:")
    for i, d in enumerate(docs, 1):
        print(f"    [{i}] {d['id']} — {d.get('title', '')[:60]}")


def print_side_by_side(query_text: str, run_results: list[dict]):
    """Print three responses side-by-side (each truncated to WORD_LIMIT words)."""
    print(f"\n  Query: \"{query_text}\"")
    print()
    for r in run_results:
        label    = r["run_label"]
        doc_ids  = ", ".join(d["id"] for d in r["docs"])
        trunc    = truncate_to_words(r["synthesis"]["response_text"])
        wrapped  = textwrap.fill(trunc, width=64, subsequent_indent="       ")
        print(f"  [{label}] (docs: {doc_ids})")
        print(f"       {wrapped}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# HUMAN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

# WHY IN PRODUCTION: automated metrics (ROUGE, BERTScore) do not correlate well
# with human preference for open-domain QA.  Production teams use human annotation
# at the 0.1 % sample level, then train an LLM judge to scale.
#
# MATHEMATICS: mean score = (Accuracy + Completeness + Grounding) / 3.
# Ordinal scale 1–3: 1=poor, 2=acceptable, 3=excellent.  Simple arithmetic mean
# is appropriate for a 3-point scale with interval-like properties.
#
# ALTERNATIVES: Likert-5 scale (more granularity), pairwise preference (A vs B
# rather than absolute scores — removes scale-use bias), automated RAGAS scoring.
def collect_scores(run_label: str) -> dict[str, int]:
    """Prompt user to score one response on three dimensions (1–3)."""
    scores: dict[str, int] = {}
    print(f"  Score [{run_label}] — enter 1 (poor) / 2 (acceptable) / 3 (excellent)")
    for dim in EVAL_DIMS:
        while True:
            raw = input(f"    {dim}: ").strip()
            if raw in ("1", "2", "3"):
                scores[dim] = int(raw)
                break
            print(f"    Please enter 1, 2, or 3.")
    mean = round(sum(scores.values()) / len(scores), 2)
    scores["mean"] = mean
    return scores


def compute_final_summary(all_results: list[dict]) -> list[dict]:
    """
    Aggregate mean scores across queries per retrieval method.
    Returns list of {run, avg_accuracy, avg_completeness, avg_grounding, overall}.
    """
    run_labels = [r["run_label"] for r in all_results[0]["runs"]]
    agg: dict[str, dict[str, list]] = {lbl: {d: [] for d in EVAL_DIMS + ["mean"]}
                                        for lbl in run_labels}
    for q_result in all_results:
        for run in q_result["runs"]:
            lbl = run["run_label"]
            if "scores" not in run:
                continue
            for dim in EVAL_DIMS + ["mean"]:
                if dim in run["scores"]:
                    agg[lbl][dim].append(run["scores"][dim])

    summary_rows = []
    for lbl in run_labels:
        row: dict[str, Any] = {"run": lbl}
        for dim in EVAL_DIMS:
            vals = agg[lbl][dim]
            row[f"avg_{dim.lower()}"] = round(sum(vals) / len(vals), 2) if vals else 0.0
        means = agg[lbl]["mean"]
        row["overall"] = round(sum(means) / len(means), 2) if means else 0.0
        summary_rows.append(row)

    summary_rows.sort(key=lambda x: x["overall"], reverse=True)
    return summary_rows


def print_final_summary(summary_rows: list[dict]):
    """Render the final summary table and declare a winner."""
    section("FINAL SUMMARY — Retrieval Method vs Synthesis Quality")
    rows = [
        (
            ("🏆 " if i == 0 else "   ") + r["run"],
            r["avg_accuracy"],
            r["avg_completeness"],
            r["avg_grounding"],
            r["overall"],
        )
        for i, r in enumerate(summary_rows)
    ]
    headers = ["Method", "Avg Accuracy", "Avg Completeness", "Avg Grounding", "Overall"]
    print(_table(rows, headers))

    winner = summary_rows[0]
    loser  = summary_rows[-1]
    delta  = round(winner["overall"] - loser["overall"], 2)
    print(f"\n  Winner:    {winner['run']} (overall {winner['overall']})")
    print(f"  Runner-up: {summary_rows[1]['run']} (overall {summary_rows[1]['overall']})")
    print(f"  Gap (best vs worst): {delta} points on a 1–3 scale")
    print()
    if delta >= 0.5:
        print("  CONCLUSION: Retrieval method choice has a MEANINGFUL impact on synthesis quality.")
    elif delta >= 0.2:
        print("  CONCLUSION: Retrieval method choice has a MODERATE impact on synthesis quality.")
    else:
        print("  CONCLUSION: Retrieval method choice has a MINIMAL impact on synthesis quality")
        print("              for this query set — consider testing with harder queries.")

    print("""
  PRODUCTION IMPLICATIONS
  ─────────────────────────────────────────────────────────────────
  • If Run C (RRF+CE) wins: the cross-encoder re-ranking cost is justified.
    Consider selective re-ranking — only re-rank queries with low BM25/dense
    agreement to save 30–60 % of latency.
  • If Run A (Dense) wins: invest in embedding model quality rather than
    fusion complexity.  Upgrade to a larger bi-encoder (BGE-large, E5-large).
  • If Run B (BM25) wins: your corpus contains domain-specific terminology
    that benefits from exact token matching.  Consider adding BM25 in
    parallel to any existing dense-only retrieval pipeline.
  • If scores are nearly equal: retrieval is not your bottleneck.  Invest in
    prompt engineering, answer length tuning, or query rewriting instead.
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    section("Experiment 5 — Does Retrieval Method Choice Affect Synthesis Quality?")
    print(f"  Model:    {GEMINI_MODEL}")
    print(f"  Queries:  {', '.join(SELECTED_QUERY_IDS)}")
    print(f"  Context:  top-{CONTEXT_K} docs per retrieval method")

    # ── Validate Gemini setup ──────────────────────────────────────────────────
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    setup_gemini(gemini_key)

    # ── Build indexes ──────────────────────────────────────────────────────────
    section("Building retrieval indexes …")
    print("  [1/3] Dense (bi-encoder) …", end=" ", flush=True)
    dense_model, collection = build_dense_index(CORPUS)
    print("done")

    print("  [2/3] BM25 …", end=" ", flush=True)
    bm25 = build_bm25_index(CORPUS)
    print("done")

    print("  [3/3] Cross-encoder …", end=" ", flush=True)
    cross_encoder = build_cross_encoder()
    print("done")

    # Build id→content lookup for cross-encoder
    corpus_lookup = {d["id"]: d["content"] for d in CORPUS}
    corpus_by_id  = {d["id"]: d            for d in CORPUS}

    # ── Filter selected queries ────────────────────────────────────────────────
    selected = [q for q in QUERIES if q["id"] in SELECTED_QUERY_IDS]

    all_results = []

    for q in selected:
        section(f"Query {q['id']} — {q['query_type'].upper()} / {q['difficulty'].upper()}")
        print(f"  \"{q['text']}\"")
        print(f"  Expected: {q['expected_doc']}  |  Notes: {q['notes'][:80]}…")

        # ── Run A: Dense top-CONTEXT_K ─────────────────────────────────────────
        dense_res  = dense_query(dense_model, collection, q["text"], top_k=TOP_K)
        dense_docs = [corpus_by_id[r["doc_id"]] for r in dense_res[:CONTEXT_K]]

        # ── Run B: BM25 top-CONTEXT_K ─────────────────────────────────────────
        bm25_res  = bm25_query(bm25, CORPUS, q["text"], top_k=TOP_K)
        bm25_docs = [corpus_by_id[r["doc_id"]] for r in bm25_res[:CONTEXT_K]]

        # ── Run C: RRF + Cross-encoder top-CONTEXT_K ──────────────────────────
        rrf_pool     = reciprocal_rank_fusion(dense_res, bm25_res)
        rrf_top      = rrf_pool[:TOP_K]
        # Rerank the RRF pool with cross-encoder
        reranked     = rerank(cross_encoder, q["text"], rrf_top, corpus_lookup)
        ce_docs      = [corpus_by_id[r["doc_id"]] for r in reranked[:CONTEXT_K]]

        # ── Synthesis ──────────────────────────────────────────────────────────
        runs = [
            {"run_label": "A: Dense",    "docs": dense_docs, "raw_results": dense_res[:TOP_K]},
            {"run_label": "B: BM25",     "docs": bm25_docs,  "raw_results": bm25_res[:TOP_K]},
            {"run_label": "C: RRF+CE",   "docs": ce_docs,    "raw_results": reranked[:TOP_K]},
        ]

        print("\n  Synthesising with Gemini …")
        for run in runs:
            print(f"    {run['run_label']} …", end=" ", flush=True)
            syn = synthesise(GEMINI_MODEL, q["text"], run["docs"])
            run["synthesis"] = syn
            print(f"done  ({syn['latency_ms']} ms, {syn['output_tokens']} tokens out)")

        # ── Display side-by-side ───────────────────────────────────────────────
        print_side_by_side(q["text"], runs)

        # ── Human scoring ──────────────────────────────────────────────────────
        print("  ── Human Evaluation ──────────────────────────────────────")
        for run in runs:
            print_context_docs(run["run_label"], run["docs"])
            run["scores"] = collect_scores(run["run_label"])
            mean = run["scores"]["mean"]
            bar  = "█" * int(mean * 10) + "░" * (30 - int(mean * 10))
            print(f"    → Mean {mean:.2f}  [{bar}]")

        all_results.append({
            "query_id":    q["id"],
            "query_text":  q["text"],
            "query_type":  q["query_type"],
            "difficulty":  q["difficulty"],
            "expected_doc": q["expected_doc"],
            "runs": [
                {
                    "run_label": run["run_label"],
                    "doc_ids":   [d["id"] for d in run["docs"]],
                    "synthesis": run["synthesis"],
                    "scores":    run["scores"],
                }
                for run in runs
            ],
        })

    # ── Final summary table ────────────────────────────────────────────────────
    summary = compute_final_summary(all_results)
    print_final_summary(summary)

    # ── Persist results ────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "experiment":    "experiment_5_comparison",
        "model":         GEMINI_MODEL,
        "context_k":     CONTEXT_K,
        "top_k_pool":    TOP_K,
        "selected_queries": SELECTED_QUERY_IDS,
        "eval_dimensions": EVAL_DIMS,
        "query_results": all_results,
        "summary":       summary,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
