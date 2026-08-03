"""
experiment_6_llm_judge.py — Automated RAG Quality Scoring via LLM-as-Judge

Central question: "How do you evaluate RAG quality at scale without manually
reviewing every response?"

This experiment implements the LLM-as-judge pattern with RAGAS-style per-dimension
scoring.  For all 10 queries it:
  1. Synthesises a response using the best retrieval pipeline (RRF + cross-encoder,
     inherited from experiment 2).
  2. Calls Gemini FOUR more times — one per dimension — as an independent judge,
     forcing structured JSON output each time.
  3. Compares the automated scores against the 3 manual scores from experiment 5.

═══════════════════════════════════════════════════════════════════════════════════
WHY IN PRODUCTION
═══════════════════════════════════════════════════════════════════════════════════
Manual evaluation is the gold standard but scales to ~100 samples/day per
annotator.  A production RAG system may handle 100,000 queries per day — you
cannot review them all.  LLM-as-judge (also called "model-graded evaluation")
lets you evaluate at scale by replacing a human annotator with a prompted LLM.

Key risk: the judge model inherits the generator model's biases.  Using the same
model for generation and judgment inflates scores (self-preference bias).  Prefer:
  • A larger or different model family as judge (e.g., Gemini Ultra judging Gemini Flash)
  • Calibration against a held-out set of human ratings before trusting the judge
  • Swap-consistency checks (reverse A/B order and verify scores are stable)

This experiment uses the same model family (Gemini) for both generator and judge,
which is a known limitation documented intentionally so you can see the bias.

═══════════════════════════════════════════════════════════════════════════════════
MATHEMATICS — RAGAS-STYLE DIMENSIONS
═══════════════════════════════════════════════════════════════════════════════════
RAGAS (Shahul Es et al., 2023) decomposes RAG quality into four independent axes:

  FAITHFULNESS     — are all claims in the answer supported by the retrieved context?
                     Score 0–1: (supported claims) / (total claims in answer)
                     Judge must quote the violating sentence when score < 1.

  ANSWER RELEVANCE — does the answer address the question that was actually asked?
                     Score 0–1: cosine(question_embedding, answer_embedding) in
                     the original RAGAS paper; here approximated by LLM judgment.

  CONTEXT PRECISION — what fraction of retrieved chunks are relevant to the answer?
                     Score 0–1: (relevant chunks) / (total retrieved chunks)
                     Measures retrieval over-fetching / noise.

  CONTEXT RECALL  — does the context contain all information needed for a complete
                     answer?  Score 0–1: (answerable sub-questions) / (total
                     sub-questions in the ideal answer).
                     High recall → low "I don't know" hedges.

  OVERALL = mean(faithfulness, answer_relevance, context_precision, context_recall)

  FAITHFULNESS ALERT threshold = 0.7 (flags for human review).

═══════════════════════════════════════════════════════════════════════════════════
ALTERNATIVES & TRADEOFFS
═══════════════════════════════════════════════════════════════════════════════════
• DeepEval: open-source library with RAGAS metrics, LLM-as-judge, hallucination
  detectors.  Can drop in as a replacement for the hand-rolled judge prompts here.
• G-Eval (Liu et al., 2023): uses chain-of-thought before scoring, reducing
  positional bias; marginally more calibrated but 2–3× the token cost.
• Prometheus (open-weight judge): fine-tuned Llama-based judge; avoids API costs
  and self-preference when judging your own model family's outputs.
• Human-in-the-loop hybrid: route only low-confidence judge scores (0.4–0.6) to
  human reviewers; auto-approve clear passes (>0.85) and clear fails (<0.3).
  Cuts annotation cost by ~70 % while preserving coverage of edge cases.
"""

import json
import os
import re
import sys
import time
from typing import Any

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi

from corpus import CORPUS
from queries import QUERIES

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_MODEL      = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

# Rate-limit guard: free-tier quota is ~10–15 requests/minute.
# Each query makes 5 Gemini calls (1 synthesis + 4 judge dimensions) × 10 queries = 50 calls.
# GEMINI_DELAY_S adds a pause AFTER every generate_content() call.
# Default 5 s → ~12 RPM, safely under a 15 RPM limit.
# Override: GEMINI_DELAY_S=0 to disable (paid tier / higher quota).
GEMINI_DELAY_S    = float(os.environ.get("GEMINI_DELAY_S", "5"))

TOP_K             = 10
CONTEXT_K         = 2         # docs fed to synthesiser
RRF_K             = 60
FAITHFULNESS_ALERT = 0.7      # below this → "NEEDS HUMAN REVIEW"

RESULTS_DIR  = "results"
EXP5_FILE    = os.path.join(RESULTS_DIR, "experiment_5_synthesis_comparison.json")
OUTPUT_FILE  = os.path.join(RESULTS_DIR, "experiment_6_judge_results.json")

# Dimension definitions — each gets its own Gemini call with a tailored prompt
DIMENSIONS = [
    {
        "key":  "faithfulness",
        "name": "Faithfulness",
        "instruction": (
            "Assess whether every factual claim in the ANSWER is explicitly supported by "
            "the RETRIEVED CONTEXT.  A claim is faithful if it can be directly traced to "
            "a sentence in the context.  A claim is a violation if it adds facts not present "
            "in the context, even if those facts are plausible or correct.\n\n"
            "If you find any violation, quote the exact violating sentence from the ANSWER "
            "in your reasoning field.\n\n"
            "Score 1.0 = fully faithful (no violations).  "
            "Score 0.0 = answer is entirely fabricated.  "
            "Score 0.5 = roughly half the claims have no grounding in the context."
        ),
    },
    {
        "key":  "answer_relevance",
        "name": "Answer Relevance",
        "instruction": (
            "Assess whether the ANSWER directly addresses the QUESTION that was asked.  "
            "Penalise answers that are off-topic, answer a different question, or are so "
            "vague that they provide no actionable information about the asked topic.\n\n"
            "Score 1.0 = answer is precisely on-topic and directly responds to the question.  "
            "Score 0.0 = answer is entirely off-topic.  "
            "Score 0.5 = answer is partially relevant but misses the core of the question."
        ),
    },
    {
        "key":  "context_precision",
        "name": "Context Precision",
        "instruction": (
            "You are given the RETRIEVED CONTEXT (one or more documents) and the ANSWER.  "
            "Assess what fraction of the retrieved documents were actually useful for "
            "constructing the answer.  A document is useful if the answer draws on "
            "information contained in it.  A document is noise if the answer could have "
            "been written without it.\n\n"
            "Score 1.0 = all retrieved documents contributed to the answer.  "
            "Score 0.0 = no retrieved document was used.  "
            "Score 0.5 = roughly half the documents were relevant."
        ),
    },
    {
        "key":  "context_recall",
        "name": "Context Recall",
        "instruction": (
            "Assess whether the RETRIEVED CONTEXT contains ALL the information needed to "
            "fully and completely answer the QUESTION.  Consider what an ideal, complete "
            "answer would need to include, then check whether that information is present "
            "in the context.\n\n"
            "Score 1.0 = context is fully sufficient; nothing is missing.  "
            "Score 0.0 = context is entirely insufficient; cannot answer the question at all.  "
            "Score 0.5 = context allows a partial answer but important information is absent."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL STACK  (same as experiment_5, inlined here so experiment_6 is
# self-contained — no dependency on experiment_5 having been run with synthesis)
# ══════════════════════════════════════════════════════════════════════════════

def build_dense_index(corpus):
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    coll   = client.create_collection("exp6")
    texts  = [d["content"] for d in corpus]
    ids    = [d["id"]      for d in corpus]
    embs   = model.encode(texts, show_progress_bar=False).tolist()
    coll.add(ids=ids, embeddings=embs, documents=texts)
    return model, coll


def dense_query(model, collection, query_text, top_k=TOP_K):
    q_emb = model.encode([query_text]).tolist()
    res   = collection.query(query_embeddings=q_emb, n_results=top_k)
    return [
        {"doc_id": doc_id, "rank": r + 1}
        for r, doc_id in enumerate(res["ids"][0])
    ]


def build_bm25_index(corpus):
    tokenised = [d["content"].lower().split() for d in corpus]
    return BM25Okapi(tokenised)


def bm25_query(bm25, corpus, query_text, top_k=TOP_K):
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"doc_id": corpus[i]["id"], "rank": r + 1} for r, (i, _) in enumerate(ranked)]


def reciprocal_rank_fusion(dense_res, bm25_res, k=RRF_K):
    scores: dict[str, float] = {}
    for lst in [dense_res, bm25_res]:
        for item in lst:
            scores[item["doc_id"]] = scores.get(item["doc_id"], 0.0) + 1.0 / (k + item["rank"])
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": d, "rrf_score": s, "rank": r + 1} for r, (d, s) in enumerate(ranked)]


def rerank_candidates(cross_encoder, query_text, candidates, corpus_lookup):
    pairs  = [(query_text, corpus_lookup[c["doc_id"]]) for c in candidates]
    scores = cross_encoder.predict(pairs)
    tagged = [{**c, "ce_score": float(scores[i])} for i, c in enumerate(candidates)]
    tagged.sort(key=lambda x: x["ce_score"], reverse=True)
    for new_rank, item in enumerate(tagged, start=1):
        item["ce_rank"] = new_rank
    return tagged


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _gemini_call(model, prompt_or_content) -> "genai.types.GenerateContentResponse":
    """Wrapper around generate_content that enforces the inter-call rate-limit delay."""
    response = model.generate_content(prompt_or_content)
    if GEMINI_DELAY_S > 0:
        time.sleep(GEMINI_DELAY_S)
    return response


def setup_gemini(api_key: str):
    if not HAS_GENAI:
        print("[ERROR] google-generativeai not installed.  Run: pip install google-generativeai")
        sys.exit(1)
    if not api_key:
        print(
            "[ERROR] GEMINI_API_KEY not set.\n"
            "        In Colab: os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')"
        )
        sys.exit(1)
    genai.configure(api_key=api_key)


# WHY IN PRODUCTION: synthesis and judging use the same model family here.
# In production you would use a stronger/different model for judging to avoid
# self-preference bias.  The separation of the synthesis prompt from the judge
# prompts is the key architectural pattern to retain.
def synthesise(model_name: str, query_text: str, docs: list[dict]) -> str:
    """Generate a response grounded in retrieved docs. Returns response text."""
    context = "\n\n".join(
        f"[Document {i+1}: {d['id']} — {d.get('title','')}]\n{d['content']}"
        for i, d in enumerate(docs)
    )
    prompt = (
        "You are a helpful telecom support assistant.\n\n"
        "Answer the following customer question using ONLY the documents provided below. "
        "Be concise (2–4 sentences). If the documents do not contain enough information "
        "to answer fully, say so.\n\n"
        f"Question: {query_text}\n\n"
        f"Documents:\n{context}\n\nAnswer:"
    )
    model    = genai.GenerativeModel(model_name)
    response = _gemini_call(model, prompt)
    return response.text.strip()


# WHY IN PRODUCTION: each dimension gets a SEPARATE Gemini call with a tightly
# constrained system prompt.  Combining all dimensions into one call produces
# lower-quality scores because the model anchors on the first dimension's
# reasoning and gives less independent consideration to later ones.  The cost
# is 4× the API calls, but each call is cheap (few-shot, small output).
#
# ALTERNATIVES: G-Eval uses chain-of-thought ("think step by step, then output
# a score") which improves calibration but adds ~200 tokens of reasoning per
# call.  Prometheus fine-tunes a judge model so no API calls are needed at all.
def judge_dimension(
    model_name: str,
    dimension: dict,
    query_text: str,
    context_text: str,
    answer_text: str,
    retries: int = 3,
) -> dict:
    """
    Call Gemini as an independent judge for one RAGAS-style dimension.
    Forces JSON output: {"score": 0.0–1.0, "reasoning": "one sentence"}.
    Retries up to `retries` times if JSON parsing fails.
    """
    system = (
        "You are an impartial quality evaluator for AI-generated answers.\n"
        "Your only job is to output a JSON object with exactly two keys:\n"
        '  "score": a float between 0.0 and 1.0\n'
        '  "reasoning": a single sentence explaining your score\n\n'
        "Do NOT output anything outside the JSON object. "
        "Do NOT wrap it in markdown code fences."
    )
    user = (
        f"TASK: {dimension['instruction']}\n\n"
        f"QUESTION: {query_text}\n\n"
        f"RETRIEVED CONTEXT:\n{context_text}\n\n"
        f"ANSWER:\n{answer_text}\n\n"
        "Output JSON only:"
    )

    gem_model = genai.GenerativeModel(
        model_name,
        system_instruction=system,
    )

    for attempt in range(retries):
        try:
            t0       = time.time()
            response = _gemini_call(gem_model, user)
            latency  = round((time.time() - t0) * 1000, 1)
            raw      = response.text.strip()

            # Strip markdown fences if the model ignored instructions
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$",          "", raw)

            parsed = json.loads(raw)
            score  = float(parsed.get("score", 0.0))
            score  = max(0.0, min(1.0, score))   # clamp to [0, 1]
            return {
                "score":      round(score, 3),
                "reasoning":  str(parsed.get("reasoning", "")),
                "latency_ms": latency,
            }
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            if attempt == retries - 1:
                return {"score": 0.0, "reasoning": f"[parse error: {exc}]", "latency_ms": 0.0}
            time.sleep(1.0 * (attempt + 1))

    return {"score": 0.0, "reasoning": "[max retries exceeded]", "latency_ms": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")


def _table(rows, headers, floatfmt=".3f"):
    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="rounded_outline", floatfmt=floatfmt)
    col_w = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
    sep   = "-" * (sum(col_w) + 2 * len(col_w))
    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))
    return "\n".join(lines)


def score_bar(score: float, width: int = 20) -> str:
    """Visual 0–1 score bar."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 CROSS-REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

# WHY IN PRODUCTION: comparing automated judge scores to human ratings is the
# calibration step that determines how much you can trust the judge.  A Pearson
# correlation > 0.7 between judge scores and human scores is considered acceptable
# for production use.  Below 0.5, the judge has low agreement with humans and
# should not be trusted as a standalone evaluator.
#
# MATHEMATICS: the experiment_5 human score is on a 1–3 ordinal scale;
# the LLM judge uses 0–1.  We normalise: human_norm = (human_score - 1) / 2.
# Mean human dim score is the average across Accuracy/Completeness/Grounding
# for the Run C (RRF+CE) response, since that is the same retrieval method
# used in experiment_6.  Disagreement = |human_norm - llm_overall|.
def load_exp5_manual_scores(path: str) -> dict[str, dict]:
    """
    Load experiment_5 results and extract manual scores for the Run C responses
    (RRF+CE, which is the method used in experiment_6).

    Returns dict keyed by query_id → {mean_1to3, mean_norm, dims}.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for qr in data.get("query_results", []):
        qid = qr["query_id"]
        for run in qr.get("runs", []):
            if "RRF" in run["run_label"] or "C:" in run["run_label"]:
                scores = run.get("scores", {})
                dims   = {k: v for k, v in scores.items() if k != "mean"}
                mean13 = scores.get("mean", 0.0)
                out[qid] = {
                    "mean_1to3": mean13,
                    "mean_norm": round((mean13 - 1) / 2, 3),
                    "dims":      dims,
                }
                break
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    section("Experiment 6 — LLM-as-Judge: Automated RAG Quality Scoring")
    print(f"  Generator model : {GEMINI_MODEL}")
    print(f"  Judge model     : {GEMINI_MODEL}  ← same family (known self-preference bias)")
    print(f"  Dimensions      : {', '.join(d['name'] for d in DIMENSIONS)}")
    print(f"  Queries         : all 10")

    # ── Setup ──────────────────────────────────────────────────────────────────
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    setup_gemini(gemini_key)

    # ── Build retrieval indexes ────────────────────────────────────────────────
    section("Building retrieval indexes …")
    print("  Dense  …", end=" ", flush=True)
    dense_model, collection = build_dense_index(CORPUS)
    print("done")

    print("  BM25   …", end=" ", flush=True)
    bm25 = build_bm25_index(CORPUS)
    print("done")

    print("  Cross-encoder …", end=" ", flush=True)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("done")

    corpus_lookup = {d["id"]: d["content"] for d in CORPUS}
    corpus_by_id  = {d["id"]: d            for d in CORPUS}

    # ── Load experiment_5 manual scores for cross-reference ───────────────────
    exp5_scores = load_exp5_manual_scores(EXP5_FILE)
    if exp5_scores:
        print(f"\n  Loaded experiment_5 manual scores for {len(exp5_scores)} queries.")
    else:
        print(f"\n  No experiment_5 results found at {EXP5_FILE} — skipping cross-reference.")

    # ── Per-query loop ─────────────────────────────────────────────────────────
    all_results = []
    total_queries = len(QUERIES)

    for qi, q in enumerate(QUERIES, start=1):
        section(f"[{qi}/{total_queries}] {q['id']} — {q['query_type'].upper()} / {q['difficulty'].upper()}")
        print(f"  Query: \"{q['text']}\"")

        # Retrieve: RRF + cross-encoder top-CONTEXT_K
        dense_res    = dense_query(dense_model, collection, q["text"])
        bm25_res     = bm25_query(bm25, CORPUS, q["text"])
        rrf_pool     = reciprocal_rank_fusion(dense_res, bm25_res)[:TOP_K]
        reranked     = rerank_candidates(cross_encoder, q["text"], rrf_pool, corpus_lookup)
        context_docs = [corpus_by_id[r["doc_id"]] for r in reranked[:CONTEXT_K]]

        doc_ids = [d["id"] for d in context_docs]
        print(f"  Context docs: {', '.join(doc_ids)}")

        # Synthesise
        print(f"  Synthesising …", end=" ", flush=True)
        answer = synthesise(GEMINI_MODEL, q["text"], context_docs)
        print("done")
        print(f"  Answer: {' '.join(answer.split()[:30])}{'…' if len(answer.split()) > 30 else ''}")

        # Build context string for judge prompts
        context_text = "\n\n".join(
            f"[Doc {i+1}: {d['id']}]\n{d['content']}"
            for i, d in enumerate(context_docs)
        )

        # Judge — one Gemini call per dimension
        print(f"  Judging ({len(DIMENSIONS)} dimensions) …")
        dim_scores: dict[str, dict] = {}
        for dim in DIMENSIONS:
            print(f"    {dim['name']:<22} …", end=" ", flush=True)
            result = judge_dimension(
                GEMINI_MODEL, dim, q["text"], context_text, answer
            )
            dim_scores[dim["key"]] = result
            bar = score_bar(result["score"], width=15)
            print(f"{result['score']:.3f}  [{bar}]  {result['latency_ms']} ms")

        overall = round(
            sum(dim_scores[d["key"]]["score"] for d in DIMENSIONS) / len(DIMENSIONS), 3
        )

        all_results.append({
            "query_id":    q["id"],
            "query_text":  q["text"],
            "query_type":  q["query_type"],
            "difficulty":  q["difficulty"],
            "expected_doc": q["expected_doc"],
            "context_doc_ids": doc_ids,
            "answer":      answer,
            "dim_scores":  {
                d["key"]: {
                    "score":     dim_scores[d["key"]]["score"],
                    "reasoning": dim_scores[d["key"]]["reasoning"],
                }
                for d in DIMENSIONS
            },
            "overall":     overall,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # OUTPUT 1: Main scoring table
    # ══════════════════════════════════════════════════════════════════════════
    section("RAGAS-Style Scores — All Queries")

    rows = []
    for r in all_results:
        ds = r["dim_scores"]
        rows.append((
            r["query_id"],
            r["query_type"][:10],
            r["difficulty"],
            f"{ds['faithfulness']['score']:.3f}",
            f"{ds['answer_relevance']['score']:.3f}",
            f"{ds['context_precision']['score']:.3f}",
            f"{ds['context_recall']['score']:.3f}",
            f"{r['overall']:.3f}",
        ))

    headers = ["Query", "Type", "Difficulty", "Faithful", "Relevance", "Ctx Prec", "Ctx Recall", "Overall"]
    print(_table(rows, headers, floatfmt=".3f"))

    # Column means
    def col_mean(key):
        return round(sum(r["dim_scores"][key]["score"] for r in all_results) / len(all_results), 3)

    print(f"\n  Column means:")
    for dim in DIMENSIONS:
        mean = col_mean(dim["key"])
        bar  = score_bar(mean, width=20)
        print(f"    {dim['name']:<22} {mean:.3f}  [{bar}]")
    overall_mean = round(sum(r["overall"] for r in all_results) / len(all_results), 3)
    print(f"    {'Overall':<22} {overall_mean:.3f}  [{score_bar(overall_mean, width=20)}]")

    # ══════════════════════════════════════════════════════════════════════════
    # OUTPUT 2: Faithfulness alerts
    # ══════════════════════════════════════════════════════════════════════════
    section(f"Faithfulness Alerts  (score < {FAITHFULNESS_ALERT})")

    flagged = [r for r in all_results if r["dim_scores"]["faithfulness"]["score"] < FAITHFULNESS_ALERT]
    if flagged:
        print(f"  {len(flagged)} quer{'y' if len(flagged)==1 else 'ies'} flagged as NEEDS HUMAN REVIEW:\n")
        for r in flagged:
            score     = r["dim_scores"]["faithfulness"]["score"]
            reasoning = r["dim_scores"]["faithfulness"]["reasoning"]
            print(f"  ⚠  {r['query_id']}  faithfulness={score:.3f}")
            print(f"     Query  : \"{r['query_text']}\"")
            print(f"     Context: {', '.join(r['context_doc_ids'])}")
            print(f"     Reason : {reasoning}")
            print()
    else:
        print(f"  No queries below the {FAITHFULNESS_ALERT} threshold.  All responses appear faithful.")

    # ══════════════════════════════════════════════════════════════════════════
    # OUTPUT 3: Cross-reference with experiment_5 manual scores
    # ══════════════════════════════════════════════════════════════════════════
    section("Cross-Reference: Manual Scores (Exp 5) vs LLM Judge (Exp 6)")

    cross_ref_rows = []
    if not exp5_scores:
        print("  No experiment_5 results to compare — run experiment_5 first to collect manual scores.")
    else:
        for r in all_results:
            qid = r["query_id"]
            if qid not in exp5_scores:
                continue
            manual  = exp5_scores[qid]
            llm_ov  = r["overall"]
            h_norm  = manual["mean_norm"]    # human score normalised to [0,1]
            disagree = round(abs(h_norm - llm_ov), 3)
            flag    = " ← DISAGREE" if disagree > 0.25 else ""
            cross_ref_rows.append((
                qid,
                f"{manual['mean_1to3']:.2f} / 3",
                f"{h_norm:.3f}",
                f"{llm_ov:.3f}",
                f"{disagree:.3f}",
                flag,
            ))

        if cross_ref_rows:
            h = ["Query", "Human (1-3)", "Human (norm)", "LLM Judge", "Δ", "Flag"]
            print(_table(cross_ref_rows, h))

            n_disagree = sum(1 for row in cross_ref_rows if row[5])
            print(f"\n  Queries with |Δ| > 0.25 : {n_disagree} / {len(cross_ref_rows)}")
            if n_disagree == 0:
                print("  → Good calibration: judge and human are in rough agreement.")
            elif n_disagree == len(cross_ref_rows):
                print("  → Poor calibration: consider using a different/larger judge model.")
            else:
                print("  → Partial agreement: investigate disagreeing queries before trusting judge at scale.")

            print("""
  INTERPRETATION GUIDE
  ─────────────────────────────────────────────────────────────────────────
  • |Δ| < 0.15 → Strong agreement (judge ≈ human)
  • 0.15 ≤ |Δ| < 0.30 → Acceptable gap (within human inter-annotator noise)
  • |Δ| ≥ 0.30 → Significant disagreement — inspect the response manually

  Known sources of disagreement:
  1. Self-preference bias: Gemini judging Gemini output tends to inflate scores.
  2. Dimension mismatch: humans scored Accuracy/Completeness/Grounding;
     the judge scores Faithfulness/Relevance/Precision/Recall — not identical.
  3. Ordinal vs. continuous: a human "2 / 3" maps ambiguously to ~0.50 but
     may intend a wider range depending on annotator calibration.
""")
        else:
            print("  No overlapping queries between experiment_5 and experiment_6.")

    # ══════════════════════════════════════════════════════════════════════════
    # PRODUCTION IMPLICATIONS
    # ══════════════════════════════════════════════════════════════════════════
    print("""
  ══════════════════════════════════════════════════════════════════════════
  PRODUCTION IMPLICATIONS
  ══════════════════════════════════════════════════════════════════════════

  SCALING STRATEGY
  ────────────────
  • Run the LLM judge on a 1–5 % random sample of production traffic daily.
  • Alert on ANY query with faithfulness < 0.70 — these are hallucination risks.
  • Track the overall score as a rolling 7-day metric; regression > 0.05 triggers
    a retrieval pipeline review.

  REDUCING JUDGE COST
  ───────────────────
  • Use a smaller model (Gemini Flash Lite) for Faithfulness only — it's a
    binary check and doesn't need strong reasoning.
  • Cache judge results keyed by (query_hash, context_hash, answer_hash) — the
    same document appearing in multiple queries doesn't need re-judging.
  • Run full 4-dimension judging only on low-confidence retrieval queries
    (those where RRF score spread is small, indicating uncertainty).

  IMPROVING JUDGE CALIBRATION
  ───────────────────────────
  • Collect ~200 human-rated examples, compute Pearson r between human and judge
    scores, apply a calibration curve (isotonic regression) to adjust raw scores.
  • Use a different model family as judge (Claude judging Gemini, or vice versa)
    to eliminate self-preference bias.
  • Add swap-consistency checks: run the judge on the same (query, answer) twice
    with reordered context; flag responses where the two scores differ by > 0.2.
""")

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "experiment":         "experiment_6_llm_judge",
        "generator_model":    GEMINI_MODEL,
        "judge_model":        GEMINI_MODEL,
        "faithfulness_alert": FAITHFULNESS_ALERT,
        "dimensions":         [d["key"] for d in DIMENSIONS],
        "query_results":      all_results,
        "summary": {
            "overall_mean":       overall_mean,
            "dim_means":          {d["key"]: col_mean(d["key"]) for d in DIMENSIONS},
            "faithfulness_flags": [r["query_id"] for r in flagged],
            "n_flagged":          len(flagged),
        },
        "cross_reference": [
            {
                "query_id":    row[0],
                "human_1to3":  row[1],
                "human_norm":  row[2],
                "llm_judge":   row[3],
                "delta":       row[4],
            }
            for row in cross_ref_rows
        ] if cross_ref_rows else [],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
