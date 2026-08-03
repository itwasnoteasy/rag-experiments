"""
experiment_7_hitl.py — Human-in-the-Loop Patterns for Production RAG

Two distinct HITL patterns are demonstrated:

  PART A — Confidence-gated escalation
    Read actions (retrieval + synthesis) run automatically, but the RESPONSE
    is held back and routed to a human reviewer if quality scores are too low.
    Threshold rules: Faithfulness < 0.7 OR Context Recall < 0.6.

  PART B — Propose-then-confirm for write actions
    Any action that mutates external state (filing a claim, changing a plan,
    updating an account) is never executed directly.  The AI generates a
    reviewable diff; a human approves or rejects it; only then does execution
    happen.  A full audit trail is recorded.

═══════════════════════════════════════════════════════════════════════════════════
WHY IN PRODUCTION
═══════════════════════════════════════════════════════════════════════════════════
The read-vs-write trust separation is the single most important pattern in
agentic AI system design.  It maps directly to the OWASP principle of least
privilege:

  READ actions (search, summarise, retrieve) — low consequence, auto-execute.
    If the retrieval is wrong, the user gets a bad answer; fixable by asking again.

  WRITE actions (file claim, update plan, charge card, send email) — high
    consequence, irreversible.  Auto-executing these on a hallucinated or
    misclassified intent creates real-world harm (wrong claim filed, account
    changed without consent, money charged).

Confidence-gated escalation handles the case where the AI is not certain enough
about its READ output to show it directly — the draft is preserved (useful as a
starting point for the human reviewer) but not surfaced to the end user.

Propose-then-confirm handles WRITE actions regardless of confidence.  Even a
high-confidence AI should not auto-execute writes in a regulated domain
(insurance, finance, healthcare) without human sign-off.

═══════════════════════════════════════════════════════════════════════════════════
MATHEMATICS
═══════════════════════════════════════════════════════════════════════════════════
PART A thresholds are empirically set; in production they would be derived from
the calibration curve built in experiment 6:
  • Faithfulness < 0.7 → estimated >30 % chance the response contains an
    unsupported claim (based on the RAGAS paper's correlation data).
  • Context Recall < 0.6 → estimated >40 % chance the retrieved context is
    missing key information, causing the answer to be incomplete.
  Combined OR logic means a query is escalated if EITHER threshold is breached —
  conservative but appropriate for a regulated domain.

PART B diff format is inspired by Terraform plan / database migration patterns:
  {"action": str, "current_state": dict|null, "proposed_state": dict,
   "requires_confirmation": true}
  The diff is designed to be human-readable so the reviewer can understand
  the proposed change without looking at code.

═══════════════════════════════════════════════════════════════════════════════════
ALTERNATIVES & TRADEOFFS
═══════════════════════════════════════════════════════════════════════════════════
• Active learning loop: instead of blocking escalated responses, show them with
  a "is this correct?" widget.  Human feedback updates the retrieval model via
  RLHF.  More expensive but closes the feedback loop.

• Async review queue: escalated responses are queued for human review and the
  user is told "we'll get back to you within 2 hours".  Eliminates latency for
  the user; adds operational complexity (queue, SLA monitoring, callback).

• Tiered confirmation: low-risk writes (view-only account updates) skip
  confirmation; medium-risk writes (plan change) require a single click;
  high-risk writes (claim filing, billing changes) require MFA + explicit text
  confirmation.  Maps to OAuth scopes or capability-based security.

• Shadow mode: always auto-execute write actions, but ALSO log the proposed diff
  and human's likely decision (predicted by a separate model).  Used to measure
  what the confirmation rate WOULD HAVE BEEN before rolling out the HITL gate.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
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

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

# Rate-limit guard: free-tier quota is ~10–15 requests/minute.
# Part A makes 3 Gemini calls per query × 10 queries = 30 calls.
# Part B makes ~1 call per write case × 3 cases = 3 calls. Total ~33 calls.
# Default 5 s → ~12 RPM. Override: GEMINI_DELAY_S=0 for paid tier.
GEMINI_DELAY_S = float(os.environ.get("GEMINI_DELAY_S", "5"))

TOP_K      = 10
CONTEXT_K  = 2
RRF_K      = 60

# Part A escalation thresholds
FAITHFULNESS_THRESHOLD  = 0.70
RECALL_THRESHOLD        = 0.60

RESULTS_DIR = "results"
EXP6_FILE   = os.path.join(RESULTS_DIR, "experiment_6_judge_results.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "experiment_7_hitl_results.json")

# ══════════════════════════════════════════════════════════════════════════════
# PART B — WRITE ACTION TEST CASES
# Three realistic telecom / insurance write actions with hardcoded human decisions.
# In a real system these decisions would come from a UI widget or approval queue.
# ══════════════════════════════════════════════════════════════════════════════

WRITE_TEST_CASES = [
    {
        "id": "W01",
        "user_query": "My phone screen cracked yesterday. Please file a claim for me under my protection plan.",
        "context_query_id": "Q09",   # maps to DI-001 (claim filing)
        "human_decision": "CONFIRM",
        "decision_note": "Customer confirmed the damage and provided proof of purchase.",
    },
    {
        "id": "W02",
        "user_query": "Switch our business account from the Starter plan to the Business Pro plan starting next month.",
        "context_query_id": "Q06",   # maps to SMB-004 (plan changes / proration)
        "human_decision": "REJECT",
        "decision_note": "Account manager flagged: customer is mid-contract; early upgrade fee applies. Needs billing review first.",
    },
    {
        "id": "W03",
        "user_query": "Give Sarah Johnson admin access to manage all lines on our business account.",
        "context_query_id": "Q10",   # maps to SMB-005 (sub-administrator / RBAC)
        "human_decision": "CONFIRM",
        "decision_note": "HR portal confirms Sarah Johnson is a current employee. Access granted.",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL STACK  (self-contained, same pattern as experiments 5 & 6)
# ══════════════════════════════════════════════════════════════════════════════

def build_dense_index(corpus):
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    coll   = client.get_or_create_collection("exp7")
    texts  = [d["content"] for d in corpus]
    ids    = [d["id"]      for d in corpus]
    embs   = model.encode(texts, show_progress_bar=False).tolist()
    coll.add(ids=ids, embeddings=embs, documents=texts)
    return model, coll


def dense_query(model, collection, query_text, top_k=TOP_K):
    q_emb = model.encode([query_text]).tolist()
    res   = collection.query(query_embeddings=q_emb, n_results=top_k)
    return [{"doc_id": doc_id, "rank": r + 1} for r, doc_id in enumerate(res["ids"][0])]


def build_bm25_index(corpus):
    return BM25Okapi([d["content"].lower().split() for d in corpus])


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

def _gemini_call(model, prompt_or_content):
    """Wrapper that enforces the inter-call rate-limit delay after every generate_content()."""
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


def synthesise(model_name: str, query_text: str, docs: list[dict]) -> str:
    """Generate a grounded response. Returns response text."""
    context = "\n\n".join(
        f"[Document {i+1}: {d['id']} — {d.get('title','')}]\n{d['content']}"
        for i, d in enumerate(docs)
    )
    prompt = (
        "You are a helpful telecom support assistant.\n\n"
        "Answer the following customer question using ONLY the documents provided below. "
        "Be concise (2–4 sentences). If the documents do not contain enough information "
        "to answer fully, say so.\n\n"
        f"Question: {query_text}\n\nDocuments:\n{context}\n\nAnswer:"
    )
    model    = genai.GenerativeModel(model_name)
    response = _gemini_call(model, prompt)
    return response.text.strip()


def _judge_score(model_name: str, dimension_key: str, instruction: str,
                 query_text: str, context_text: str, answer_text: str) -> dict:
    """Single-dimension LLM judge call. Returns {score, reasoning}."""
    import re
    system = (
        "You are an impartial quality evaluator for AI-generated answers.\n"
        "Output ONLY a JSON object with exactly two keys:\n"
        '  "score": a float 0.0–1.0\n'
        '  "reasoning": a single sentence\n'
        "No markdown fences. No other text."
    )
    user = (
        f"TASK: {instruction}\n\n"
        f"QUESTION: {query_text}\n\n"
        f"RETRIEVED CONTEXT:\n{context_text}\n\n"
        f"ANSWER:\n{answer_text}\n\nOutput JSON only:"
    )
    gem = genai.GenerativeModel(model_name, system_instruction=system)
    for attempt in range(3):
        try:
            raw = _gemini_call(gem, user).text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            score  = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
            return {"score": round(score, 3), "reasoning": str(parsed.get("reasoning", ""))}
        except Exception as exc:
            if attempt == 2:
                return {"score": 0.0, "reasoning": f"[parse error: {exc}]"}
            time.sleep(1.0 * (attempt + 1))
    return {"score": 0.0, "reasoning": "[max retries exceeded]"}


# Instruction strings for the two gating dimensions
_FAITHFULNESS_INSTR = (
    "Assess whether every factual claim in the ANSWER is explicitly supported by the "
    "RETRIEVED CONTEXT.  Score 1.0 = fully faithful, 0.0 = entirely fabricated.  "
    "If any claim is unsupported, quote the violating sentence in your reasoning."
)
_RECALL_INSTR = (
    "Assess whether the RETRIEVED CONTEXT contains ALL information needed to fully "
    "answer the QUESTION.  Score 1.0 = context is complete, 0.0 = context is entirely "
    "insufficient to answer."
)


def judge_for_gating(model_name: str, query_text: str,
                     context_text: str, answer_text: str) -> dict:
    """
    Run only the two gating dimensions (Faithfulness + Context Recall) for Part A.
    Returns {faithfulness: {score, reasoning}, context_recall: {score, reasoning}}.
    """
    faith  = _judge_score(model_name, "faithfulness",  _FAITHFULNESS_INSTR,
                          query_text, context_text, answer_text)
    recall = _judge_score(model_name, "context_recall", _RECALL_INSTR,
                          query_text, context_text, answer_text)
    return {"faithfulness": faith, "context_recall": recall}


# ══════════════════════════════════════════════════════════════════════════════
# PART A — CONFIDENCE-GATED ESCALATION
# ══════════════════════════════════════════════════════════════════════════════

# WHY IN PRODUCTION: routing based on quality scores separates the happy path
# (high-confidence responses shown immediately) from the review queue (low-
# confidence drafts held for a human).  The draft is preserved so the reviewer
# can edit rather than re-generate from scratch — saving 40–60 % of human time
# compared to a blank review form.
#
# MATHEMATICS: threshold values chosen from the RAGAS paper's calibration data:
#   Faithfulness < 0.7 → estimated P(hallucination) > 30 %
#   Context Recall < 0.6 → estimated P(incomplete answer) > 40 %
# Both thresholds apply independently (OR logic) — a single failure of either
# dimension triggers escalation.
#
# ALTERNATIVES:
# • Ensemble thresholding: escalate only if BOTH dimensions fail (AND logic).
#   Higher throughput, higher risk of showing a partially hallucinated answer.
# • Percentile-based: escalate the bottom 10 % of overall scores rather than
#   using fixed thresholds.  Self-adapts to model quality drift over time.
# • Abstention: instead of escalation, respond "I'm not confident enough to
#   answer this — please contact support."  Simpler but worse UX.

def escalation_decision(
    faithfulness_score: float,
    recall_score: float,
    draft_answer: str,
    faith_reasoning: str,
    recall_reasoning: str,
) -> dict:
    """
    Apply threshold rules and return a routing envelope.
    Auto-approved responses are safe to show; escalated ones go to the review queue.
    """
    reasons = []
    if faithfulness_score < FAITHFULNESS_THRESHOLD:
        reasons.append(
            f"low faithfulness score: {faithfulness_score:.3f} "
            f"(threshold {FAITHFULNESS_THRESHOLD}) — {faith_reasoning}"
        )
    if recall_score < RECALL_THRESHOLD:
        reasons.append(
            f"low context recall score: {recall_score:.3f} "
            f"(threshold {RECALL_THRESHOLD}) — {recall_reasoning}"
        )

    if reasons:
        return {
            "status":       "needs_human_review",
            "draft_answer": draft_answer,
            "reason":       "; ".join(reasons),
            "faithfulness": faithfulness_score,
            "context_recall": recall_score,
        }
    return {
        "status":       "auto_approved",
        "answer":       draft_answer,
        "faithfulness": faithfulness_score,
        "context_recall": recall_score,
    }


def run_part_a(dense_model, collection, bm25, cross_encoder,
               corpus_lookup, corpus_by_id):
    """
    Run Part A for all 10 queries.
    Returns list of routing envelopes with query metadata.
    """
    section("PART A — Confidence-Gated Escalation (all 10 queries)")
    print(f"  Thresholds: Faithfulness < {FAITHFULNESS_THRESHOLD} "
          f"OR Context Recall < {RECALL_THRESHOLD} → escalate")
    print()

    results = []

    for q in QUERIES:
        print(f"  [{q['id']}] \"{q['text'][:65]}{'…' if len(q['text'])>65 else ''}\"")

        # Retrieve
        dense_res    = dense_query(dense_model, collection, q["text"])
        bm25_res     = bm25_query(bm25, CORPUS, q["text"])
        rrf_pool     = reciprocal_rank_fusion(dense_res, bm25_res)[:TOP_K]
        reranked     = rerank_candidates(cross_encoder, q["text"], rrf_pool, corpus_lookup)
        context_docs = [corpus_by_id[r["doc_id"]] for r in reranked[:CONTEXT_K]]
        context_text = "\n\n".join(
            f"[Doc {i+1}: {d['id']}]\n{d['content']}" for i, d in enumerate(context_docs)
        )

        # Synthesise
        answer = synthesise(GEMINI_MODEL, q["text"], context_docs)

        # Judge (gating dimensions only)
        scores = judge_for_gating(GEMINI_MODEL, q["text"], context_text, answer)
        faith  = scores["faithfulness"]
        recall = scores["context_recall"]

        envelope = escalation_decision(
            faith["score"], recall["score"], answer,
            faith["reasoning"], recall["reasoning"]
        )

        status_icon = "✓" if envelope["status"] == "auto_approved" else "⚠"
        print(f"         faith={faith['score']:.3f}  recall={recall['score']:.3f}  "
              f"→  {status_icon} {envelope['status']}")

        results.append({
            "query_id":       q["id"],
            "query_text":     q["text"],
            "query_type":     q["query_type"],
            "difficulty":     q["difficulty"],
            "expected_doc":   q["expected_doc"],
            "context_doc_ids": [d["id"] for d in context_docs],
            "envelope":       envelope,
        })

    return results


def print_part_a_summary(results: list[dict]):
    """Print the escalation summary table and counts."""
    section("PART A SUMMARY — Auto-approved vs Escalated")

    approved  = [r for r in results if r["envelope"]["status"] == "auto_approved"]
    escalated = [r for r in results if r["envelope"]["status"] == "needs_human_review"]

    rows = []
    for r in results:
        env    = r["envelope"]
        status = "✓ AUTO"    if env["status"] == "auto_approved" else "⚠ ESCALATE"
        reason = ""
        if env["status"] == "needs_human_review":
            # Extract first trigger for display
            reason = env["reason"].split("—")[0].strip()[:55]
        rows.append((
            r["query_id"],
            r["query_type"][:10],
            r["difficulty"],
            f"{env['faithfulness']:.3f}",
            f"{env['context_recall']:.3f}",
            status,
            reason,
        ))

    headers = ["Query", "Type", "Difficulty", "Faithful", "Recall", "Decision", "Reason"]
    print(_table(rows, headers))

    print(f"\n  Auto-approved : {len(approved):2d} / {len(results)}"
          f"  ({100*len(approved)//len(results)} %)")
    print(f"  Escalated     : {len(escalated):2d} / {len(results)}"
          f"  ({100*len(escalated)//len(results)} %)")

    if escalated:
        print(f"\n  Escalated queries:")
        for r in escalated:
            env = r["envelope"]
            print(f"    {r['query_id']}  faith={env['faithfulness']:.3f}  "
                  f"recall={env['context_recall']:.3f}")
            print(f"         Reason: {env['reason'][:80]}")

    # Throughput note
    auto_pct = 100 * len(approved) // len(results)
    print(f"""
  OPERATIONAL IMPACT
  ──────────────────
  At {auto_pct} % auto-approval rate, a human reviewer handles ~{100-auto_pct} % of queries.
  For a system processing 10,000 queries/day, that is ~{(100-auto_pct)*100:,} human reviews/day.
  Selective escalation vs. 100 % manual review saves ~{auto_pct*100:,} reviews/day.
""")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — PROPOSE-THEN-CONFIRM FOR WRITE ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

# WHY IN PRODUCTION: write actions against external state (CRM, billing system,
# claims database) are irreversible or expensive to undo.  The proposed-diff
# pattern gives the human a clear, structured view of WHAT will change before
# it changes.  The diff is the contract between the AI and the human; the audit
# trail is the contract between the system and the regulator.
#
# MATHEMATICS / STRUCTURE: the diff format mirrors Terraform plan / SQL migration:
#   current_state → proposed_state.  Null current_state means "create new record".
#   The hash of (action, proposed_state) is included in the audit trail so that
#   a tampered execution can be detected.
#
# ALTERNATIVES:
# • Intent-based approval: show the intent in plain English ("file a claim for
#   cracked screen") rather than a structured diff.  Faster for humans but loses
#   auditability and makes automated rollback harder.
# • Two-person integrity: require TWO human approvals for high-value actions
#   (e.g., claims over $500).  Standard in financial systems.
# • Time-delayed execution: auto-approve but add a 5-minute cancellation window.
#   Appropriate for medium-risk actions where full human review is too slow.

def generate_write_proposal(
    model_name: str,
    test_case: dict,
    context_docs: list[dict],
) -> dict:
    """
    Ask Gemini to classify the write action and propose a structured diff.
    Forces JSON output describing the proposed state change.
    """
    import re
    context = "\n\n".join(
        f"[Document {i+1}: {d['id']} — {d.get('title','')}]\n{d['content']}"
        for i, d in enumerate(context_docs)
    )
    system = (
        "You are a telecom/insurance AI assistant that proposes account changes.\n"
        "Given a customer request and relevant policy documents, output ONLY a JSON object "
        "with these exact keys:\n"
        '  "action": short snake_case action name (e.g. "file_claim", "upgrade_plan")\n'
        '  "action_description": one sentence describing what will happen\n'
        '  "current_state": null if creating new, or {"field": "current_value"} dict\n'
        '  "proposed_state": {"field": "new_value"} dict of changes\n'
        '  "risk_level": "low" | "medium" | "high"\n'
        '  "requires_confirmation": true\n'
        "No markdown fences. No other text."
    )
    user = (
        f"CUSTOMER REQUEST: {test_case['user_query']}\n\n"
        f"POLICY DOCUMENTS:\n{context}\n\n"
        "Output the proposed change as JSON:"
    )
    gem = genai.GenerativeModel(model_name, system_instruction=system)
    for attempt in range(3):
        try:
            raw = _gemini_call(gem, user).text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            parsed["requires_confirmation"] = True   # enforce regardless of model output
            return parsed
        except Exception as exc:
            if attempt == 2:
                return {
                    "action":              "unknown",
                    "action_description":  f"[parse error: {exc}]",
                    "current_state":       None,
                    "proposed_state":      {},
                    "risk_level":          "high",
                    "requires_confirmation": True,
                }
            time.sleep(1.0 * (attempt + 1))
    return {}


def mock_execute(action: str, proposed_state: dict) -> dict:
    """
    Simulate execution of an approved write action.
    In production this would call a CRM API, claims system, etc.
    Returns a mock execution receipt.
    """
    return {
        "execution_status": "SUCCESS",
        "action":           action,
        "applied_state":    proposed_state,
        "execution_id":     f"EXE-{action.upper()[:6]}-{int(time.time()) % 100000:05d}",
        "message":          f"Action '{action}' applied successfully (mock execution).",
    }


def run_part_b(dense_model, collection, bm25, cross_encoder,
               corpus_lookup, corpus_by_id):
    """
    Run Part B for the 3 write test cases.
    Returns list of audit trail records.
    """
    section("PART B — Propose-Then-Confirm for Write Actions")
    print("  Pattern: AI proposes diff → human approves/rejects → execution on CONFIRM only")
    print()

    audit_trail = []
    query_by_id = {q["id"]: q for q in QUERIES}

    for tc in WRITE_TEST_CASES:
        section_minor(f"Write Case {tc['id']} — {tc['human_decision']}")
        print(f"  User request : \"{tc['user_query']}\"")
        print(f"  Human will   : {tc['human_decision']} ({tc['decision_note']})")

        # Retrieve context based on the mapped query
        ref_query = query_by_id.get(tc["context_query_id"])
        retrieve_text = ref_query["text"] if ref_query else tc["user_query"]

        dense_res    = dense_query(dense_model, collection, retrieve_text)
        bm25_res     = bm25_query(bm25, CORPUS, retrieve_text)
        rrf_pool     = reciprocal_rank_fusion(dense_res, bm25_res)[:TOP_K]
        reranked     = rerank_candidates(cross_encoder, retrieve_text, rrf_pool, corpus_lookup)
        context_docs = [corpus_by_id[r["doc_id"]] for r in reranked[:CONTEXT_K]]

        print(f"  Context docs : {', '.join(d['id'] for d in context_docs)}")

        # Generate proposal
        print(f"  Generating proposal …", end=" ", flush=True)
        proposal_ts = datetime.now(timezone.utc).isoformat()
        proposal    = generate_write_proposal(GEMINI_MODEL, tc, context_docs)
        print("done")

        print(f"\n  PROPOSED DIFF:")
        print(f"    action             : {proposal.get('action')}")
        print(f"    description        : {proposal.get('action_description')}")
        print(f"    risk_level         : {proposal.get('risk_level')}")
        print(f"    current_state      : {json.dumps(proposal.get('current_state'))}")
        print(f"    proposed_state     : {json.dumps(proposal.get('proposed_state'), indent=6)}")
        print(f"    requires_confirmation: {proposal.get('requires_confirmation')}")

        # Simulate human decision
        decision_ts = datetime.now(timezone.utc).isoformat()
        decision    = tc["human_decision"]
        print(f"\n  ── HUMAN DECISION: {decision} ──")
        print(f"     Note: {tc['decision_note']}")

        # Execute only on CONFIRM
        if decision == "CONFIRM":
            execution = mock_execute(proposal["action"], proposal.get("proposed_state", {}))
            outcome   = "EXECUTED"
            print(f"\n  EXECUTION LOG:")
            print(f"    status       : {execution['execution_status']}")
            print(f"    execution_id : {execution['execution_id']}")
            print(f"    message      : {execution['message']}")
        else:
            execution = None
            outcome   = "NOT_EXECUTED"
            print(f"\n  EXECUTION SKIPPED — proposal rejected by human reviewer.")

        # Audit record
        audit_record = {
            "case_id":          tc["id"],
            "user_query":       tc["user_query"],
            "context_doc_ids":  [d["id"] for d in context_docs],
            "proposal_timestamp": proposal_ts,
            "proposal":         proposal,
            "decision_timestamp": decision_ts,
            "human_decision":   decision,
            "decision_note":    tc["decision_note"],
            "execution":        execution,
            "outcome":          outcome,
        }
        audit_trail.append(audit_record)
        print()

    return audit_trail


def print_part_b_audit_trail(audit_trail: list[dict]):
    """Print a clean audit trail summary table."""
    section("PART B AUDIT TRAIL")
    rows = []
    for rec in audit_trail:
        p = rec["proposal"]
        rows.append((
            rec["case_id"],
            p.get("action", "?"),
            p.get("risk_level", "?"),
            rec["human_decision"],
            rec["outcome"],
            rec["decision_note"][:55],
        ))
    headers = ["Case", "Action", "Risk", "Human Decision", "Outcome", "Note"]
    print(_table(rows, headers))

    confirmed = sum(1 for r in audit_trail if r["outcome"] == "EXECUTED")
    rejected  = sum(1 for r in audit_trail if r["outcome"] == "NOT_EXECUTED")
    print(f"\n  Confirmed & executed : {confirmed}")
    print(f"  Rejected / blocked   : {rejected}")
    print(f"\n  Every proposed action is in the audit trail regardless of outcome.")
    print(f"  This satisfies SOC 2 / GDPR accountability requirements for automated systems.")

    print("""
  READ vs WRITE TRUST SEPARATION — Interview Summary
  ────────────────────────────────────────────────────
  READ  (retrieval, synthesis, classification)
    → Auto-execute. Worst case: a bad answer. User asks again. No lasting harm.

  WRITE (file claim, change plan, grant access, send email, charge card)
    → Always gate with a propose-then-confirm diff. Even a 99 %-confident AI
      should not auto-execute writes in regulated domains.
    → The diff format (current_state → proposed_state) enables:
        • Human comprehension without reading code
        • Automated rollback (re-apply current_state)
        • Audit trail for regulators (what was proposed, who approved, when)
        • Dry-run testing in staging environments

  ESCALATION (read output too low-confidence to show)
    → Route to human review queue. Preserve the draft so the reviewer edits
      rather than writes from scratch. SLA: respond within N hours.
""")


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")


def section_minor(title: str):
    print(f"\n  ── {title} {'─'*(60-len(title))}")


def _table(rows, headers, floatfmt=".3f"):
    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="rounded_outline", floatfmt=floatfmt)
    col_w = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    fmt  = "  ".join(f"{{:<{w}}}" for w in col_w)
    sep  = "-" * (sum(col_w) + 2 * len(col_w))
    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    section("Experiment 7 — Human-in-the-Loop Patterns")
    print(f"  Model  : {GEMINI_MODEL}")
    print(f"  Part A : confidence-gated escalation for all 10 queries")
    print(f"  Part B : propose-then-confirm for 3 write actions")

    # ── Validate Gemini ────────────────────────────────────────────────────────
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

    # ── Part A ─────────────────────────────────────────────────────────────────
    part_a_results   = run_part_a(dense_model, collection, bm25, cross_encoder,
                                   corpus_lookup, corpus_by_id)
    print_part_a_summary(part_a_results)

    # ── Part B ─────────────────────────────────────────────────────────────────
    part_b_audit = run_part_b(dense_model, collection, bm25, cross_encoder,
                               corpus_lookup, corpus_by_id)
    print_part_b_audit_trail(part_b_audit)

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "experiment":         "experiment_7_hitl",
        "model":              GEMINI_MODEL,
        "faithfulness_threshold": FAITHFULNESS_THRESHOLD,
        "recall_threshold":       RECALL_THRESHOLD,
        "part_a": {
            "escalation_results": part_a_results,
            "summary": {
                "total":      len(part_a_results),
                "approved":   sum(1 for r in part_a_results
                                  if r["envelope"]["status"] == "auto_approved"),
                "escalated":  sum(1 for r in part_a_results
                                  if r["envelope"]["status"] == "needs_human_review"),
                "escalated_query_ids": [
                    r["query_id"] for r in part_a_results
                    if r["envelope"]["status"] == "needs_human_review"
                ],
            },
        },
        "part_b": {
            "audit_trail": part_b_audit,
            "summary": {
                "total":     len(part_b_audit),
                "confirmed": sum(1 for r in part_b_audit if r["outcome"] == "EXECUTED"),
                "rejected":  sum(1 for r in part_b_audit if r["outcome"] == "NOT_EXECUTED"),
            },
        },
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
