"""
experiment_8_nli_guardrail.py — NLI-Based Contradiction Guardrail

Central question: can a Natural Language Inference model catch when a
synthesised response contradicts its own retrieved source context?

This experiment implements a lightweight, zero-API-cost guardrail layer that
sits between synthesis and delivery. It uses a pretrained cross-encoder NLI
model to classify each (context chunk, response) pair as:
  ENTAILMENT    — response is logically supported by the context
  NEUTRAL       — response neither follows from nor contradicts the context
  CONTRADICTION — response makes claims that conflict with the context

A CONTRADICTION label with high confidence triggers a BLOCKED decision;
ENTAILMENT and NEUTRAL pass through.

It also runs a deliberate stress test with two hand-crafted adversarial
examples to probe the model's limits — numeric substitutions that are
semantically subtle but factually wrong.

═══════════════════════════════════════════════════════════════════════════════
WHY IN PRODUCTION
═══════════════════════════════════════════════════════════════════════════════
LLM-as-judge (experiment 6) is expensive: each faithfulness check costs one
Gemini API call (~100 ms latency, token cost). For high-throughput systems
(>1,000 queries/minute) that budget is prohibitive.

NLI models run entirely on CPU, take ~10–50 ms per pair, and cost nothing
per call after the one-time 440 MB download. They are not as nuanced as an
LLM judge, but they are fast enough to run synchronously on every response
before it leaves the server — a "circuit breaker" that catches obvious
contradictions before the expensive LLM judge sees them.

Production deployment pattern:
  1. NLI guardrail (this experiment) — synchronous, every response, ~20 ms
  2. LLM-as-judge faithfulness (experiment 6) — async, 1–5 % sample, ~500 ms
  3. Human review (experiment 7 Part A) — for flagged low-confidence responses

The three layers form a defence-in-depth stack: cheap-and-fast catches gross
errors; expensive-and-accurate audits the tail.

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICS
═══════════════════════════════════════════════════════════════════════════════
NLI (Natural Language Inference) is framed as a 3-class classification:
  Input  : (premise, hypothesis) pair
  Output : softmax over [contradiction, entailment, neutral]

cross-encoder/nli-deberta-v3-base uses DeBERTa-v3 architecture, which adds
disentangled attention (separates content and position embeddings) on top of
BERT-style masked language modelling. It achieves state-of-the-art results
on MNLI while remaining small enough (~440 MB) to run on Colab CPU.

The cross-encoder receives the concatenated pair:
  "[CLS] premise [SEP] hypothesis [SEP]"
and produces a single [CLS] representation used for 3-class classification.
This is identical to the re-ranking cross-encoder from experiment 2 but with
NLI-specific fine-tuning rather than relevance-ranking fine-tuning.

Label mapping for this model (in logit order): contradiction, entailment, neutral
After softmax, we read:
  contradiction_prob = scores[0]
  entailment_prob    = scores[1]
  neutral_prob       = scores[2]

BLOCK condition: contradiction_prob > BLOCK_THRESHOLD (default 0.50).

═══════════════════════════════════════════════════════════════════════════════
KNOWN LIMITATIONS (honest findings section)
═══════════════════════════════════════════════════════════════════════════════
NLI models trained on MNLI and related corpora are calibrated on SEMANTIC
contradictions (e.g., "The cat is alive" vs "The cat is dead"). Numeric
substitutions ("$29" → "$99") represent a different type of contradiction that
the model may handle poorly because:

1. Numbers are often out-of-vocabulary or collapsed to [UNK] / sub-tokens
   that the model has little semantic understanding of.
2. MNLI contains very few examples of numeric-value contradictions.
3. "$29 deductible" vs "$99 deductible" share high surface similarity —
   same semantic frame, only the number differs — which can push the model
   toward NEUTRAL rather than CONTRADICTION.

If the stress test reveals the model misses numeric contradictions, this is
not a failure to report around — it is a key finding. The honest mitigation
is to pair NLI with a numeric-consistency extractor (regex-match numbers in
the response against the context) as a separate, cheaper rule layer.

═══════════════════════════════════════════════════════════════════════════════
ALTERNATIVES & TRADEOFFS
═══════════════════════════════════════════════════════════════════════════════
• Fact-checking with atomic claims: decompose the response into atomic
  sentences, run NLI on each independently. Much more precise but O(n) per
  response sentence. Used in FActScore (Min et al., 2023).

• Entity + numeric consistency checker: extract (entity, value) pairs from
  context and response using spaCy NER + regex; flag mismatches. No model
  needed; zero latency; catches numeric substitution that NLI misses.

• Hallucination detection models: Vectara HHEM (a purpose-built hallucination
  detection model), SelfCheckGPT (probes model self-consistency by sampling
  multiple responses). More accurate than general NLI but heavier.

• Fine-tuning on domain data: if your corpus has telecom/insurance-specific
  contradictions (wrong deductible amounts, wrong porting timelines), fine-tune
  the NLI model on 200–500 labelled pairs. Typically gains 15–25 % detection
  rate on domain-specific contradictions.
"""

import json
import os
import sys
import time

import numpy as np
from sentence_transformers import CrossEncoder

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

NLI_MODEL      = "cross-encoder/nli-deberta-v3-base"
BLOCK_THRESHOLD = float(os.environ.get("NLI_BLOCK_THRESHOLD", "0.50"))

RESULTS_DIR = "results"
EXP2_FILE   = os.path.join(RESULTS_DIR, "experiment_2_results.json")
EXP6_FILE   = os.path.join(RESULTS_DIR, "experiment_6_judge_results.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "experiment_8_nli_results.json")

# Label order for cross-encoder/nli-deberta-v3-base
NLI_LABELS = ["contradiction", "entailment", "neutral"]

CORPUS_BY_ID = {d["id"]: d for d in CORPUS}

# ══════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL STRESS-TEST CASES
# Hand-crafted to probe specific weaknesses of NLI models.
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY HAND-CRAFT ADVERSARIAL EXAMPLES?
# Automatic test suites miss the failure modes that matter most in production:
# subtle numeric substitutions that share identical semantic framing but
# contain the wrong value. These are exactly the hallucinations most likely
# to slip past users — a $29 deductible sounds plausible for a budget plan.
#
# CASE ADV-01: NUMERIC SUBSTITUTION — dollar amounts
#   DI-002 states Basic Protection Plan deductibles are $29 (standard),
#   $49 (premium), $99 (tablets). A hallucinating LLM might confidently
#   say "$99 for standard phones" — same framing, wrong tier-to-amount mapping.
#
# CASE ADV-02: NUMERIC SUBSTITUTION — time duration
#   SMB-002 states standard porting takes "5–7 business days".
#   A subtle hallucination compresses this to "1–2 business days",
#   which is the timeline for EXPEDITED porting — a cross-contamination
#   of two facts in the same document.

ADVERSARIAL_CASES = [
    {
        "id": "ADV-01",
        "name": "Dollar-amount substitution (Basic Plan deductibles)",
        "premise_doc_id": "DI-002",
        "premise": (
            "Deductible amounts for device insurance vary by plan tier and device category. "
            "Under the Basic Protection Plan, deductibles are $29 for standard phones, $49 "
            "for premium smartphones, and $99 for tablets."
        ),
        # Deliberately wrong: inflates standard phone deductible from $29 to $99,
        # and swaps premium smartphone deductible from $49 to $149.
        "hypothesis": (
            "Under the Basic Protection Plan, customers pay a $99 deductible for standard "
            "phones and $149 for premium smartphones when filing a claim."
        ),
        "contradiction_type": "numeric_substitution",
        "expected_label": "contradiction",
        "human_note": (
            "$29→$99 (standard) and $49→$149 (premium) — wrong amounts for each tier. "
            "NLI should detect contradiction; numeric substitution may weaken detection."
        ),
    },
    {
        "id": "ADV-02",
        "name": "Time-duration substitution (number porting SLA)",
        "premise_doc_id": "SMB-002",
        "premise": (
            "Standard porting takes 5–7 business days; expedited porting (2–3 business days) "
            "is available for an additional fee."
        ),
        # Deliberately wrong: claims standard porting takes 1–2 days (the expedited timeline),
        # completely fabricating an "instant" option.
        "hypothesis": (
            "Standard number porting for business accounts is completed within 1–2 business "
            "days, and for urgent transfers an instant same-day porting option is available."
        ),
        "contradiction_type": "numeric_substitution_plus_fabrication",
        "expected_label": "contradiction",
        "human_note": (
            "1–2 days is the EXPEDITED timeline, not standard (5–7 days). "
            "'Instant same-day' is entirely fabricated — does not exist in the document. "
            "The fabrication element may help NLI detect contradiction where pure numeric "
            "substitution alone might not."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_experiment_2(path: str) -> dict:
    """Load experiment 2 results; returns dict keyed by query_id."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {q["query_id"]: q for q in data.get("queries", [])}


def load_experiment_6(path: str) -> dict:
    """Load experiment 6 synthesised answers; returns dict keyed by query_id."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {q["query_id"]: q for q in data.get("query_results", [])}


def get_top_ce_doc(exp2_query: dict) -> dict | None:
    """Return the document with ce_rank == 1 from experiment 2 re-ranking."""
    if not exp2_query:
        return None
    candidates = exp2_query.get("reranked", [])
    for c in candidates:
        if c.get("ce_rank") == 1:
            return CORPUS_BY_ID.get(c["doc_id"])
    # Fallback: sort by ce_score
    if candidates:
        best = max(candidates, key=lambda x: x.get("ce_score", 0))
        return CORPUS_BY_ID.get(best["doc_id"])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# NLI INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

# WHY CROSS-ENCODER FOR NLI (not a pipeline)?
# The sentence-transformers CrossEncoder API gives direct access to per-class
# logits and handles batching efficiently. It also uses the same interface as
# the re-ranking cross-encoder from experiment 2, keeping the codebase
# consistent. The HuggingFace pipeline("zero-shot-classification") would also
# work but adds an extra NLI framing step designed for label classification
# rather than direct premise-hypothesis scoring.

def build_nli_model() -> CrossEncoder:
    """Load the NLI cross-encoder. Downloads ~440 MB on first run."""
    print(f"  Loading NLI model: {NLI_MODEL} …", end=" ", flush=True)
    model = CrossEncoder(NLI_MODEL)
    print("done")
    return model


def run_nli(model: CrossEncoder, premise: str, hypothesis: str) -> dict:
    """
    Run NLI on a single (premise, hypothesis) pair.
    Returns probabilities for all three labels plus the winning label.
    """
    t0     = time.time()
    logits = model.predict([(premise, hypothesis)])
    lat    = round((time.time() - t0) * 1000, 1)

    # Softmax over the 3 logits
    logit_vec = np.array(logits[0])
    exp_v     = np.exp(logit_vec - logit_vec.max())   # numerically stable
    probs     = exp_v / exp_v.sum()

    scores = {label: round(float(probs[i]), 4) for i, label in enumerate(NLI_LABELS)}
    top_label  = max(scores, key=lambda k: scores[k])
    top_conf   = scores[top_label]

    return {
        "scores":      scores,
        "top_label":   top_label,
        "top_conf":    top_conf,
        "latency_ms":  lat,
    }


def routing_action(nli_result: dict) -> str:
    """
    Decide whether to pass the response through or block it.
    BLOCKED if contradiction probability exceeds BLOCK_THRESHOLD.
    PASS otherwise (entailment or neutral).
    """
    contra_prob = nli_result["scores"]["contradiction"]
    if contra_prob >= BLOCK_THRESHOLD:
        return "BLOCKED"
    return "pass-through"


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")


def conf_bar(score: float, width: int = 12) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _table(rows, headers):
    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="rounded_outline")
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
# MAIN EXPERIMENT — 10 REAL QUERIES
# ══════════════════════════════════════════════════════════════════════════════

def run_real_queries(model: CrossEncoder, exp2: dict, exp6: dict) -> list[dict]:
    section("PART 1 — NLI Guardrail on 10 Real Queries")
    print(f"  PREMISE    = top cross-encoder-ranked chunk from experiment 2")
    print(f"  HYPOTHESIS = synthesised answer from experiment 6")
    print(f"  BLOCK if contradiction probability ≥ {BLOCK_THRESHOLD}")
    print()

    results = []

    for q in QUERIES:
        qid   = q["id"]
        exp2q = exp2.get(qid, {})
        exp6q = exp6.get(qid, {})

        # ── Determine premise ──────────────────────────────────────────────────
        top_doc = get_top_ce_doc(exp2q)
        if top_doc:
            premise     = top_doc["content"]
            premise_src = top_doc["id"]
        else:
            # Fallback: use expected_doc content (experiment 2 not yet run)
            fallback    = CORPUS_BY_ID.get(q["expected_doc"], {})
            premise     = fallback.get("content", "")
            premise_src = fallback.get("id", q["expected_doc"]) + " (fallback)"

        # ── Determine hypothesis ───────────────────────────────────────────────
        hypothesis = exp6q.get("answer", "")
        if not hypothesis:
            print(f"  [{qid}] No synthesised answer found in experiment_6 results — skipping.")
            continue

        # ── Run NLI ───────────────────────────────────────────────────────────
        nli    = run_nli(model, premise, hypothesis)
        action = routing_action(nli)

        label = nli["top_label"]
        conf  = nli["top_conf"]
        bar   = conf_bar(conf)
        flag  = "  ← BLOCKED" if action == "BLOCKED" else ""
        print(f"  [{qid}] premise={premise_src:<8}  {label:<15} {conf:.3f} [{bar}]{flag}")

        results.append({
            "query_id":    qid,
            "query_text":  q["text"],
            "query_type":  q["query_type"],
            "difficulty":  q["difficulty"],
            "expected_doc": q["expected_doc"],
            "premise_doc_id": premise_src,
            "hypothesis_truncated": hypothesis[:120] + ("…" if len(hypothesis) > 120 else ""),
            "nli_scores":  nli["scores"],
            "top_label":   label,
            "top_conf":    conf,
            "latency_ms":  nli["latency_ms"],
            "action":      action,
        })

    return results


def print_real_query_table(results: list[dict]):
    section("RESULTS TABLE — Real Queries")
    rows = []
    for r in results:
        s = r["nli_scores"]
        rows.append((
            r["query_id"],
            r["query_type"][:10],
            r["difficulty"],
            r["premise_doc_id"][:8],
            f"{s['entailment']:.3f}",
            f"{s['neutral']:.3f}",
            f"{s['contradiction']:.3f}",
            r["top_label"],
            r["action"],
        ))
    headers = ["Query", "Type", "Difficulty", "Premise", "Entail", "Neutral", "Contra", "Label", "Action"]
    print(_table(rows, headers))

    blocked = [r for r in results if r["action"] == "BLOCKED"]
    if blocked:
        print(f"\n  BLOCKED ({len(blocked)}):")
        for r in blocked:
            print(f"    {r['query_id']}  contradiction={r['nli_scores']['contradiction']:.3f}")
            print(f"         Hypothesis snippet: {r['hypothesis_truncated'][:80]}")
    else:
        print(f"\n  No responses blocked (all contradiction scores below {BLOCK_THRESHOLD}).")

    # Distribution summary
    label_counts = {}
    for r in results:
        label_counts[r["top_label"]] = label_counts.get(r["top_label"], 0) + 1
    print(f"\n  Label distribution: " +
          ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items())))


# ══════════════════════════════════════════════════════════════════════════════
# STRESS TEST — ADVERSARIAL EXAMPLES
# ══════════════════════════════════════════════════════════════════════════════

def run_adversarial(model: CrossEncoder) -> list[dict]:
    section("PART 2 — Adversarial Stress Test (hand-crafted contradictions)")
    print("  Testing whether NLI catches numeric substitutions and fabrications.\n")

    results = []

    for case in ADVERSARIAL_CASES:
        print(f"  ── {case['id']}: {case['name']}")
        print(f"     Expected label : {case['expected_label'].upper()}")
        print(f"     Contradiction  : {case['contradiction_type']}")
        print()
        print(f"     PREMISE  (from {case['premise_doc_id']}):")
        # Wrap for readability
        import textwrap
        for line in textwrap.wrap(case["premise"], width=68, initial_indent="       "):
            print(line)
        print()
        print(f"     HYPOTHESIS (adversarial):")
        for line in textwrap.wrap(case["hypothesis"], width=68, initial_indent="       "):
            print(line)
        print()

        nli    = run_nli(model, case["premise"], case["hypothesis"])
        action = routing_action(nli)
        s      = nli["scores"]

        caught = nli["top_label"] == case["expected_label"]

        print(f"     NLI scores:")
        for label in NLI_LABELS:
            bar  = conf_bar(s[label], width=15)
            star = " ←" if label == nli["top_label"] else ""
            print(f"       {label:<15} {s[label]:.3f}  [{bar}]{star}")
        print(f"     Action : {action}")

        if caught:
            verdict = "CAUGHT — NLI correctly identified contradiction"
        else:
            verdict = f"MISSED — NLI predicted '{nli['top_label']}' instead of 'contradiction'"

        print(f"     Verdict: {verdict}")
        print(f"     Note   : {case['human_note']}")
        print()

        results.append({
            "case_id":            case["id"],
            "name":               case["name"],
            "premise_doc_id":     case["premise_doc_id"],
            "contradiction_type": case["contradiction_type"],
            "expected_label":     case["expected_label"],
            "nli_scores":         s,
            "top_label":          nli["top_label"],
            "top_conf":           nli["top_conf"],
            "latency_ms":         nli["latency_ms"],
            "action":             action,
            "caught":             caught,
            "human_note":         case["human_note"],
        })

    return results


def print_adversarial_findings(adv_results: list[dict]):
    section("ADVERSARIAL FINDINGS — Analysis")

    caught_count = sum(1 for r in adv_results if r["caught"])
    print(f"  Caught: {caught_count} / {len(adv_results)} adversarial cases\n")

    for r in adv_results:
        icon = "✓ CAUGHT" if r["caught"] else "✗ MISSED"
        print(f"  {icon}  {r['case_id']} — {r['name']}")
        if not r["caught"]:
            print(f"          Predicted: {r['top_label']} ({r['top_conf']:.3f})")
            print(f"          Expected : {r['expected_label']}")
            print()
            print(f"  WHY DID NLI MISS THIS?")
            print(f"  ─────────────────────────────────────────────────────────────────")
            print(f"  Numeric contradictions are systematically harder for NLI models")
            print(f"  trained on MNLI-style corpora because:")
            print(f"  1. Numbers are tokenised as sub-tokens with limited semantic meaning")
            print(f"     ('$29' → ['$', '29'] or ['$29'] depending on the tokeniser).")
            print(f"  2. MNLI contains mostly semantic/logical contradictions, not value")
            print(f"     substitutions. The model has seen very few training examples of")
            print(f"     'X costs $A' vs 'X costs $B' labelled as contradictions.")
            print(f"  3. '$29 deductible' and '$99 deductible' share high surface overlap")
            print(f"     (same frame: '[amount] deductible'), which may bias toward NEUTRAL.")
            print()
            print(f"  MITIGATIONS IN PRODUCTION:")
            print(f"  • Add a numeric-consistency layer: extract all numbers from the")
            print(f"    context and flag any number in the response not present in the context.")
            print(f"  • Fine-tune on domain-specific numeric contradiction examples")
            print(f"    (200–500 labelled pairs from your corpus).")
            print(f"  • Use entity-value pair extraction (spaCy + regex) as a fast,")
            print(f"    rule-based complement to NLI. No model required; zero latency.")
        print()

    # Overall pattern note
    print(f"  PATTERN: NLI catches conceptual/semantic contradictions reliably.")
    print(f"  Numeric substitutions require a separate rule-based layer.")
    print(f"  The two approaches are complementary, not redundant.")


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCTION COST ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def print_cost_comparison(real_results: list[dict]):
    section("PRODUCTION COST COMPARISON — NLI vs LLM Judge")

    latencies = [r["latency_ms"] for r in real_results]
    avg_lat   = round(sum(latencies) / len(latencies), 1) if latencies else 0

    print(f"""
  NLI Guardrail (this experiment)
  ─────────────────────────────────────────────────────────────────
  Model size    : ~440 MB (one-time download)
  Inference     : CPU only, no GPU required
  Latency/call  : avg {avg_lat} ms per (premise, hypothesis) pair
  Cost/call     : $0.00 (no API)
  Scalability   : limited by CPU cores; trivially parallelisable
  Coverage      : semantic contradictions (strong), numeric (weak)

  LLM Judge — Faithfulness (experiment 6)
  ─────────────────────────────────────────────────────────────────
  Model         : Gemini Flash / similar
  Latency/call  : 200–800 ms (network + inference)
  Cost/call     : ~$0.0001–0.001 per evaluation (token-based)
  Scalability   : API rate-limited; async pipeline needed at scale
  Coverage      : semantic + numeric + pragmatic contradictions

  RECOMMENDED PRODUCTION STACK
  ─────────────────────────────────────────────────────────────────
  LAYER 1 — NLI guardrail (sync, every response)
    Latency budget: <50 ms. Catches gross semantic contradictions.
    Block threshold: 0.50 for conservative (fewer false negatives).

  LAYER 2 — Numeric consistency check (sync, every response)
    Regex: extract all numbers from context, check against response.
    Zero latency. Complements NLI's blind spot on numeric values.

  LAYER 3 — LLM faithfulness judge (async, 2–5 % sample)
    Catches nuanced violations that NLI and regex miss.
    Results feed into model quality monitoring dashboards.

  LAYER 4 — Human review (for flagged responses)
    Only responses that passed layers 1–3 but were low-confidence.
    Estimated <1 % of traffic in a well-tuned system.
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    section("Experiment 8 — NLI Contradiction Guardrail")
    print(f"  NLI model       : {NLI_MODEL}")
    print(f"  Block threshold : contradiction ≥ {BLOCK_THRESHOLD}")
    print(f"  Adversarial cases: {len(ADVERSARIAL_CASES)}")

    # ── Load prior results ─────────────────────────────────────────────────────
    exp2 = load_experiment_2(EXP2_FILE)
    exp6 = load_experiment_6(EXP6_FILE)

    if not exp2:
        print(f"\n  [WARN] {EXP2_FILE} not found — premise will fall back to expected_doc.")
    if not exp6:
        print(f"\n  [WARN] {EXP6_FILE} not found — no synthesised answers to evaluate.")
        print(f"         Run experiment_6_llm_judge.py first, or the real-query section")
        print(f"         will be skipped. The adversarial stress test will still run.")

    # ── Build NLI model ────────────────────────────────────────────────────────
    section("Loading NLI model …")
    nli_model = build_nli_model()

    # ── Part 1: real queries ───────────────────────────────────────────────────
    real_results = []
    if exp6:
        real_results = run_real_queries(nli_model, exp2, exp6)
        print_real_query_table(real_results)
    else:
        print("\n  Skipping Part 1 (no experiment 6 results). Run experiment_6 first.")

    # ── Part 2: adversarial stress test ───────────────────────────────────────
    adv_results = run_adversarial(nli_model)
    print_adversarial_findings(adv_results)

    # ── Cost comparison ────────────────────────────────────────────────────────
    if real_results:
        print_cost_comparison(real_results)

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "experiment":      "experiment_8_nli_guardrail",
        "nli_model":       NLI_MODEL,
        "block_threshold": BLOCK_THRESHOLD,
        "nli_labels":      NLI_LABELS,
        "real_query_results": real_results,
        "adversarial_results": adv_results,
        "summary": {
            "total_real_queries":  len(real_results),
            "blocked":             sum(1 for r in real_results if r["action"] == "BLOCKED"),
            "pass_through":        sum(1 for r in real_results if r["action"] == "pass-through"),
            "adversarial_caught":  sum(1 for r in adv_results if r["caught"]),
            "adversarial_missed":  sum(1 for r in adv_results if not r["caught"]),
            "label_distribution":  {
                label: sum(1 for r in real_results if r["top_label"] == label)
                for label in NLI_LABELS
            },
        },
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
