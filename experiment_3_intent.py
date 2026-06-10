"""
Experiment 3: Zero-Shot Intent Classification and Retrieval Routing
====================================================================

WHY THIS EXPERIMENT EXISTS IN PRODUCTION
-----------------------------------------
A single vector index over an entire knowledge base works for small corpora,
but real enterprise systems partition their data into namespaces or separate
indexes by product line, region, or topic (e.g., a separate index for billing
documents vs. claims documents). Routing a query to the wrong index returns
irrelevant results regardless of how good your retriever is. Intent
classification solves this: classify the query first, then send it only to
the relevant namespace. This reduces retrieval latency (searching 500 docs
instead of 5000), increases precision (no cross-contamination between unrelated
topic clusters), and enables topic-specific tuning (different retrieval
parameters per intent class). Teams at Salesforce, Zendesk, and most large
telecom/insurance operators classify customer intent before routing to their
knowledge sub-systems for exactly these reasons.

A second production use case is escalation routing: if the classifier is
confident the query is about claim filing, route to the claims team bot; if
it is about billing, route to the billing team. The confidence threshold
controls whether you trust the classifier or fall back to a human agent.

Run:
    python experiment_3_intent.py

Outputs:
    - Per-query classification results and routing decisions to stdout
    - Threshold analysis table to stdout
    - results/experiment_3_results.json
"""

import json
import os

from transformers import pipeline
from tabulate import tabulate

from queries import QUERIES

RESULTS_DIR  = "results"
OUTPUT_FILE  = os.path.join(RESULTS_DIR, "experiment_3_results.json")

# Confidence threshold used for primary routing simulation
PRIMARY_THRESHOLD = 0.70

# Thresholds to sweep in section 4
THRESHOLDS = [0.50, 0.60, 0.70, 0.80]


# =============================================================================
# SECTION 1 — INTENT CLASS DEFINITIONS
# =============================================================================
#
# WHY IN PRODUCTION
# -----------------
# Intent classes are the contract between the classifier and the retrieval
# router. Poorly defined classes lead to overlapping predictions that the model
# cannot distinguish — a common failure mode when teams add "Other" as a catch-
# all instead of modelling low-confidence queries explicitly. The four classes
# here are deliberately fine-grained within the same product family (device
# insurance and SMB) to stress-test the classifier on queries that share
# vocabulary. In production you would derive these classes from your support
# ticket taxonomy or call centre disposition codes, ensuring the classes map
# directly to actionable routing targets.
#
# MATHEMATICS / TECHNICAL MECHANISM
# ----------------------------------
# facebook/bart-large-mnli is a BART model fine-tuned on the Multi-Genre
# Natural Language Inference (MultiNLI) corpus. Zero-shot classification
# reframes intent detection as a textual entailment problem:
#
#   Premise  : the user's query
#   Hypothesis: "This text is about {label}"
#
# The model outputs three probabilities — entailment, neutral, contradiction.
# The entailment probability for each hypothesis becomes that label's score.
# Scores across all labels are then softmax-normalised so they sum to 1.
#
# This approach requires zero labelled training examples for your domain.
# It generalises because MNLI training covered diverse reasoning patterns,
# and the entailment framing lets the model apply that reasoning to any
# label description you provide.
#
# ALTERNATIVES & TRADEOFFS
# -------------------------
# • facebook/bart-large-mnli (this experiment)
#     Pro: free, runs locally, no labelled data needed, solid baseline.
#     Con: 400M parameters — slow on CPU; accuracy drops when label descriptions
#          are wordy or overlapping; English-only.
#
# • cross-encoder/nli-deberta-v3-small
#     Pro: smaller (184M), faster, often better accuracy than BART on NLI tasks.
#     Con: still requires crafting clear hypothesis strings; English-primary.
#
# • OpenAI GPT-4o with structured outputs / function calling
#     Pro: best zero-shot accuracy; can return a JSON intent + confidence;
#          handles ambiguous and multi-intent queries gracefully.
#     Con: $0.005–$0.015 per query adds up; 200–800ms latency; data privacy.
#
# • Fine-tuned BERT classifier (e.g. bert-base-uncased + classification head)
#     Pro: very fast inference (~5ms on GPU), highest accuracy when you have
#          500+ labelled examples per class, deterministic output.
#     Con: requires labelled training data (expensive to collect); retraining
#          every time you add a new intent class; brittle to distribution shift.
#
# • Cohere Classify API
#     Pro: few-shot — needs only 2–5 examples per class, not hundreds; fast;
#          managed, no hosting burden.
#     Con: per-call cost; data leaves your environment; limited to text inputs.
#
# • SetFit (sentence-transformers + logistic regression)
#     Pro: achieves fine-tuned accuracy with as few as 8 labelled examples
#          per class using contrastive learning; very fast inference after
#          training; open-source.
#     Con: still requires some labelled data; training loop adds operational
#          complexity compared to pure zero-shot.
#
# LABEL DESIGN TRADEOFF
# ----------------------
# Zero-shot accuracy is highly sensitive to how you phrase the hypothesis.
# "This text is about filing an insurance claim" will outperform a generic
# label like "DEVICE_INSURANCE_CLAIM". The descriptions below are written
# to be maximally distinctive — each includes unique trigger words that the
# model can latch onto.

INTENT_CLASSES = [
    {
        "label":       "DEVICE_INSURANCE_CLAIM",
        "description": "filing, checking status, or processing a device insurance claim",
        "namespace":   "device_insurance_claims",
        "doc_prefix":  "DI-00",   # DI-001 to DI-009 (claim-process docs)
        "category":    "device_insurance",
    },
    {
        "label":       "DEVICE_INSURANCE_POLICY",
        "description": "coverage rules, exclusions, deductibles, or policy terms for device insurance",
        "namespace":   "device_insurance_policy",
        "doc_prefix":  "DI-0",    # broader — DI-002, DI-003, DI-005 etc.
        "category":    "device_insurance",
    },
    {
        "label":       "SMB_ONBOARDING",
        "description": "setting up a business account, porting numbers, activating lines, or accessing the admin portal",
        "namespace":   "smb_onboarding",
        "doc_prefix":  "SMB-",
        "category":    "smb_onboarding",
    },
    {
        "label":       "SMB_BILLING",
        "description": "business billing cycles, invoices, payments, or data overage charges",
        "namespace":   "smb_billing",
        "doc_prefix":  "SMB-004",  # specifically SMB-004 billing doc
        "category":    "smb_onboarding",
    },
]

# Map label → intent class dict for fast lookup
INTENT_MAP  = {ic["label"]: ic for ic in INTENT_CLASSES}
LABEL_TEXTS = [ic["description"] for ic in INTENT_CLASSES]
LABEL_NAMES = [ic["label"]       for ic in INTENT_CLASSES]

# Ground-truth category derived from expected_doc prefix
def expected_category(q):
    return "device_insurance" if q["expected_doc"].startswith("DI") else "smb_onboarding"


# =============================================================================
# SECTION 2 — CLASSIFIER SETUP AND INFERENCE
# =============================================================================
#
# WHY IN PRODUCTION
# -----------------
# Loading the model once and calling it in batch is critical for production
# throughput. A single model load on GPU takes ~2–5 seconds; per-query loads
# would make the system unusable. In production this model would be served
# via a model-serving framework (TorchServe, Triton, or a managed endpoint
# like HuggingFace Inference Endpoints or SageMaker) that keeps the model
# warm and batches concurrent requests.
#
# TECHNICAL DETAIL — MULTI_CLASS vs MULTI_LABEL
# ----------------------------------------------
# pipeline(..., multi_label=False) applies softmax across all candidate labels
# so scores sum to 1 — the model must pick one dominant class. This is correct
# for routing decisions where you need a single destination.
# multi_label=True applies sigmoid independently per label — each score is
# independent of the others, allowing a query to score high on multiple labels
# simultaneously. Use multi_label=True when a single query can genuinely
# belong to multiple categories and you want to route to all of them.
# For routing, multi_label=False is almost always the right choice.

def build_classifier():
    print("=== Loading zero-shot classifier (facebook/bart-large-mnli) ===")
    clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        multi_label=False,   # softmax — scores sum to 1, pick one routing target
    )
    print("  Model loaded.\n")
    return clf


def classify_query(clf, query_text):
    """
    Runs zero-shot classification. Internally the model evaluates:
      P(entailment | "[query_text]", "This text is about [description]")
    for each candidate description, then softmax-normalises across all labels.
    Returns label names and scores sorted by score descending.
    """
    result = clf(query_text, candidate_labels=LABEL_TEXTS)

    # Map description strings back to label names
    desc_to_label = {ic["description"]: ic["label"] for ic in INTENT_CLASSES}
    ranked = [
        {
            "label": desc_to_label[desc],
            "score": round(score, 4),
        }
        for desc, score in zip(result["labels"], result["scores"])
    ]
    return ranked   # already sorted descending by score


# =============================================================================
# SECTION 3 — ROUTING DECISION LOGIC
# =============================================================================
#
# WHY IN PRODUCTION
# -----------------
# The confidence threshold is the single most important tunable parameter in
# an intent-routing system. Setting it correctly is a precision/recall tradeoff:
#
#   Low threshold (e.g. 0.50): the classifier routes most queries to a specific
#     namespace → high specificity, reduced search space, lower latency.
#     Risk: misclassified queries go to the wrong namespace and get bad results,
#     which may never surface the correct answer (a silent failure — worse than
#     a "I don't know").
#
#   High threshold (e.g. 0.90): only very confident queries get routed;
#     most queries fall back to full-corpus search → safer but loses the
#     latency and precision benefits of namespacing.
#
# In production, the threshold is not a global setting — it is tuned per intent
# class. A claim-filing intent with 0.95 confidence is very reliable; an
# ambiguous billing/policy query at 0.65 should go to full corpus. Teams use
# offline eval data (labelled query sets) to find the threshold that maximises
# precision@k for each class independently.
#
# SIMULATED NAMESPACE ROUTING
# ---------------------------
# In a real system this function would call:
#   chromadb_client.query(collection=namespace, ...)
# Here we simulate it by printing the decision, which is sufficient to
# understand the routing logic without needing the full index running.

def routing_decision(ranked_classes, threshold=PRIMARY_THRESHOLD):
    top = ranked_classes[0]
    if top["score"] >= threshold:
        intent_class = INTENT_MAP[top["label"]]
        return {
            "decision":    "SPECIFIC_NAMESPACE",
            "namespace":   intent_class["namespace"],
            "label":       top["label"],
            "confidence":  top["score"],
            "description": f"Route to '{intent_class['namespace']}' index only",
        }
    else:
        return {
            "decision":    "FULL_CORPUS",
            "namespace":   "all",
            "label":       top["label"],
            "confidence":  top["score"],
            "description": "Confidence below threshold — search full corpus",
        }


def routing_correct(route, q):
    """
    A routing decision is 'correct' if the predicted category matches the
    ground-truth category of the expected document. For FULL_CORPUS decisions
    this is always considered correct (we didn't misroute — we hedged).
    For SPECIFIC_NAMESPACE decisions we check whether the routed namespace
    covers the expected document's category.
    """
    if route["decision"] == "FULL_CORPUS":
        return True   # full corpus never excludes the right answer
    predicted_category = INTENT_MAP[route["label"]]["category"]
    return predicted_category == expected_category(q)


# =============================================================================
# SECTION 4 — THRESHOLD SWEEP ANALYSIS
# =============================================================================
#
# WHY IN PRODUCTION
# -----------------
# A single threshold evaluation tells you how one setting performs. A threshold
# sweep shows you the shape of the precision/recall curve so you can make an
# informed tradeoff decision. This is the standard evaluation loop teams run
# before deploying a classifier-based router:
#
#   Precision of routing = (correct specific-namespace routes) / (all specific-namespace routes)
#     → "When we routed to a specific namespace, how often were we right?"
#
#   Recall of routing = (correct specific-namespace routes) / (total queries)
#     → "What fraction of all queries did we correctly route?"
#
#   Coverage = (queries routed to specific namespace) / (total queries)
#     → "How often did the classifier have enough confidence to route at all?"
#
# The best threshold maximises precision while maintaining acceptable coverage.
# If coverage is too low at your precision target, your classifier needs better
# label descriptions, more training data, or a larger model.
#
# ALTERNATIVES TO A FIXED THRESHOLD
# -----------------------------------
# • Per-class thresholds: tune a separate threshold for each intent class
#     based on that class's historical confusion rate.
#
# • Entropy-based confidence: instead of using the top score, measure the
#     entropy of the full score distribution. Low entropy (one class dominates)
#     → high confidence. High entropy (scores spread evenly) → route to full
#     corpus. More principled than a raw score cutoff.
#     H = -Σ p_i × log(p_i)
#
# • Abstention class: add an explicit "UNCLEAR" or "GENERAL" intent class.
#     The model routes to full corpus when this class wins rather than using
#     a numeric threshold.
#
# • Cascading classifiers: a fast binary classifier first (device vs. SMB),
#     then a fine-grained classifier within the winning branch. Lower total
#     latency than one 4-class model because each model is simpler.

def threshold_analysis(all_results, thresholds=THRESHOLDS):
    rows = []
    for thresh in thresholds:
        specific   = 0
        full       = 0
        correct    = 0
        misrouted  = 0

        for r in all_results:
            q      = r["query"]
            ranked = r["ranked_classes"]
            route  = routing_decision(ranked, threshold=thresh)

            if route["decision"] == "SPECIFIC_NAMESPACE":
                specific += 1
                if routing_correct(route, q):
                    correct += 1
                else:
                    misrouted += 1
            else:
                full += 1
                # Full corpus is never misrouted but doesn't count as specific-correct

        total     = len(all_results)
        coverage  = specific / total
        precision = (correct / specific) if specific > 0 else 0.0
        # Recall = correct specific routes / total queries
        recall    = correct / total

        rows.append({
            "threshold":  thresh,
            "specific":   specific,
            "full":       full,
            "correct":    correct,
            "misrouted":  misrouted,
            "coverage":   round(coverage, 2),
            "precision":  round(precision, 2),
            "recall":     round(recall, 2),
            "f1":         round(
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0, 2
            ),
        })
    return rows


def best_threshold(sweep_rows):
    """
    Best threshold = highest F1 (harmonic mean of precision and recall).
    F1 balances the two: a threshold with perfect precision but 10% recall
    is not useful in practice.
    """
    return max(sweep_rows, key=lambda r: r["f1"])


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def print_classification_results(all_results):
    print("=" * 80)
    print("  CLASSIFICATION RESULTS — All scores per query")
    print("=" * 80)
    print()
    for r in all_results:
        q = r["query"]
        print(f"  {q['id']} [{q['query_type']}] [{q['difficulty']}]")
        print(f"  Query   : {q['text']}")
        print(f"  Expected doc: {q['expected_doc']}  |  Expected category: {expected_category(q)}")
        print()

        rows = [
            [
                rc["label"],
                f"{rc['score']:.4f}",
                "█" * int(rc["score"] * 30),
            ]
            for rc in r["ranked_classes"]
        ]
        print(tabulate(rows, headers=["Intent Class", "Score", ""], tablefmt="rounded_outline"))
        print()


def print_routing_log(all_results, threshold=PRIMARY_THRESHOLD):
    print("=" * 80)
    print(f"  ROUTING DECISION LOG  (threshold = {threshold})")
    print("=" * 80)
    print()

    rows = []
    for r in all_results:
        q     = r["query"]
        route = routing_decision(r["ranked_classes"], threshold=threshold)
        ok    = routing_correct(route, q)

        decision_str = (
            route["namespace"] if route["decision"] == "SPECIFIC_NAMESPACE"
            else "FULL CORPUS"
        )
        rows.append([
            q["id"],
            route["label"][:28],
            f"{route['confidence']:.2f}",
            decision_str,
            "✓ correct" if ok else "✗ misrouted",
        ])

    print(tabulate(
        rows,
        headers=["Query", "Predicted Intent", "Conf", "Routing Decision", "Outcome"],
        tablefmt="rounded_outline",
        colalign=("left", "left", "center", "left", "left"),
    ))
    print()


def print_threshold_analysis(sweep_rows):
    print("=" * 80)
    print("  THRESHOLD SWEEP ANALYSIS")
    print("=" * 80)
    print()

    rows = [
        [
            r["threshold"],
            f"{r['specific']}/{r['specific'] + r['full']}",
            r["misrouted"],
            f"{r['coverage']:.0%}",
            f"{r['precision']:.0%}",
            f"{r['recall']:.0%}",
            f"{r['f1']:.2f}",
        ]
        for r in sweep_rows
    ]
    print(tabulate(
        rows,
        headers=["Threshold", "Specific Routes", "Misrouted", "Coverage", "Precision", "Recall", "F1"],
        tablefmt="rounded_outline",
        colalign=("center", "center", "center", "center", "center", "center", "center"),
    ))
    print()

    best = best_threshold(sweep_rows)
    print(f"  Best threshold by F1: {best['threshold']}")
    print(f"    Coverage  : {best['coverage']:.0%} of queries routed to specific namespace")
    print(f"    Precision : {best['precision']:.0%} of specific routes were correct")
    print(f"    Recall    : {best['recall']:.0%} of all queries correctly routed")
    print(f"    F1        : {best['f1']:.2f}")
    print()
    print("  Interpretation")
    print("  " + "─" * 60)
    print("  • High threshold → conservative routing, fewer misroutes,")
    print("    but more queries fall back to expensive full-corpus search.")
    print("  • Low threshold → aggressive routing, better latency and precision,")
    print("    but misrouted queries silently return irrelevant results.")
    print("  • Ambiguous and context-dependent queries (Q05–Q08) will drive")
    print("    misroutes at lower thresholds — their confidence scores are")
    print("    naturally spread across classes, a signal worth monitoring.")
    print("  • In production, track misroute rate per intent class separately.")
    print("    A billing query going to claims is a different severity than")
    print("    a claims query going to full corpus.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    clf = build_classifier()

    print("=" * 80)
    print("  RUNNING ZERO-SHOT CLASSIFICATION ON ALL QUERIES")
    print("=" * 80)
    print()

    all_results = []

    for q in QUERIES:
        print(f"  Classifying {q['id']}: {q['text'][:70]}...")
        ranked = classify_query(clf, q["text"])
        route  = routing_decision(ranked, threshold=PRIMARY_THRESHOLD)
        ok     = routing_correct(route, q)

        all_results.append({
            "query":         q,
            "ranked_classes": ranked,
            "route":         route,
            "route_correct": ok,
        })

    print()
    print_classification_results(all_results)
    print_routing_log(all_results, threshold=PRIMARY_THRESHOLD)

    sweep = threshold_analysis(all_results, thresholds=THRESHOLDS)
    print_threshold_analysis(sweep)

    # Accuracy summary
    correct_routes = sum(1 for r in all_results if r["route_correct"])
    total          = len(all_results)
    print(f"  Routing accuracy at threshold {PRIMARY_THRESHOLD}: {correct_routes}/{total}")
    print()

    # Persist results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(
            {
                "threshold_used": PRIMARY_THRESHOLD,
                "queries": [
                    {
                        "query_id":       r["query"]["id"],
                        "query_text":     r["query"]["text"],
                        "query_type":     r["query"]["query_type"],
                        "difficulty":     r["query"]["difficulty"],
                        "expected_doc":   r["query"]["expected_doc"],
                        "expected_category": expected_category(r["query"]),
                        "ranked_classes": r["ranked_classes"],
                        "route":          r["route"],
                        "route_correct":  r["route_correct"],
                    }
                    for r in all_results
                ],
                "threshold_sweep": sweep,
                "best_threshold":  best_threshold(sweep),
            },
            f,
            indent=2,
        )
    print(f"  Results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
