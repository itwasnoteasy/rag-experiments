# RAG Experiments

A hands-on learning project for understanding **Retrieval-Augmented Generation**, **evaluation**, **guardrails**, and **multi-agent orchestration** through a synthetic telecom/insurance knowledge base. Every concept is designed to be interview-ready for an Agentic AI Architect role.

---

## Project Structure

```
rag-experiments/
├── corpus.py                       # 20 synthetic documents (telecom/insurance KB)
├── queries.py                      # 10 labelled test queries across 5 difficulty types
├── setup.py                        # Confirmation script — prints corpus + query summary
├── requirements.txt                # Python dependencies
│
├── experiment_1_retrieval.py       # Dense vs BM25 vs RRF comparison
├── experiment_2_reranking.py       # Cross-encoder re-ranking of RRF candidates
├── experiment_3_intent.py          # Zero-shot intent classification + routing
├── experiment_4_langfuse.py        # Full 7-span Langfuse observability pipeline
├── experiment_5_comparison.py      # Retrieval method vs synthesis quality (human eval)
├── experiment_6_llm_judge.py       # Automated RAGAS-style scoring via LLM-as-judge
├── experiment_7_hitl.py            # Human-in-the-loop: escalation + write-action diffs
├── experiment_8_nli_guardrail.py   # NLI contradiction detection guardrail
├── experiment_9_multiagent.py      # 3-agent CrewAI crew with revision loop
│
└── results/                        # JSON outputs from each experiment
    ├── experiment_1_results.json
    ├── experiment_2_results.json
    ├── experiment_3_results.json
    ├── experiment_4_latency_report.json
    ├── experiment_5_synthesis_comparison.json
    ├── experiment_6_judge_results.json
    ├── experiment_7_hitl_results.json
    ├── experiment_8_nli_results.json
    └── experiment_9_multiagent_results.json
```

---

## The Knowledge Base

### corpus.py — 20 Synthetic Documents

| Category | Doc IDs | Topics |
|---|---|---|
| `device_insurance` | DI-001 → DI-010 | Claim filing, deductibles, exclusions, device swap, water damage, enrollment, claim limits, third-party repair, appeals, loaner devices |
| `smb_onboarding` | SMB-001 → SMB-010 | Account setup, number porting, plan selection, billing cycles, admin portal, line activation, fraud prevention, onboarding checklist, porting away, international roaming |

Documents are 100–200 words each in realistic support-KB tone. Deliberate vocabulary overlap between categories makes retrieval non-trivial.

### queries.py — 10 Labelled Test Queries

| ID | Type | Difficulty | Expected Doc | Why It's Interesting |
|---|---|---|---|---|
| Q01 | `exact_match` | easy | DI-002 | Contains policy number `INS-POL-2024` — BM25 token hit |
| Q02 | `exact_match` | easy | DI-004 | Contains `$200` non-return fee — exact dollar amount |
| Q03 | `semantic` | hard | DI-005 | "handset took a dip in the toilet" ≠ "liquid immersion" |
| Q04 | `semantic` | hard | SMB-009 | "keep existing contact numbers" ≠ "port your numbers" |
| Q05 | `ambiguous` | medium | DI-006 | Insurance enrollment: consumer or SMB context? |
| Q06 | `ambiguous` | medium | SMB-004 | Plan change mid-month: billing doc or plan-selection doc? |
| Q07 | `context_dep` | hard | DI-002 | "What about the deductible?" — meaningless without prior context |
| Q08 | `context_dep` | hard | SMB-002 | "How long does that take?" — requires prior porting conversation |
| Q09 | `clear_intent` | easy | DI-001 | "File a claim for broken screen" — unambiguous |
| Q10 | `clear_intent` | easy | SMB-005 | "Give employee admin access" — unambiguous SMB |

---

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Confirm setup
python setup.py
```

**Running in Google Colab:**
```python
!git clone https://github.com/itwasnoteasy/rag-experiments.git
%cd rag-experiments
!pip install -r requirements.txt

import os
from google.colab import userdata
os.environ["GEMINI_API_KEY"]    = userdata.get("GEMINI_API_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = userdata.get("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = userdata.get("LANGFUSE_SECRET_KEY")
```

**Rate-limit notes (free-tier Gemini):**
- `GEMINI_DELAY_S=5` (default) — 5 s between direct Gemini calls; ~12 RPM
- `INTER_QUERY_DELAY_S=20` (default in exp 9) — 20 s between CrewAI queries
- Set `GEMINI_DELAY_S=0` on a paid tier

---

## Experiments

### Experiment 1 — Dense vs BM25 vs RRF Retrieval
**File:** `experiment_1_retrieval.py`  
**Depends on:** nothing (builds its own indexes)

Compares three retrieval strategies on all 10 queries:
- **Dense** (all-MiniLM-L6-v2 → ChromaDB cosine similarity)
- **BM25** (rank-bm25 Okapi BM25)
- **RRF** (Reciprocal Rank Fusion combining both, k=60)

**What you'll observe:**
- BM25 wins on Q01/Q02 (exact token matches: policy number, dollar amount)
- Dense wins on Q03/Q04 (semantic paraphrases with zero keyword overlap)
- RRF is never worst — it consistently rescues what either method misses

**Key interview concept:** RRF is immune to score-scale differences between BM25 (unbounded) and cosine similarity [0,1]. Formula: `score = Σ 1/(k + rank)`.

**Output:** Per-query ranked tables + win-count summary → `results/experiment_1_results.json`

---

### Experiment 2 — Cross-Encoder Re-ranking
**File:** `experiment_2_reranking.py`  
**Depends on:** `results/experiment_1_results.json`

Takes the RRF top-10 candidates and re-scores them with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

**Why this exists:** Bi-encoders encode query and document independently. Cross-encoders feed the pair as `[query][SEP][document]` through a single transformer — full self-attention captures fine-grained relevance that bi-encoders miss.

**What you'll observe:**
- Displacement table: how many positions each document moves up or down
- "Rescue" rate: queries where the correct doc was at rank 3+ in RRF but moved to rank 1 after re-ranking
- Some queries show zero displacement — re-ranking confirmed RRF was already correct

**Key interview concept:** Two-stage retrieval (fast approximate → slow accurate) is the production standard. Re-ranking runs on only the top-20–50 candidates, not the full corpus.

**Output:** Per-query displacement tables + summary → `results/experiment_2_results.json`

---

### Experiment 3 — Zero-Shot Intent Classification & Routing
**File:** `experiment_3_intent.py`  
**Depends on:** nothing

Uses `facebook/bart-large-mnli` (NLI-based zero-shot classifier) to classify each query into one of four intents:
- `DEVICE_INSURANCE_CLAIM`, `DEVICE_INSURANCE_POLICY`
- `SMB_ONBOARDING`, `SMB_BILLING`

Then routes to a specific namespace (smaller index) if confidence ≥ threshold, or falls back to full-corpus retrieval.

**What you'll observe:**
- Q09/Q10 (clear intent): high confidence (>0.85), correct namespace
- Q05/Q06 (ambiguous): confidence near 0.5, falls through to full-corpus
- Threshold sweep across [0.50, 0.60, 0.70, 0.80] with F1 to find the optimal value

**Key interview concept:** NLI framing — premise = query, hypothesis = "This text is about {description}". Softmax over entailment/contradiction/neutral across all labels.

**Output:** Per-query classification + threshold sweep table → `results/experiment_3_results.json`

---

### Experiment 4 — Langfuse Observability
**File:** `experiment_4_langfuse.py`  
**Depends on:** Langfuse account + `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `GEMINI_API_KEY`

Instruments the full pipeline with 7 Langfuse spans per query using OpenTelemetry-based tracing:
1. `intent-classification` (embedding span)
2. `dense-retrieval` (retriever span)
3. `bm25-retrieval` (retriever span)
4. `rrf-fusion` (span)
5. `cross-encoder-rerank` (retriever span)
6. `synthesis` (generation span — includes token counts)
7. Root pipeline span wrapping all

**What you'll observe in Langfuse dashboard:**
- Waterfall trace showing latency breakdown per step
- Which step dominates latency (typically synthesis > re-ranking > embedding)
- Token usage per synthesis call

**Key interview concept:** Production RAG without observability is undebuggable. Langfuse traces let you answer "why did this query get a bad answer?" after the fact.

**Output:** Latency report table → `results/experiment_4_latency_report.json`

---

### Experiment 5 — Retrieval Method vs Synthesis Quality (Human Eval)
**File:** `experiment_5_comparison.py`  
**Depends on:** `GEMINI_API_KEY`

Synthesises answers for 3 selected queries (Q03/Q07/Q09) using three retrieval methods as context:
- **Run A:** Dense top-2
- **Run B:** BM25 top-2
- **Run C:** RRF + cross-encoder top-2

Then collects human scores (1–3) on **Accuracy, Completeness, Grounding** for each response, producing a final ranking table.

**What you'll observe:**
- Side-by-side responses truncated to 100 words
- Whether retrieval quality actually changes the synthesised answer
- Interactive scoring prompt — you score each response yourself

**Key interview concept:** "Context poisoning" — wrong or noisy context causes the LLM to hallucinate, hedge excessively, or silently blend irrelevant facts. This experiment makes that cost visible.

**Model note:** Defaults to `gemini-2.0-flash-lite`. Override: `GEMINI_MODEL=gemini-2.5-flash`.

**Output:** Scores + final ranking → `results/experiment_5_synthesis_comparison.json`

---

### Experiment 6 — LLM-as-Judge Automated Evaluation
**File:** `experiment_6_llm_judge.py`  
**Depends on:** `GEMINI_API_KEY`

Implements RAGAS-style automated scoring for all 10 queries using **4 separate Gemini calls** (one per dimension) as an independent judge:

| Dimension | What it measures |
|---|---|
| **Faithfulness** | Does the answer contain only claims supported by the retrieved context? Quote the violating sentence if not. |
| **Answer Relevance** | Does the answer address what was actually asked? |
| **Context Precision** | What fraction of retrieved docs were used to construct the answer? |
| **Context Recall** | Does the retrieved context contain everything needed for a complete answer? |

Each judge call forces JSON output: `{"score": 0.0–1.0, "reasoning": "one sentence"}`.

**What you'll observe:**
- Faithfulness alerts: any query with faithfulness < 0.70 flagged as `NEEDS HUMAN REVIEW`
- Cross-reference table comparing automated scores vs your manual experiment_5 scores
- Disagreement flags where |Δ| > 0.25

**Key interview concept:** At 10,000 queries/day, human review of every response is impossible. LLM-as-judge evaluates at scale but inherits self-preference bias when using the same model family for both generation and judgment.

**Output:** Full scoring table + calibration comparison → `results/experiment_6_judge_results.json`

---

### Experiment 7 — Human-in-the-Loop Patterns
**File:** `experiment_7_hitl.py`  
**Depends on:** `GEMINI_API_KEY`

Two distinct HITL patterns:

**Part A — Confidence-Gated Escalation**
Routes all 10 queries through two quality gates. If either threshold is breached:
- `Faithfulness < 0.70` OR `Context Recall < 0.60` → `{"status": "needs_human_review", "draft_answer": "...", "reason": "..."}`
- Otherwise → `{"status": "auto_approved", "answer": "..."}`

Prints summary: how many of 10 queries auto-approved vs escalated.

**Part B — Propose-Then-Confirm for Write Actions**
Simulates 3 write actions (file claim, upgrade plan, grant admin access):
1. AI generates a structured diff: `{"action": "...", "current_state": null, "proposed_state": {...}, "requires_confirmation": true}`
2. Hardcoded human decisions: W01=CONFIRM, W02=REJECT, W03=CONFIRM
3. Execution only happens on CONFIRM; full audit trail recorded with timestamps

**Key interview concept:** Read-vs-write trust separation — the single most important pattern in agentic AI system design. Read actions auto-execute; write actions require a reviewable diff regardless of confidence.

**Output:** Routing envelopes + audit trail → `results/experiment_7_hitl_results.json`

---

### Experiment 8 — NLI Contradiction Guardrail
**File:** `experiment_8_nli_guardrail.py`  
**Depends on:** `results/experiment_6_judge_results.json` (for synthesised answers; runs adversarial test regardless)

Uses `cross-encoder/nli-deberta-v3-base` (~440 MB, CPU-only, no API cost) to classify each (retrieved context, synthesised response) pair as `entailment / neutral / contradiction`. Blocks if `contradiction ≥ 0.50`.

**Part 1 — Real queries:** NLI guardrail on all 10 query responses from experiment 6.

**Part 2 — Adversarial stress test:** Two hand-crafted contradiction cases:
- **ADV-01:** DI-002 says `$29 deductible` for standard phones; adversarial response says `$99` — pure numeric substitution
- **ADV-02:** SMB-002 says `5–7 business days` standard porting; adversarial response says `1–2 days` + fabricates a "same-day" option

**Expected honest finding:** ADV-01 (pure number swap) may be MISSED because NLI models trained on MNLI see very few numeric-value contradictions. ADV-02 (number swap + fabrication) is more likely CAUGHT. The file documents why and what to do about it (numeric regex extractor as a complementary layer).

**Key interview concept:** NLI is a cheap (<50 ms, CPU, no API) circuit breaker that catches semantic contradictions. Pair it with a rule-based numeric extractor to cover its blind spot.

**Output:** NLI scores + adversarial verdicts → `results/experiment_8_nli_results.json`

---

### Experiment 9 — Multi-Agent RAG with CrewAI
**File:** `experiment_9_multiagent.py`  
**Depends on:** `GEMINI_API_KEY`, `pip install crewai`

Implements a 3-agent sequential crew:

| Agent | Role | Tools |
|---|---|---|
| **Retriever Agent** | Calls hybrid retrieval pipeline, returns formatted context | `hybrid_retrieval` tool |
| **Analyst Agent** | Drafts answer AND explicitly lists knowledge gaps | none |
| **Reviewer Agent** | Runs NLI check, approves or sends back for revision (max 1 loop) | `nli_contradiction_check` tool |

**Colab compatibility:** CrewAI's synchronous `kickoff()` fails inside Jupyter's running event loop. Fixed by running `crew.kickoff()` inside a `ThreadPoolExecutor` worker thread — worker threads have no inherited event loop so the check passes.

**Comparison:** The 3 hardest queries (Q04/Q06/Q08) run through both:
- Single-agent baseline (retrieve + one Gemini call)
- 3-agent crew

Side-by-side answers + latency measurement for each.

**What you'll observe:**
- Multi-agent always adds latency (sequential agent calls + tool invocations)
- Q06 (ambiguous): multi-agent advantage — Analyst's explicit `INFORMATION GAPS:` surfaces ambiguity that single-agent silently resolves one way
- Q08 (context-dep): both pipelines are handicapped by the near-empty query; Reviewer/NLI may catch fabrications
- Honest latency overhead: typically 2–4× single-agent per query

**Key interview concept:** Multi-agent is worth the cost when: response quality is a hard constraint, queries require surfacing knowledge gaps, or you have an async review SLA. Single-agent wins when sub-200 ms is required.

**Output:** Per-query agent outputs + latency comparison → `results/experiment_9_multiagent_results.json`

---

## Concept Quick Reference

| Concept | Where it appears | Interview one-liner |
|---|---|---|
| **BM25** | Exp 1 | Sparse probabilistic retrieval; rewards rare token matches; zero semantic understanding |
| **Dense retrieval** | Exp 1 | Bi-encoder maps text to vector space; semantically similar text clusters even with zero word overlap |
| **RRF** | Exp 1, 2, 5–9 | Rank-based fusion; immune to score-scale differences between BM25 and cosine similarity |
| **Cross-encoder re-ranking** | Exp 2 | Joint (query, doc) attention captures fine-grained relevance; O(n) per query — applied to top-20–50 candidates only |
| **Zero-shot NLI classification** | Exp 3 | NLI entailment framing: premise=query, hypothesis="This text is about {label}"; no labelled data needed |
| **Langfuse tracing** | Exp 4 | OpenTelemetry-based; spans typed as retriever/embedding/generation; `flush()` required before process exit |
| **Context poisoning** | Exp 5 | Wrong context causes LLM to hallucinate, hedge, or blend irrelevant facts — retrieval quality directly affects answer quality |
| **LLM-as-judge** | Exp 6 | Automated evaluation at scale; 4 RAGAS dimensions; self-preference bias when same model judges its own output |
| **RAGAS dimensions** | Exp 6 | Faithfulness, Answer Relevance, Context Precision, Context Recall — each scored 0–1 independently |
| **Confidence-gated escalation** | Exp 7A | Route low-confidence responses to human review queue; preserve draft so reviewer edits rather than rewrites |
| **Propose-then-confirm** | Exp 7B | Write actions always generate a structured diff first; human approves/rejects; audit trail for SOC 2 compliance |
| **Read vs write trust separation** | Exp 7 | Read = auto-execute; Write = always gate, even at 99% confidence, in regulated domains |
| **NLI guardrail** | Exp 8 | CPU-only contradiction detection (<50 ms); semantic contradictions: strong; numeric substitutions: weak — complement with regex |
| **Lost in the middle** | Exp 2, 5 | LLMs preferentially attend to context at positions 1 and n; re-ranking puts the best doc at position 1 |
| **Multi-agent orchestration** | Exp 9 | CrewAI sequential process; agents communicate via task outputs; worth it for quality-gated regulated workflows |
| **Trajectory evaluation** | (planned) | Scoring each agent decision in a multi-step pipeline independently, not just the final answer |

---

## Production Architecture: Defence-in-Depth Stack

Built across experiments 1–9, this is the production-grade RAG quality stack:

```
LAYER 0  Intent classification (exp 3)
         Route to namespace or full-corpus before retrieval
         Cost: ~50ms | Benefit: halves retrieval search space

LAYER 1  Hybrid retrieval: RRF(Dense + BM25) (exp 1)
         Never worse than either method alone
         Cost: ~100ms | Benefit: catches both keyword and semantic matches

LAYER 2  Cross-encoder re-ranking (exp 2)
         Re-score top-20 candidates with full joint attention
         Cost: ~200ms | Benefit: positions best doc at rank 1 (fights "lost in the middle")

LAYER 3  Synthesis (exps 5–9)
         Gemini / any LLM; 2–4 sentences grounded in top-2 docs
         Cost: ~500ms | Benefit: the answer

LAYER 4  NLI guardrail (exp 8)
         Contradiction check, CPU-only, synchronous
         Cost: ~20ms | Benefit: catches semantic contradictions before delivery

LAYER 5  LLM faithfulness judge (exp 6)
         Async, 2–5% sample of production traffic
         Cost: ~500ms | Benefit: catches nuanced violations; feeds quality dashboards

LAYER 6  Confidence-gated escalation (exp 7A)
         Faithfulness < 0.70 OR Recall < 0.60 → human review queue
         Cost: human time | Benefit: regulated-domain compliance

LAYER 7  Propose-then-confirm for writes (exp 7B)
         Structured diff → human approval → execution → audit trail
         Cost: human time + latency | Benefit: SOC 2 / GDPR accountability
```

---

## Interview Trade-offs Cheat Sheet

**BM25 vs Dense vs Hybrid**
- BM25 for exact codes, policy numbers, dollar amounts. Dense for paraphrases. Always use hybrid in production; RRF costs nothing and is never worst.

**Re-ranking cost justification**
- Cross-encoder is O(n) per query at inference time but runs on only ~20 candidates. Selective re-ranking (skip queries with high BM25/dense agreement) saves 30–60% of latency.

**LLM-as-judge vs human eval**
- Human eval: gold standard, ~100 samples/day per annotator. LLM-as-judge: scales to 10,000/day but needs calibration curve against human scores. Pearson r > 0.7 required before trusting at scale.

**RAGAS vs custom judge prompts**
- RAGAS library: drop-in benchmark, couples to OpenAI by default. Custom prompts (what we built): full control, visible mechanics, easier to adapt to domain-specific contradictions.

**Multi-agent vs single-agent**
- Single-agent wins on latency (<200 ms required, structured queries). Multi-agent wins on quality when: response must be verified before delivery, ambiguity must be surfaced explicitly, or an async review SLA exists.

**RAG vs fine-tuning**
- RAG for knowledge that changes frequently (support KB, product catalogue). Fine-tuning for style, format, or domain-specific reasoning that doesn't change. Both for regulated domains where answer grounding must be auditable.
