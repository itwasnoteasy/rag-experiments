# RAG Experiments

A hands-on learning project for understanding **Retrieval-Augmented Generation** and **intent classification** through a synthetic telecom/insurance knowledge base.

---

## Learning Goals

| Concept | What You'll Observe |
|---|---|
| BM25 vs Dense Retrieval | When sparse keyword matching beats embeddings and vice versa |
| Query difficulty | How exact-match, semantic, and ambiguous queries stress-test retrievers |
| Intent classification | BERT-style classifiers on clear vs. ambiguous queries |
| Multi-turn context | Why standalone follow-up queries fail without conversation history |
| Observability | Tracing retrieval pipelines with Langfuse |

---

## Project Structure

```
rag-experiments/
├── corpus.py          # 20 synthetic documents (telecom/insurance KB)
├── queries.py         # 10 labelled test queries across 5 difficulty types
├── setup.py           # Confirmation script — prints corpus + query summary
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

### corpus.py — The Knowledge Base

20 documents split evenly across two categories:

| Category | Doc IDs | Topics |
|---|---|---|
| `device_insurance` | DI-001 → DI-010 | Claim filing, deductibles, exclusions, device swap, water damage, enrollment, loaner devices, appeals |
| `smb_onboarding` | SMB-001 → SMB-010 | Account setup, number porting, plan selection, billing cycles, admin portal, security, international roaming |

Documents are 100–200 words each, written in realistic support-KB tone. **Deliberate overlap**: terms like "account", "plan", "device", and "insurance" appear in both categories to make retrieval non-trivial.

### queries.py — The Test Queries

10 labelled queries across 5 types, each with `expected_doc`, `difficulty`, and `query_type`:

| ID | Type | Difficulty | Expected Doc | Why It's Interesting |
|---|---|---|---|---|
| Q01 | `exact_match` | easy | DI-002 | Contains policy number `INS-POL-2024` — BM25 token hit |
| Q02 | `exact_match` | easy | DI-004 | Contains `$200` non-return fee — exact dollar amount |
| Q03 | `semantic` | hard | DI-005 | "handset took a dip in the toilet" ≠ "liquid immersion" |
| Q04 | `semantic` | hard | SMB-009 | "keep existing contact numbers" ≠ "port your numbers" |
| Q05 | `ambiguous` | medium | DI-006 | Insurance enrollment: consumer or SMB? |
| Q06 | `ambiguous` | medium | SMB-004 | Plan change mid-month: billing or plan-selection doc? |
| Q07 | `context_dep` | hard | DI-002 | "What about the deductible?" — meaningless without history |
| Q08 | `context_dep` | hard | SMB-002 | "How long does that take?" — requires prior porting context |
| Q09 | `clear_intent` | easy | DI-001 | "File a claim for broken screen" — unambiguous |
| Q10 | `clear_intent` | easy | SMB-005 | "Give employee access to manage account" — unambiguous SMB |

---

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Confirm setup
python setup.py
```

Expected output:
```
============================================================
  RAG Experiments — Project Setup Confirmation
============================================================

Corpus loaded: 20 documents
...
Queries ready: 10
...
✓ Setup complete. Ready to run experiments.
```

---

## Planned Experiments

### Experiment 1 — BM25 Retrieval
**File:** `experiments/bm25_retrieval.py` *(coming soon)*

Uses `rank-bm25` to index the corpus and retrieve top-k documents for each query.
- Expected winners: Q01, Q02 (exact-match queries)
- Expected losers: Q03, Q04 (semantic queries with no keyword overlap)

**Key concept:** BM25 is a bag-of-words model. It rewards term frequency and penalises document length. It has zero understanding of meaning.

### Experiment 2 — Dense Retrieval with ChromaDB
**File:** `experiments/dense_retrieval.py` *(coming soon)*

Uses `sentence-transformers` to embed documents and queries, stores vectors in ChromaDB, and retrieves by cosine similarity.
- Expected winners: Q03, Q04 (semantic paraphrases)
- Expected observation: may struggle on Q01/Q02 if exact terms don't dominate the embedding space

**Key concept:** Dense retrieval maps text into a continuous vector space. Semantically similar sentences cluster together even with zero word overlap.

### Experiment 3 — Hybrid Retrieval (BM25 + Dense)
**File:** `experiments/hybrid_retrieval.py` *(coming soon)*

Combines BM25 and dense scores via reciprocal rank fusion (RRF) or weighted sum.
**Key trade-off:** How to weight sparse vs. dense signals. Hybrid typically outperforms either alone across query types.

### Experiment 4 — Intent Classification
**File:** `experiments/intent_classifier.py` *(coming soon)*

Fine-tunes or zero-shot prompts a BERT-style model to classify queries into `device_insurance` vs. `smb_onboarding`.
- Q09, Q10 should yield high-confidence scores (> 0.9)
- Q05, Q06 (ambiguous) should score near 0.5 — revealing classifier uncertainty

### Experiment 5 — Multi-turn Query Rewriting
**File:** `experiments/query_rewriting.py` *(coming soon)*

Demonstrates how context-dependent queries (Q07, Q08) fail retrieval standalone, and how an LLM can rewrite them into self-contained queries using conversation history.

### Experiment 6 — Observability with Langfuse
**File:** `experiments/traced_pipeline.py` *(coming soon)*

Instruments the retrieval pipeline with Langfuse traces to observe latency, retrieved doc IDs, and relevance scores per query. Essential for production RAG debugging.

---

## Key Concepts Quick Reference

| Term | Definition |
|---|---|
| **BM25** | Probabilistic sparse retrieval based on term frequency; no semantic understanding |
| **Dense Retrieval** | Embedding-based retrieval; captures meaning but slower and requires a vector store |
| **RAG** | Retrieval-Augmented Generation — retrieve relevant docs, then generate an answer |
| **ChromaDB** | Lightweight local vector database; good for experiments, not production scale |
| **Reciprocal Rank Fusion** | Score fusion technique for combining rankings from multiple retrievers |
| **Intent Classification** | Predicting the category/intent of a query before retrieval for routing |
| **Langfuse** | Open-source LLM observability platform for tracing, evals, and dashboards |

---

## Interview-Ready Trade-offs

**BM25 vs Dense:**
- BM25 is fast, interpretable, and great for exact matches. Use it for keyword-heavy domains (legal, medical codes).
- Dense is better for natural language queries. Requires more compute and a vector store.
- In production: **always start with hybrid**.

**ChromaDB vs Production Vector Stores:**
- ChromaDB is easy to run locally; data lives on disk.
- For production use Pinecone, Weaviate, or pgvector. Consider throughput, filtering, and replication needs.

**Intent Classification before Retrieval:**
- Classifying intent first lets you route queries to specialized indexes (e.g., a separate index per category).
- Trade-off: adds latency; misclassification sends the query to the wrong index entirely.
