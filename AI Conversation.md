### Concrete recommendation
Structure the PoC as a sequence of deliberate experiments, not a build-to-completion project. Each experiment answers one question that could come up in the interview.
### Week 1 — the retrieval stack
Get a small document corpus (10–20 insurance-style documents — use synthetic policy docs if needed), embed them, and run the same 5 queries through three retrieval modes back to back: dense only, BM25 sparse only, then RRF hybrid. Look at the actual ranked results. Find one query where dense wins, one where BM25 wins. Now you can say in the interview: "I ran this exact comparison and found that exact-match policy numbers and claim IDs retrieved better with BM25, while semantic paraphrases of coverage questions retrieved better with dense — which is why hybrid with RRF is the right default." That's a 30-second answer that sounds like lived experience because it is.
Then add the cross-encoder re-ranking step. Look at cases where the bi-encoder top result gets displaced by the cross-encoder. Understand why — this is the "lost in the middle" story you've been telling. Seeing it happen on real outputs makes it concrete.
### Week 2 — intent classification and routing
Run BERT intent classification on a small labeled set of queries. Deliberately include ambiguous ones. Find where the confidence score drops below your threshold and observe what happens when you route those to the full LLM path instead. This gives you the threshold calibration story from firsthand experience.
### Week 3 — Langfuse instrumentation
Don't save Langfuse for last. Wire it in during Week 1. By Week 3 you want to be reading trace data and cost attribution naturally. The interview question "how do you know your pipeline is working" should be answerable with a specific anecdote: "I was seeing retrieval latency spikes and Langfuse traces showed it was embedding generation, not Pinecone search — which changed where I focused the optimization."

### What to skip
Don't implement the full voice pipeline (Pipecat, LiveKit, STT, TTS) for this purpose. You've already built that. The interview gaps aren't in the voice layer — they're in the retrieval and evaluation layers. Focused experiments on retrieval, re-ranking, and observability will give you more interview-ready material per hour than completing the full end-to-end stack.
Also skip fine-tuning implementation. Reading about PEFT and LoRA at the conceptual level is sufficient for interview conversations about when to fine-tune versus RAG. Actually implementing a fine-tuning run would take a week and the interview question won't require that depth.

### On the YouTube question
Watch one thing only: Karpathy's "Let's build GPT" if you haven't already. Two hours, directly relevant to the transformer internals questions Richard Chen might ask, and it builds genuine intuition rather than surface familiarity. Everything else in the retrieval and agentic space you'll learn faster by running it than watching someone explain it.

### The minimal stack — all local except Langfuse and Gemini
No Pinecone account needed. No cloud vector DB. Everything runs on your laptop.
```
sentence-transformers  → embeddings + cross-encoder (local)
chromadb               → vector store (local)
rank-bm25              → sparse retrieval (local)
transformers           → BERT intent classifier (local)
langfuse               → observability (free cloud tier)
google-generativeai    → Gemini Flash synthesis (you already have this)
```
Total setup: under 15 minutes. No paid APIs except Gemini which you already use.

### Prompt 0 — Paste this first to set up everything
```
Set up a Python project called "rag-experiments" for running retrieval and 
intent classification experiments. Do the following:

1. Create a virtual environment and install:
   sentence-transformers, chromadb, rank-bm25, transformers, torch,
   langfuse, google-generativeai, pandas, tabulate

2. Create a synthetic corpus of exactly 20 documents in a file called 
   corpus.py. The documents should simulate a telecom/insurance knowledge 
   base covering these topics across roughly equal coverage:
   - Device insurance: claim filing, deductible amounts, coverage exclusions,
     device swap process, water damage policy
   - SMB onboarding: account setup, number porting, plan selection, 
     billing cycles, admin portal access
   Each document should be 100-200 words, realistic in tone, with some 
   overlapping terminology between categories to make retrieval non-trivial.

3. Create a file called queries.py containing exactly 10 test queries.
   Design them deliberately:
   - 2 queries with exact-match terms (policy numbers, specific dollar 
     amounts) that should favor BM25
   - 2 queries that are semantic paraphrases with no keyword overlap 
     that should favor dense retrieval
   - 2 ambiguous queries that could match both categories
   - 2 multi-turn context-dependent queries (e.g. "what about the 
     deductible?" that only makes sense with prior context)
   - 2 clear single-intent queries that BERT should classify with 
     high confidence
   Label each query with its "expected best match" document ID and 
   "intended difficulty" (easy/medium/hard).

4. Create a README.md explaining the project structure and how to 
   run each experiment.

Print a confirmation showing corpus loaded (20 docs) and queries ready (10).
```

### Prompt 1 — Dense vs BM25 vs RRF comparison
```
Using the corpus and queries from the setup, build experiment_1_retrieval.py 
that does the following in sequence:

1. DENSE RETRIEVAL
   - Embed all 20 documents using sentence-transformers 
     (model: all-MiniLM-L6-v2)
   - Store in ChromaDB
   - For each of the 10 queries, retrieve top 5 results with cosine 
     similarity scores

2. BM25 SPARSE RETRIEVAL
   - Index all 20 documents with rank-bm25
   - For each of the 10 queries, retrieve top 5 results with BM25 scores

3. RRF HYBRID FUSION
   - Implement Reciprocal Rank Fusion: score = sum(1 / (k + rank)) 
     where k=60, combining dense rank and BM25 rank for each document
   - For each of the 10 queries, show the fused top 5

4. OUTPUT
   For each query print a comparison table with three columns:
   Dense Top-3 | BM25 Top-3 | RRF Top-3
   
   Then print a summary answering:
   - Which queries did dense outperform BM25? (by checking against 
     expected_best_match from queries.py)
   - Which queries did BM25 outperform dense?
   - How often did RRF match or beat the better of the two individual 
     methods?
   
   Save the full results to results/experiment_1_results.json

This is a comparison experiment. Do not skip the summary analysis — 
that is the most important output.
```

### Prompt 2 — Cross-encoder re-ranking
```
Build experiment_2_reranking.py that extends the RRF results from 
experiment 1.

1. Load the RRF top-10 results for each query from 
   results/experiment_1_results.json

2. Re-rank each query's top-10 using a cross-encoder:
   model: cross-encoder/ms-marco-MiniLM-L-6-v2
   The cross-encoder scores query-document pairs jointly.

3. OUTPUT
   For each query, print a "displacement table":
   
   Doc ID | RRF Rank | Cross-encoder Rank | Score | Moved Up/Down/Same
   
   Then print a summary:
   - How many documents moved more than 2 positions?
   - Which query showed the most re-ranking displacement?
   - Find one concrete example where the cross-encoder promoted a 
     document that was ranked 4th or lower by RRF to rank 1 or 2.
     Print that query and explain in one sentence why the cross-encoder 
     likely preferred it.
   
   Save results to results/experiment_2_results.json

   Add a comment in the code at the point where cross-encoder scoring 
   happens explaining in plain English what computation is happening 
   that makes it more accurate but slower than the bi-encoder.
   ```

### Prompt 3 — BERT intent classification and routing
```
Build experiment_3_intent.py that implements intent classification 
and routing decisions.

1. INTENT CLASSIFIER SETUP
   Define 4 intent classes:
   - DEVICE_INSURANCE_CLAIM (filing, status, process)
   - DEVICE_INSURANCE_POLICY (coverage, exclusions, deductibles)
   - SMB_ONBOARDING (setup, porting, plans)
   - SMB_BILLING (billing, payments, invoices)
   
   Use zero-shot classification with:
   model: facebook/bart-large-mnli
   This avoids needing labeled training data.

2. CLASSIFICATION RUN
   Run all 10 queries through the classifier.
   For each query output:
   - Predicted intent
   - Confidence score (0-1)
   - All four class scores ranked

3. ROUTING DECISION SIMULATION
   Apply a confidence threshold of 0.70:
   - Above threshold: route to the specific product namespace in 
     ChromaDB (simulated — just print which namespace would be queried)
   - Below threshold: route to full corpus search (all namespaces)
   
   Print a routing decision log showing for each query:
   Query | Predicted Intent | Confidence | Routing Decision | 
   Would have been correct? (compare to expected category in queries.py)

4. THRESHOLD ANALYSIS
   Run the same classification at thresholds 0.50, 0.60, 0.70, 0.80.
   For each threshold print:
   - How many queries routed to specific namespace vs full corpus
   - How many routing decisions were correct
   
   Print the table and identify the threshold with best precision/recall 
   trade-off for this query set.
   
   Save to results/experiment_3_results.json
```

### Prompt 4 — Langfuse instrumentation
```
Build experiment_4_langfuse.py that wires Langfuse tracing into the 
full pipeline and runs an end-to-end test.

1. SETUP
   Initialize Langfuse client. Use environment variables for keys:
   LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
   Print setup confirmation.

2. INSTRUMENTED PIPELINE
   Build a function run_instrumented_query(query) that:
   a. Creates a Langfuse trace with the query as name
   b. Span 1 "intent-classification": run BERT classifier, log 
      predicted intent and confidence score as output
   c. Span 2 "embedding": embed the query, log latency
   d. Span 3 "retrieval-dense": ChromaDB query, log top-3 doc IDs 
      and scores
   e. Span 4 "retrieval-bm25": BM25 query, log top-3 doc IDs and scores
   f. Span 5 "rrf-fusion": compute RRF scores, log top-5 result
   g. Span 6 "cross-encoder-rerank": re-rank top-5, log final ranking
   h. Span 7 "synthesis": call Gemini Flash with the top-2 documents 
      as context and the query. Log input token count, output token 
      count, and the response.
   
   Each span should record start time, end time, and latency in ms.

3. RUN
   Execute run_instrumented_query() for all 10 queries sequentially.
   
4. LOCAL LATENCY REPORT
   After all runs, print a latency breakdown table:
   
   Span Name | Min ms | Max ms | Avg ms | % of total pipeline time
   
   Then answer in print statements:
   - Which span consumed the most total time across all 10 queries?
   - Which span had the highest variance (max-min)?
   - If you had to optimize one span to hit a 1.5s total pipeline 
     target, which would you target and why?
   
   Save the report to results/experiment_4_latency_report.json
   
   Note: also check the Langfuse dashboard UI at langfuse.com — the 
   traces should appear there within seconds of running.
```

### Prompt 5 — The connecting experiment
```
Build experiment_5_comparison.py that answers the one question that 
ties all four experiments together:

"Does retrieval method choice affect synthesis quality?"

1. For 3 of the 10 queries (pick the 2 hard ones and 1 easy one), 
   run synthesis with Gemini Flash THREE times each using different 
   context:
   
   Run A: Top-2 documents from DENSE ONLY retrieval
   Run B: Top-2 documents from BM25 ONLY retrieval  
   Run C: Top-2 documents from RRF + cross-encoder re-ranking
   
2. For each of the 3 queries, print:
   - The query
   - The 3 synthesized responses side by side (truncated to 100 words)
   - Which retrieved documents were used as context for each run
   
3. Manually evaluate (you do this part — the code just structures it):
   The code should print a simple scoring prompt asking you to rate 
   each response 1-3 on:
   - Accuracy (does it answer correctly?)
   - Completeness (does it cover the full answer?)
   - Grounding (does it stay within the provided context?)
   
   Accept your scores as input (1-3 for each dimension, each run).
   Print a final summary table of your scores.
   
   Save everything to results/experiment_5_synthesis_comparison.json

This experiment is the payoff for all previous work. The output gives 
you a concrete, personally observed data point about why retrieval 
quality affects synthesis quality — which is the core argument behind 
why cross-encoder re-ranking is worth the latency cost.
```

Use Colab if your laptop is underpowered — the free GPU tier handles all five experiments comfortably with that model swap.
### Practical setup — one time, five minutes
```
# Cell 1 in your notebook — run once
!pip install sentence-transformers chromadb rank-bm25 \
             transformers torch langfuse \
             google-generativeai pandas tabulate

# Cell 2 — set your keys once
import os
os.environ["GEMINI_API_KEY"] = "your-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = "your-key"
os.environ["LANGFUSE_SECRET_KEY"] = "your-key"
```

### The two queries to pay close attention to in every experiment
**Q07 and Q08 are your most valuable test cases** — the context-dependent ones. "What about the deductible?" and "And how long does that usually take?" contain almost no retrievable signal on their own. Every retrieval method will struggle with them. That struggle is intentional and directly maps to a real interview talking point: why conversational AI systems need query rewriting before retrieval, which is exactly what your Verizon pipeline did.
When Prompt 1 results come back, Q07 and Q08 results will likely be wrong or random across all three methods. That's not a failure — that's the experiment working correctly. The correct answer when asked about this in an interview is: "These queries are why we added a query rewriting step before retrieval — to resolve coreference and expand context before the embedding is generated."

### write down your prediction for just these three:
```
# Run this cell — just prints the queries to look at
from queries import QUERIES
for q in QUERIES:
    if q['id'] in ['Q01', 'Q03', 'Q07']:
        print(f"{q['id']} ({q['type']}, expected: {q['expected_best_match']})")
        print(f"  → {q['query']}")
        print(f"  → I predict: BM25 wins / Dense wins / Both fail")
        print()
```

## What to do next — the learning loop 
### After Prompt0
The real learning happens by running retrievers against the queries and observing where they succeed and fail. Here's the progression:
#### Step 1 — BM25 (sparse retrieval)
Add a new cell and run this:
```
from rank_bm25 import BM25Okapi
from corpus import CORPUS
from queries import QUERIES

# Index the corpus
tokenized_corpus = [doc["content"].lower().split() for doc in CORPUS]
bm25 = BM25Okapi(tokenized_corpus)

# Run all 10 queries
for q in QUERIES:
    tokens = q["text"].lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = scores.argsort()[::-1][:3]
    top_docs = [CORPUS[i]["id"] for i in top_idx]
    hit = "✓" if q["expected_doc"] in top_docs else "✗"
    print(f"{hit} [{q['query_type']:12}] {q['id']} | Expected: {q['expected_doc']} | Got: {top_docs}")
```
What to observe:
- Q01, Q02 (exact_match) should show ✓ — BM25 finds exact tokens like INS-POL-2024 and $200
- Q03, Q04 (semantic) should show ✗ — "handset" and "toilet" don't exist in the corpus, BM25 is blind to meaning
- Q07, Q08 (context_dep) will almost certainly fail — 4-word queries have no signal

#### Step 2 — Dense retrieval (semantic embeddings)
```
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # fast, good quality

# Embed all documents
doc_texts = [doc["content"] for doc in CORPUS]
doc_embeddings = model.encode(doc_texts, show_progress_bar=True)

# Run all 10 queries
for q in QUERIES:
    q_emb = model.encode(q["text"])
    sims = np.dot(doc_embeddings, q_emb) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    top_idx = sims.argsort()[::-1][:3]
    top_docs = [CORPUS[i]["id"] for i in top_idx]
    hit = "✓" if q["expected_doc"] in top_docs else "✗"
    print(f"{hit} [{q['query_type']:12}] {q['id']} | Expected: {q['expected_doc']} | Got: {top_docs}")
```
What to observe:
- Q03, Q04 (semantic) should now flip to ✓ — embeddings understand "handset" ≈ "device", "keep numbers" ≈ "porting"
- Q01, Q02 (exact_match) may or may not hold — interesting to compare with BM25
- Q07, Q08 (context_dep) still likely fail — short decontextualized queries are hard even for embeddings

#### Step 3 — Side-by-side comparison (the real insight)
```
# Run both and compare
results = []
for q in QUERIES:
    # BM25
    tokens = q["text"].lower().split()
    bm25_top = [CORPUS[i]["id"] for i in bm25.get_scores(tokens).argsort()[::-1][:1]]
    
    # Dense
    q_emb = model.encode(q["text"])
    sims = np.dot(doc_embeddings, q_emb) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(q_emb))
    dense_top = [CORPUS[i]["id"] for i in sims.argsort()[::-1][:1]]
    
    bm25_hit = "✓" if q["expected_doc"] == bm25_top[0] else "✗"
    dense_hit = "✓" if q["expected_doc"] == dense_top[0] else "✗"
    results.append([q["id"], q["query_type"], q["difficulty"], q["expected_doc"], 
                    f"{bm25_hit} {bm25_top[0]}", f"{dense_hit} {dense_top[0]}"])

from tabulate import tabulate
print(tabulate(results, headers=["ID","Type","Diff","Expected","BM25 Top1","Dense Top1"], tablefmt="rounded_outline"))
```
This gives you a head-to-head table — the clearest way to internalize when each approach wins and why.

#### The nuances to look for
| Observation | What it teaches |
| ----------  | ------ |
| BM25 nails Q01/Q02 Dense also does | Exact terms dominate the embedding space too — not always a pure tradeoff |
| Dense fails Q07/Q08 ("What about the deductible?") | Short context-free queries lack enough signal even for embeddings — motivates query rewriting |
| Both struggle on Q05/Q06 (ambiguous)	| When both categories are plausible retrieval alone can't resolve intent — motivates classification before retrieval |
| Dense top-3 includes the right doc but not at rank 1 | Recall vs precision tradeoff — in RAG you typical |

Run Step 1 first, write down your predictions for Dense before running Step 2, then check. The surprises are where the learning sticks.
