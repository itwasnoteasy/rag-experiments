"""
10 test queries for RAG retrieval experiments.

Query types:
  exact_match   — contain policy numbers / dollar amounts; BM25 should win
  semantic      — paraphrases with zero keyword overlap; dense retrieval should win
  ambiguous     — could match device_insurance OR smb_onboarding
  context_dep   — multi-turn; only meaningful with prior context
  clear_intent  — single, unambiguous intent; BERT should classify with high confidence

Each query carries:
  expected_doc   : best-match document ID from corpus.py
  difficulty     : easy | medium | hard
  query_type     : one of the five types above
  notes          : brief rationale
"""

QUERIES = [
    # ── Exact-match (BM25-favored) ────────────────────────────────────────────
    {
        "id": "Q01",
        "text": "What is the deductible for a premium smartphone under policy INS-POL-2024?",
        "query_type": "exact_match",
        "expected_doc": "DI-002",
        "difficulty": "easy",
        "notes": (
            "Contains exact policy document number 'INS-POL-2024' and the term 'deductible'. "
            "BM25 should score DI-002 highly on both token matches."
        ),
    },
    {
        "id": "Q02",
        "text": "How much is the non-return fee if I don't send back my swapped device — is it $200?",
        "query_type": "exact_match",
        "expected_doc": "DI-004",
        "difficulty": "easy",
        "notes": (
            "Dollar amount '$200' and phrase 'non-return fee' appear verbatim in DI-004. "
            "BM25 should latch on to exact token overlap."
        ),
    },

    # ── Semantic / paraphrase (dense-retrieval-favored) ───────────────────────
    {
        "id": "Q03",
        "text": "My handset took a dip in the toilet — will my protection plan cover the repair?",
        "query_type": "semantic",
        "expected_doc": "DI-005",
        "difficulty": "hard",
        "notes": (
            "'Handset', 'took a dip', 'toilet' share no tokens with DI-005 which uses "
            "'device', 'liquid immersion', 'sink'. Dense embeddings should bridge the gap."
        ),
    },
    {
        "id": "Q04",
        "text": "We're moving our company to a new telecom provider and need to keep our existing contact numbers.",
        "query_type": "semantic",
        "expected_doc": "SMB-009",
        "difficulty": "hard",
        "notes": (
            "Semantically describes outbound number porting without using 'port', 'PAC', "
            "or 'transfer PIN'. Dense retrieval should match SMB-009."
        ),
    },

    # ── Ambiguous (matches both categories) ───────────────────────────────────
    {
        "id": "Q05",
        "text": "How do I add insurance to devices on my account?",
        "query_type": "ambiguous",
        "expected_doc": "DI-006",
        "difficulty": "medium",
        "notes": (
            "Could match DI-006 (enrolling in device insurance) or SMB-008 (onboarding checklist "
            "step 6 mentions insurance enrollment). Context of 'my account' is ambiguous between "
            "consumer and SMB."
        ),
    },
    {
        "id": "Q06",
        "text": "What happens when I switch to a different plan mid-month?",
        "query_type": "ambiguous",
        "expected_doc": "SMB-004",
        "difficulty": "medium",
        "notes": (
            "Overlaps with both SMB-003 (plan changes take effect next cycle) and SMB-004 "
            "(mid-cycle proration). Retrieval system must disambiguate billing vs. plan-selection context."
        ),
    },

    # ── Context-dependent / multi-turn ────────────────────────────────────────
    {
        "id": "Q07",
        "text": "What about the deductible?",
        "query_type": "context_dep",
        "expected_doc": "DI-002",
        "difficulty": "hard",
        "notes": (
            "Standalone this is nearly meaningless. Only resolves to DI-002 if prior conversation "
            "turn established the topic of device insurance claims. Tests whether the retrieval "
            "system can use conversation history for query rewriting."
        ),
    },
    {
        "id": "Q08",
        "text": "And how long does that usually take?",
        "query_type": "context_dep",
        "expected_doc": "SMB-002",
        "difficulty": "hard",
        "notes": (
            "Completely context-dependent. Assumes prior turn asked about number porting, "
            "making the answer SMB-002's '5–7 business days'. Tests temporal/procedural "
            "follow-up resolution."
        ),
    },

    # ── Clear single-intent (BERT high-confidence) ────────────────────────────
    {
        "id": "Q09",
        "text": "I want to file a claim for my broken screen.",
        "query_type": "clear_intent",
        "expected_doc": "DI-001",
        "difficulty": "easy",
        "notes": (
            "Unambiguously device insurance claim filing. BERT intent classifier should assign "
            "high-confidence 'device_insurance' label. Maps cleanly to DI-001."
        ),
    },
    {
        "id": "Q10",
        "text": "How do I give one of my employees access to manage the company's phone account?",
        "query_type": "clear_intent",
        "expected_doc": "SMB-005",
        "difficulty": "easy",
        "notes": (
            "Unambiguously SMB admin portal / RBAC. BERT should classify as 'smb_onboarding' "
            "with high confidence. Maps directly to SMB-005's sub-administrator section."
        ),
    },
]

if __name__ == "__main__":
    from tabulate import tabulate

    rows = [
        (q["id"], q["query_type"], q["difficulty"], q["expected_doc"], q["text"][:55] + "…")
        for q in QUERIES
    ]
    print(
        tabulate(
            rows,
            headers=["ID", "Type", "Difficulty", "Expected Doc", "Query"],
            tablefmt="rounded_outline",
        )
    )
    print(f"\nQueries ready: {len(QUERIES)}")
