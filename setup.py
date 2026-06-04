"""
Confirms the corpus and queries are correctly loaded.
Run: python setup.py
"""

from corpus import CORPUS
from queries import QUERIES
from tabulate import tabulate


def main():
    print("=" * 60)
    print("  RAG Experiments — Project Setup Confirmation")
    print("=" * 60)

    # ── Corpus summary ──────────────────────────────────────────
    print(f"\nCorpus loaded: {len(CORPUS)} documents\n")
    corpus_rows = [(d["id"], d["category"], d["title"][:55]) for d in CORPUS]
    print(tabulate(corpus_rows, headers=["ID", "Category", "Title"], tablefmt="rounded_outline"))

    by_category = {}
    for d in CORPUS:
        by_category.setdefault(d["category"], 0)
        by_category[d["category"]] += 1
    print("\nCategory breakdown:")
    for cat, count in by_category.items():
        print(f"  {cat}: {count} documents")

    # ── Query summary ───────────────────────────────────────────
    print(f"\nQueries ready: {len(QUERIES)}\n")
    query_rows = [
        (q["id"], q["query_type"], q["difficulty"], q["expected_doc"], q["text"][:50] + "…")
        for q in QUERIES
    ]
    print(
        tabulate(
            query_rows,
            headers=["ID", "Type", "Difficulty", "Expected", "Query"],
            tablefmt="rounded_outline",
        )
    )

    type_counts = {}
    for q in QUERIES:
        type_counts.setdefault(q["query_type"], 0)
        type_counts[q["query_type"]] += 1
    print("\nQuery type breakdown:")
    for qtype, count in type_counts.items():
        print(f"  {qtype}: {count} queries")

    print("\n✓ Setup complete. Ready to run experiments.\n")


if __name__ == "__main__":
    main()
