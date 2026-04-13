"""
Knowledge Awareness — the brain's self-knowledge layer.

Answers: "How well do we already know the answer to this query?"

For a given goal, it:
  1. Queries the vector store for relevant knowledge chunks
  2. Classifies coverage as "full", "partial", or "none"
  3. Returns weak_topics for targeted follow-up searches

Coverage levels
---------------
  full    — 2+ high-confidence chunks (score >= MEMORY_SCORE_THRESHOLD)
            -> skip search, go straight to QA
  partial — some chunks found but confidence is low
            -> targeted search on weak_topics only
  none    — store is empty OR no relevant chunks found
            -> full search on the goal
"""

from memory.store import recall


def assess(query: str, search_feedback: str = "") -> dict:
    # Imported inside the function so the live threshold is always used,
    # even in long-running notebook sessions where modules may be reloaded.
    from core.state import MEMORY_SCORE_THRESHOLD

    """
    Assess how well long-term memory covers a query.

    Args:
        query:           The user's current goal.
        search_feedback: Optional hint from previous search validation —
                         used to sharpen weak_topics on partial coverage.

    Returns:
        {
            "coverage":     "full" | "partial" | "none",
            "best_score":   float,
            "chunks":       list[{desc, raw, score}],
            "weak_topics":  list[str],
        }
    """
    result = recall(query, n=6)

    if result["status"] == "empty":
        return {
            "coverage": "none",
            "best_score": 0.0,
            "chunks": [],
            "weak_topics": [query],
        }

    chunks = result["context"]
    best_score = max(c["score"] for c in chunks)
    high_conf = [c for c in chunks if c["score"] >= MEMORY_SCORE_THRESHOLD]

    # Full coverage: >= 2 chunks are high-confidence
    if len(high_conf) >= 2:
        return {
            "coverage": "full",
            "best_score": best_score,
            "chunks": high_conf,
            "weak_topics": [],
        }

    # Partial coverage: some chunks exist but confidence is low
    # Use search_feedback as a topic hint if available, else use the query itself
    weak_topics = [search_feedback] if search_feedback else [query]
    return {
        "coverage": "partial" if chunks else "none",
        "best_score": best_score,
        "chunks": high_conf if high_conf else chunks[:2],
        "weak_topics": weak_topics,
    }
