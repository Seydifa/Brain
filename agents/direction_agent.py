"""
Direction Agent — conversational direction intelligence.

Determines the structural direction of each turn using two layers:

Layer 1 — Hard signals (no LLM needed, instant):
  • Empty episode history                             → new_topic
  • Explicit disclaimer regex ("setting X aside", etc.)→ new_topic
  • Cosine similarity < NEW_TOPIC_THRESHOLD           → new_topic

Layer 2 — LLM disambiguation (only when similarity is in the ambiguous range):
  • Cosine similarity >= NEW_TOPIC_THRESHOLD          → LLM decides
    (follow_up / elaboration / clarification / correction / new_topic)

When a topic change is detected and prior episodes exist, generates a
bridge sentence that the QA Agent will incorporate at the start of its answer:

    "We were discussing [prev_topic], but your question now shifts to
     a new topic — starting fresh."

Public API
----------
direction_node(state, config)   LangGraph node. Always runs first.
                                Writes direction_result to state.
classify_direction(goal, thread_id)  Pure function — usable outside the graph.

direction_result schema
-----------------------
{
    "turn_type":       str,          # new_topic | follow_up | elaboration | ...
    "parent_id":       str | None,
    "bridge_sentence": str,          # "" if same topic or no prior episodes
    "semantic_sim":    float,        # cosine(goal_emb, best_recent_emb), 0–1
    "method":          str,          # "empty_history" | "disclaimer" | "semantic" | "llm"
}
"""

from __future__ import annotations

import re
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from core.state import BrainState
from memory.episodes import get_recent
from prompts import DIRECTION_SYSTEM


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Cosine similarity below this → definitely new topic, no LLM needed
NEW_TOPIC_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Module-level singletons (shared with memory/store to stay consistent)
# ---------------------------------------------------------------------------

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ---------------------------------------------------------------------------
# Explicit topic-change disclaimer patterns
# These phrases signal an intentional subject switch regardless of similarity
# ---------------------------------------------------------------------------

_DISCLAIMER_PATTERNS = [
    r"\bsetting .{2,50}? aside\b",
    r"\bswitching (back )?to\b",
    r"\b(completely |totally |entirely )?different (question|topic|subject)\b",
    r"\bchanging (the )?subject\b",
    r"\bforget(ting)? .{2,50}? (for now|for a moment|a moment)\b",
    r"\bleaving .{2,50}? (behind|aside|for now)\b",
    r"\bapart from .{2,60}?,\b",
    r"\binstead[,]? let'?s\b",
    r"\bnow (I want to |let'?s )(talk|ask|discuss|explore)\b",
    r"\bon (a )?different (note|topic|subject)\b",
    r"\bunrelated (question|to)\b",
    r"\bjump(ing)? (to|over to)\b",
    r"\bmov(ing|e) on to\b",
    r"\bshift(ing)? (to|our focus)\b",
    r"\bnew (line of |)?question\b",
]

_DISCLAIMER_RE = re.compile("|".join(_DISCLAIMER_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_disclaimer(goal: str) -> bool:
    """Return True if the goal contains an explicit topic-change signal."""
    return bool(_DISCLAIMER_RE.search(goal))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _embed(text: str) -> list[float]:
    """Embed a single text string."""
    return _embeddings.embed_query(text)


def _topic_label(episode: dict) -> str:
    """Short human-readable topic label for an episode."""
    cluster = (episode.get("topic_cluster") or "").strip()
    if cluster:
        return cluster
    req = episode.get("user_request") or ""
    words = req.split()[:6]
    return " ".join(words) + ("…" if len(req.split()) > 6 else "")


def _build_bridge_sentence(goal: str, prev_episode: dict, method: str) -> str:
    """
    Template-based transition sentence — no extra LLM call.
    Acknowledges what was being discussed and frames the topic change.
    """
    prev_label = _topic_label(prev_episode)
    # First ~7 words of the new goal as a preview
    new_words = goal.split()[:7]
    new_label = " ".join(new_words) + ("…" if len(goal.split()) > 7 else "")

    if method == "disclaimer":
        return (
            f"I see you're setting aside our discussion on **{prev_label}** — "
            f"your new question (*{new_label}*) starts a fresh topic."
        )
    return (
        f"We were discussing **{prev_label}**; "
        f"your question now shifts to a new topic — starting fresh."
    )


def _llm_classify(
    goal: str,
    recent_episodes: list[dict],
    semantic_sim: float,
) -> tuple[str, str | None]:
    """
    Ask the Direction LLM to disambiguate when the semantic similarity is
    in the ambiguous range. Returns (turn_type, parent_id | None).
    """
    history = "\n".join(
        f"[{ep['id']}] {ep['user_request'][:120]}" for ep in recent_episodes[:3]
    )
    resp = _llm.invoke(
        [
            DIRECTION_SYSTEM,
            HumanMessage(
                content=(
                    f"Current request: {goal}\n"
                    f"Semantic similarity to most recent episode: {semantic_sim:.3f}\n\n"
                    f"Recent episode history:\n{history}"
                )
            ),
        ]
    )

    type_m = re.search(r"TYPE:\s*(\w+)", resp.content, re.IGNORECASE)
    parent_m = re.search(r"PARENT:\s*(\S+)", resp.content, re.IGNORECASE)

    turn_type = type_m.group(1).lower() if type_m else "new_topic"
    raw_parent = parent_m.group(1) if parent_m else "null"
    parent_id = None if raw_parent.lower() == "null" else raw_parent
    return turn_type, parent_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_direction(goal: str, thread_id: str = "") -> dict:
    """
    Classify the conversational direction of the current goal.

    Two-layer approach:
      Layer 1: hard signals → new_topic without LLM (fast path)
      Layer 2: LLM with semantic_sim context → any type (slow path)

    Returns a direction_result dict.
    """
    recent_episodes = get_recent(n=5, thread_id=thread_id)

    # ── Layer 1a: empty history ──────────────────────────────────────────────
    if not recent_episodes:
        return {
            "turn_type": "new_topic",
            "parent_id": None,
            "bridge_sentence": "",
            "semantic_sim": 0.0,
            "method": "empty_history",
        }

    # ── Layer 1b: explicit disclaimer regex ─────────────────────────────────
    if _detect_disclaimer(goal):
        bridge = _build_bridge_sentence(goal, recent_episodes[0], method="disclaimer")
        return {
            "turn_type": "new_topic",
            "parent_id": None,
            "bridge_sentence": bridge,
            "semantic_sim": 0.0,
            "method": "disclaimer",
        }

    # ── Layer 1c: semantic distance (embed goal vs most recent episode) ──────
    goal_emb = _embed(goal)
    recent_req = recent_episodes[0].get("user_request", "")
    sim = _cosine_similarity(goal_emb, _embed(recent_req)) if recent_req else 0.0

    # Also check 2nd episode if available for a better parent match
    best_sim = sim
    best_ep = recent_episodes[0]
    if len(recent_episodes) > 1:
        sim2 = _cosine_similarity(
            goal_emb, _embed(recent_episodes[1].get("user_request", ""))
        )
        if sim2 > best_sim:
            best_sim = sim2
            best_ep = recent_episodes[1]

    # Hard cut: too far → new_topic, skip LLM
    if best_sim < NEW_TOPIC_THRESHOLD:
        bridge = _build_bridge_sentence(goal, recent_episodes[0], method="semantic")
        return {
            "turn_type": "new_topic",
            "parent_id": None,
            "bridge_sentence": bridge,
            "semantic_sim": best_sim,
            "method": "semantic",
        }

    # ── Layer 2: LLM disambiguation (ambiguous similarity range) ────────────
    turn_type, parent_id = _llm_classify(goal, recent_episodes, best_sim)
    bridge = (
        _build_bridge_sentence(goal, recent_episodes[0], method="semantic")
        if turn_type == "new_topic"
        else ""
    )
    return {
        "turn_type": turn_type,
        "parent_id": parent_id or (best_ep["id"] if turn_type != "new_topic" else None),
        "bridge_sentence": bridge,
        "semantic_sim": best_sim,
        "method": "llm",
    }


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def direction_node(state: BrainState, config: RunnableConfig) -> dict:
    """
    LangGraph entry node. Runs before memory_classify every turn.
    Writes direction_result to state so classify_node can skip LLM classification.
    """
    goal = state["goal"]
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")

    result = classify_direction(goal, thread_id=thread_id)

    return {
        "direction_result": result,
        "reasoning_trace": [
            f"direction: {result['turn_type']} | "
            f"method={result['method']} | "
            f"sim={result['semantic_sim']:.3f} | "
            f"bridge={'yes' if result['bridge_sentence'] else 'no'}"
        ],
    }
