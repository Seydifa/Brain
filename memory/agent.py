"""
Memory Agent — the brain's central intelligence.

This is the ONLY component that has a complete world view:
  - Episode diary  (memory/episodes.py)
  - Knowledge store (memory/store.py)
  - Coverage awareness (memory/awareness.py)

Every other agent operates on an ORIENTED CONTEXT produced here.
No other agent ever reads from episodes or the vector store directly.

Public API
----------
classify_and_orient(goal, search_feedback) -> oriented_context dict
update_coverage(goal, current_context)     -> updated oriented_context dict
finalize_episode(episode_id, ...)          -> None (persists completed turn)

oriented_context schema
-----------------------
{
    "turn_type":            str,          # new_topic | follow_up | elaboration | ...
    "relevant_knowledge":   list[dict],   # [{desc, raw, score}, ...]
    "conversation_thread":  list[dict],   # relevant past turns
    "coverage":             str,          # full | partial | none
    "weak_topics":          list[str],    # targeted search hints
    "current_episode_id":   str,
    "parent_episode_id":    str | None,
    "knowledge_confidence": float,
}
"""

import re
import json
from langchain_core.messages import HumanMessage
from core.config import get_llm

from memory.episodes import (
    make_episode_id,
    save_episode,
    get_recent,
    get_by_id,
)
from memory.awareness import assess
from memory.store import store
from core.prompts import MEMORY_CLASSIFY_SYSTEM


_llm = get_llm(temperature=0, role="purpose")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_turn(
    goal: str,
    recent_episodes: list[dict],
) -> tuple[str, str | None]:
    """Classify the turn type and identify a parent episode if relevant."""
    if not recent_episodes:
        return "new_topic", None

    history = "\n".join(
        f"[{ep['id']}] {ep['user_request'][:120]}" for ep in recent_episodes[:3]
    )

    resp = _llm.invoke(
        [
            MEMORY_CLASSIFY_SYSTEM,
            HumanMessage(
                content=(
                    f"Current request: {goal}\n\nRecent episode history:\n{history}"
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


def _build_conversation_thread(turn_type: str, parent_id: str | None) -> list[dict]:
    """
    Build the conversation thread to pass to QA Agent.

    Only returns content for follow-up / elaboration / correction turns
    where continuity matters. New topics get an empty thread.
    """
    if turn_type not in ("follow_up", "elaboration", "clarification", "correction"):
        return []

    if not parent_id:
        return []

    parent = get_by_id(parent_id)
    if not parent:
        return []

    return [
        {
            "role": "previous_turn",
            "request": parent["user_request"],
            "response": parent["chosen_response"] or "",
            "flags": json.loads(parent.get("flags") or "[]"),
        }
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_and_orient(
    goal: str,
    search_feedback: str = "",
    thread_id: str = "",
    direction_result: dict | None = None,
) -> dict:
    """
    Build the oriented_context for the current turn.

    Steps:
    1. If direction_result is provided (from direction_node), use its turn_type
       and parent_id directly — skips LLM classification entirely.
       Otherwise fall back to local LLM classification (backward compat).
    2. Assess knowledge coverage in the vector store
    3. Build conversation thread (only if this is a follow-up type turn)
    4. Register a placeholder episode (so the id exists for downstream reasoning)
    5. Return full oriented_context (includes bridge_sentence for topic changes)
    """
    if direction_result:
        turn_type = direction_result.get("turn_type", "new_topic")
        parent_id = direction_result.get("parent_id")
        bridge_sentence = direction_result.get("bridge_sentence", "")
    else:
        # Fallback: no direction_node upstream (backward compat or direct use)
        recent_episodes = get_recent(n=5, thread_id=thread_id)
        turn_type, parent_id = _classify_turn(goal, recent_episodes)
        bridge_sentence = ""

    coverage_result = assess(goal, search_feedback=search_feedback)
    conversation_thread = _build_conversation_thread(turn_type, parent_id)

    episode_id = make_episode_id(goal)
    save_episode(
        episode_id=episode_id,
        user_request=goal,
        turn_type=turn_type,
        follow_up_of=parent_id,
        flags=["in_progress"],
        thread_id=thread_id,
    )

    return {
        "turn_type": turn_type,
        "relevant_knowledge": coverage_result["chunks"],
        "conversation_thread": conversation_thread,
        "coverage": coverage_result["coverage"],
        "weak_topics": coverage_result["weak_topics"],
        "current_episode_id": episode_id,
        "parent_episode_id": parent_id,
        "knowledge_confidence": coverage_result["best_score"],
        "bridge_sentence": bridge_sentence,
    }


def update_coverage(goal: str, current_context: dict) -> dict:
    """
    After new knowledge has been stored, re-assess coverage without
    re-running the LLM classification (the turn type is already known).

    Returns an updated oriented_context with refreshed knowledge fields.
    """
    coverage_result = assess(goal)
    return {
        **current_context,
        "relevant_knowledge": coverage_result["chunks"],
        "coverage": coverage_result["coverage"],
        "weak_topics": coverage_result["weak_topics"],
        "knowledge_confidence": coverage_result["best_score"],
    }


def finalize_episode(
    episode_id: str,
    goal: str,
    response: str,
    reasoning_trace: list[str],
    turn_type: str,
    follow_up_of: str | None,
    knowledge_conf: float,
    search_was_used: bool = False,
    qa_retried: bool = False,
    thread_id: str = "",
) -> None:
    """
    Update the in-progress episode placeholder with the final response,
    full reasoning trace, and computed flags.
    """
    flags = []
    if turn_type != "new_topic":
        flags.append(turn_type)
    if knowledge_conf < 0.5:
        flags.append("partial_coverage")
    if search_was_used:
        flags.append("search_used")
    if qa_retried:
        flags.append("qa_retried")

    topic_cluster = "_".join(goal.lower().split()[:3]).strip("?.,!")

    save_episode(
        episode_id=episode_id,
        user_request=goal,
        chosen_response=response,
        reasoning_trace=reasoning_trace,
        flags=flags,
        topic_cluster=topic_cluster,
        follow_up_of=follow_up_of,
        turn_type=turn_type,
        knowledge_conf=knowledge_conf,
        thread_id=thread_id,
    )
