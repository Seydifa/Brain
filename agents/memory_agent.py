"""
Memory Agent nodes — thin graph wrappers around memory/agent.py.

Four nodes exposed to the LangGraph graph:

  classify_node          Start of every turn. Calls classify_and_orient(),
                         which: loads recent episodes, LLM-classifies the turn,
                         assesses knowledge coverage, builds conversation thread,
                         registers placeholder episode. Sets oriented_context.

  store_knowledge_node   After a validated search result. Stores the raw
                         search content into the ChromaDB vector store.

  update_coverage_node   After storing new knowledge. Re-runs awareness
                         assessment (no LLM call) and refreshes the relevant
                         knowledge / coverage fields in oriented_context.

  store_episode_node     Final node of every turn (after qa_final).
                         Reads the full reasoning_trace and finalizes the
                         episode in the SQLite diary.
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from core.state import BrainState
from memory.agent import classify_and_orient, update_coverage, finalize_episode
from memory.store import store


# ---------------------------------------------------------------------------
# Node: classify_node
# ---------------------------------------------------------------------------


def classify_node(state: BrainState, config: RunnableConfig) -> dict:
    """
    Entry node for every turn.
    Produces the oriented_context that all downstream agents will use.
    """
    goal = state["goal"]
    feedback = state.get("search_feedback", "")
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")

    oriented = classify_and_orient(goal, search_feedback=feedback, thread_id=thread_id)

    return {
        "oriented_context": oriented,
        "messages": [HumanMessage(content=goal)],
        "reasoning_trace": [
            f"classified as {oriented['turn_type']} | "
            f"coverage={oriented['coverage']} | "
            f"parent={oriented.get('parent_episode_id', 'none')}"
        ],
    }


# ---------------------------------------------------------------------------
# Node: store_knowledge_node
# ---------------------------------------------------------------------------


def store_knowledge_node(state: BrainState) -> dict:
    """
    Store the validated search result in the ChromaDB knowledge store.
    Reads the last message (search output) from state.
    """
    last_msg = next(
        (m for m in reversed(state.get("messages", [])) if hasattr(m, "content")),
        None,
    )
    raw = last_msg.content if last_msg else ""
    # Gemini can return a list of content blocks instead of a plain string
    if isinstance(raw, list):
        content = " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in raw
        )
    else:
        content = raw

    if content:
        store(content, source="web_search")

    return {
        "status": "partial",
        "reasoning_trace": [f"knowledge stored ({len(content)} chars)"],
    }


# ---------------------------------------------------------------------------
# Node: update_coverage_node
# ---------------------------------------------------------------------------


def update_coverage_node(state: BrainState) -> dict:
    """
    Re-assess knowledge coverage after new material was stored.
    Updates oriented_context in-place (no LLM re-classification).
    """
    goal = state["goal"]
    current_ctx = state.get("oriented_context", {})

    updated_ctx = update_coverage(goal, current_ctx)

    return {
        "oriented_context": updated_ctx,
        "reasoning_trace": [
            f"coverage re-assessed: {updated_ctx.get('coverage')} | "
            f"confidence={updated_ctx.get('knowledge_confidence', 0):.2f}"
        ],
    }


# ---------------------------------------------------------------------------
# Node: store_episode_node
# ---------------------------------------------------------------------------


def store_episode_node(state: BrainState, config: RunnableConfig) -> dict:
    """
    Finalize and persist the current turn as a completed episode.
    Called after qa_final, reads the full reasoning_trace from state.
    """
    ctx = state.get("oriented_context", {})
    episode_id = ctx.get("current_episode_id", "")
    parent_id = ctx.get("parent_episode_id")
    turn_type = ctx.get("turn_type", "new_topic")
    knowledge_conf = ctx.get("knowledge_confidence", 0.0)
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")

    reasoning_trace = state.get("reasoning_trace", [])
    search_was_used = any("search" in s for s in reasoning_trace)
    qa_retried = any("qa scored" in s and "retry" in s for s in reasoning_trace)

    if episode_id:
        finalize_episode(
            episode_id=episode_id,
            goal=state["goal"],
            response=state.get("response", ""),
            reasoning_trace=reasoning_trace,
            turn_type=turn_type,
            follow_up_of=parent_id,
            knowledge_conf=knowledge_conf,
            search_was_used=search_was_used,
            qa_retried=qa_retried,
            thread_id=thread_id,
        )

    return {
        "status": "done",
        "reasoning_trace": ["episode finalized and stored"],
    }
