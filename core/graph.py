"""
Brain graph — the LangGraph wiring of all agent nodes.

Flow
----
Every turn starts at memory_classify which builds the oriented_context.
Coverage drives the first routing decision:

  full    -> qa_draft  (we already know enough)
  partial -> search    (fill the gaps in weak_topics)
  none    -> search    (brand-new topic, search everything)

After search:

  search_validator
    valid    -> memory_store_knowledge -> memory_update_coverage -> qa_draft
    retry    -> search (up to MAX_SEARCH_RETRIES)
    clarify  -> goal_evaluator -> END

QA loop:

  orchestrator_validate_qa
    approved -> qa_final -> memory_store_episode -> END
    retry    -> qa_draft (up to MAX_QA_ATTEMPTS)
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from core.state import BrainState, MAX_SEARCH_RETRIES, MAX_QA_ATTEMPTS

from agents.memory_agent import (
    classify_node,
    store_knowledge_node,
    update_coverage_node,
    store_episode_node,
)
from agents.direction_agent import direction_node
from agents.search_agent import search_node
from agents.search_validator import validate_search_node
from agents.qa_agent import qa_draft_node, qa_final_node
from agents.orchestrator import validate_qa_node
from agents.goal_evaluator import goal_evaluator_node


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_classify(state: BrainState) -> str:
    ctx = state.get("oriented_context", {})
    coverage = ctx.get("coverage", "none")
    if coverage == "full":
        return "qa"
    return "search"


def _route_after_search_validation(state: BrainState) -> str:
    if state.get("search_valid"):
        return "store"
    if state.get("retry_count", 0) >= MAX_SEARCH_RETRIES:
        return "clarify"
    return "retry"


def _route_after_qa_validation(state: BrainState) -> str:
    if state.get("qa_approved"):
        return "final"
    if state.get("qa_attempts", 0) >= MAX_QA_ATTEMPTS:
        return "final"
    return "retry"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None) -> StateGraph:
    g = StateGraph(BrainState)

    # Register nodes
    g.add_node("direction", direction_node)
    g.add_node("memory_classify", classify_node)
    g.add_node("search", search_node)
    g.add_node("search_validator", validate_search_node)
    g.add_node("memory_store_knowledge", store_knowledge_node)
    g.add_node("memory_update_coverage", update_coverage_node)
    g.add_node("goal_evaluator", goal_evaluator_node)
    g.add_node("qa_draft", qa_draft_node)
    g.add_node("orchestrator_validate", validate_qa_node)
    g.add_node("qa_final", qa_final_node)
    g.add_node("memory_store_episode", store_episode_node)

    # Entry point → direction → memory_classify
    g.set_entry_point("direction")
    g.add_edge("direction", "memory_classify")

    # memory_classify -> qa or search
    g.add_conditional_edges(
        "memory_classify",
        _route_after_classify,
        {"qa": "qa_draft", "search": "search"},
    )

    # search -> search_validator
    g.add_edge("search", "search_validator")

    # search_validator -> store | retry | clarify
    g.add_conditional_edges(
        "search_validator",
        _route_after_search_validation,
        {
            "store": "memory_store_knowledge",
            "retry": "search",
            "clarify": "goal_evaluator",
        },
    )

    # knowledge stored -> re-assess coverage -> qa
    g.add_edge("memory_store_knowledge", "memory_update_coverage")
    g.add_edge("memory_update_coverage", "qa_draft")

    # goal_evaluator -> episode store -> END
    # Finalise the in-progress placeholder even for clarification turns so
    # subsequent turns can correctly classify as follow_up / elaboration.
    g.add_edge("goal_evaluator", "memory_store_episode")

    # qa_draft -> orchestrator validation
    g.add_edge("qa_draft", "orchestrator_validate")

    # orchestrator -> final | retry
    g.add_conditional_edges(
        "orchestrator_validate",
        _route_after_qa_validation,
        {"final": "qa_final", "retry": "qa_draft"},
    )

    # qa_final -> episode store -> END
    g.add_edge("qa_final", "memory_store_episode")
    g.add_edge("memory_store_episode", END)

    return g.compile(checkpointer=checkpointer)


def get_graph():
    """Return a compiled graph with SQLite checkpointing."""
    import sqlite3

    conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return build_graph(checkpointer=checkpointer)
