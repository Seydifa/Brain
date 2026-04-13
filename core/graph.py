"""
Brain graph — the LangGraph wiring of all agent nodes.

Flow
----
Every turn starts at direction which classifies topic continuity + detects
whether the request needs code execution (needs_action).

  direction
    needs_action=True  → action      (run code / validate / check_env)
    needs_action=False → memory_classify (skip execution)

  action → _route_after_action:
    success  → store_solution → memory_classify
    failed + retries left → action  (retry with scratch pad)
    exhausted → memory_classify      (give up, pass failure info to QA)

Coverage drives the first routing decision after memory_classify:

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

from core.state import (
    BrainState,
    MAX_SEARCH_RETRIES,
    MAX_QA_ATTEMPTS,
    MAX_ACTION_RETRIES,
)

from agents.memory_agent import (
    classify_node,
    store_knowledge_node,
    update_coverage_node,
    store_episode_node,
)
from agents.direction_agent import direction_node
from agents.action_agent import action_node
from agents.search_agent import search_node
from agents.search_validator import validate_search_node
from agents.qa_agent import qa_draft_node, qa_final_node
from agents.orchestrator import validate_qa_node
from agents.goal_evaluator import goal_evaluator_node
from memory.store import store as memory_store


# ---------------------------------------------------------------------------
# Solution storage node
# ---------------------------------------------------------------------------


def store_solution_node(state: BrainState) -> dict:
    """
    Persist a successful action solution into long-term memory.

    Called only when action succeeded — stores the solution code + description
    so the brain can recall it in future turns when facing similar problems.
    """
    result = state.get("action_result", {})
    goal = state.get("goal", "")
    solution_code = result.get("solution_code", "")
    solution_desc = result.get("solution_desc", "")
    attempts = state.get("action_attempts", 1)
    scratch = state.get("action_scratch", [])

    # Build a rich solution record for long-term memory
    parts = [
        f"Problem: {goal}",
        f"Action type: {result.get('action_type', '')}",
        f"Attempts needed: {attempts}",
    ]

    # Include the fix journey if retries were needed
    if attempts > 1 and scratch:
        parts.append("\nDebug journey:")
        for entry in scratch:
            if entry.get("diagnosis"):
                parts.append(
                    f"  Attempt {entry.get('attempt', '?')}: {entry['diagnosis']}"
                )

    if solution_code:
        parts.append(f"\nWorking code:\n```python\n{solution_code}\n```")

    if solution_desc:
        parts.append(f"\n{solution_desc}")

    raw_text = "\n".join(parts)
    memory_store(raw_text, source="solution")

    return {
        "reasoning_trace": [
            f"solution stored in long-term memory | "
            f"attempts={attempts} | type={result.get('action_type', '')}"
        ],
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_direction(state: BrainState) -> str:
    """Route to action_node if execution is needed, else skip to memory_classify."""
    direction = state.get("direction_result", {})
    if direction.get("needs_action"):
        return "action"
    return "classify"


def _route_after_action(state: BrainState) -> str:
    """
    Route after action execution:
      success  → store solution in long-term memory
      failed + retries left → retry action with scratch pad
      exhausted → continue to memory_classify with failure info
    """
    result = state.get("action_result", {})
    status = result.get("status", "failed")
    attempts = state.get("action_attempts", 0)

    if status == "success":
        # Only store if there is actual solution code
        if result.get("solution_code"):
            return "store_solution"
        return "classify"

    if attempts < MAX_ACTION_RETRIES and status in ("failed", "partial"):
        return "retry"

    return "classify"


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
        # Search budget exhausted — fall back to LLM knowledge via qa_draft
        # rather than asking for clarification on answerable questions.
        return "qa"
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
    g.add_node("action", action_node)
    g.add_node("store_solution", store_solution_node)
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

    # Entry: direction → action (optional) → memory_classify
    g.set_entry_point("direction")
    g.add_conditional_edges(
        "direction",
        _route_after_direction,
        {"action": "action", "classify": "memory_classify"},
    )

    # Action → route: success → store_solution, retry → action, exhausted → classify
    g.add_conditional_edges(
        "action",
        _route_after_action,
        {
            "store_solution": "store_solution",
            "retry": "action",
            "classify": "memory_classify",
        },
    )
    g.add_edge("store_solution", "memory_classify")

    # memory_classify -> qa or search
    g.add_conditional_edges(
        "memory_classify",
        _route_after_classify,
        {"qa": "qa_draft", "search": "search"},
    )

    # search -> search_validator
    g.add_edge("search", "search_validator")

    # search_validator -> store | retry | qa (LLM fallback when budget exhausted)
    g.add_conditional_edges(
        "search_validator",
        _route_after_search_validation,
        {
            "store": "memory_store_knowledge",
            "retry": "search",
            "qa": "qa_draft",
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
    compiled = build_graph(checkpointer=checkpointer)
    # Raise the recursion limit from the default (25) to accommodate
    # the action retry loop (up to MAX_ACTION_RETRIES=5) plus all other nodes.
    return compiled.with_config(recursion_limit=100)
