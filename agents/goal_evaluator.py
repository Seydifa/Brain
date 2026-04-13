"""
Goal Evaluator — last resort when all search retries are exhausted.

Generates 2–3 targeted clarifying questions for the user explaining
specifically what information is needed to proceed.

Sets:
    state["needs_clarification"]    = True
    state["clarification_questions"] = [question, ...]
    state["status"]                 = "needs_clarification"

main.py detects the needs_clarification flag, presents the questions,
collects the user's answer, then re-invokes the brain with an enriched
goal (original goal + "User clarification: <answer>").
"""

import re

from langchain_core.messages import HumanMessage

from config import get_llm
from core.state import BrainState
from prompts import GOAL_EVALUATOR_SYSTEM

_llm = get_llm(temperature=0.3)


def _parse_response(text: str) -> tuple[str, list[str]]:
    # Handle plain and markdown-bold variants: WHY:, **WHY:**, **WHY**:, etc.
    why_match = re.search(r"\*{0,2}WHY\*{0,2}:\**\s*(.+)", text, re.IGNORECASE)
    reason = why_match.group(1).strip().rstrip("*") if why_match else ""
    # Only explicitly numbered lines count as questions (ignores preamble / WHY line)
    questions = [
        m.group(1).strip()
        for m in re.finditer(r"^\s*\d+[.\)\]]\s*(.+)", text, re.MULTILINE)
        if len(m.group(1).strip()) > 5
    ]
    return reason, questions[:3]


def goal_evaluator_node(state: BrainState) -> dict:
    """
    Produce clarifying questions based on the goal and the last search feedback.

    Returns:
        needs_clarification      True
        clarification_questions  list[str]
        status                   "needs_clarification"
    """
    resp = _llm.invoke(
        [
            GOAL_EVALUATOR_SYSTEM,
            HumanMessage(
                content=(
                    f"User goal: {state['goal']}\n\n"
                    f"Last search feedback: {state.get('search_feedback', 'no specific feedback')}"
                )
            ),
        ]
    )

    reason, questions = _parse_response(resp.content)

    return {
        "needs_clarification": True,
        "clarification_reason": reason,
        "clarification_questions": questions,
        "status": "needs_clarification",
    }
