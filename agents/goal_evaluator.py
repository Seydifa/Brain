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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.state import BrainState

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

_SYSTEM = SystemMessage(
    content=(
        "The search agent failed to find sufficient information for the user's goal "
        "after multiple attempts.\n"
        "Generate 2–3 short, targeted clarifying questions that would help narrow "
        "the search and address the specific gaps identified.\n"
        "Return ONLY a numbered list of questions, nothing else."
    )
)


def _parse_questions(text: str) -> list[str]:
    lines = text.strip().split("\n")
    questions = []
    for line in lines:
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if len(line) > 5:
            questions.append(line)
    return questions[:3]


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
            _SYSTEM,
            HumanMessage(
                content=(
                    f"User goal: {state['goal']}\n\n"
                    f"Last search feedback: {state.get('search_feedback', 'no specific feedback')}"
                )
            ),
        ]
    )

    questions = _parse_questions(resp.content)

    return {
        "needs_clarification": True,
        "clarification_questions": questions,
        "status": "needs_clarification",
    }
