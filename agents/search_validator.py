"""
Search Validator — quality gate between search and memory storage.

Checks whether the search result is:
  - Relevant     : does it address the user's actual goal?
  - Complete     : does it contain enough information to answer?
  - From source  : can a specific source be identified?

Decision:
  valid    -> forward to memory storage
  retry    -> search again with feedback (if budget allows)
  clarify  -> search budget exhausted, escalate to goal_evaluator
"""

import re
from langchain_core.messages import HumanMessage

from core.config import get_llm
from core.state import BrainState, MAX_SEARCH_RETRIES
from core.prompts import SEARCH_VALIDATOR_SYSTEM


_llm = get_llm(temperature=0, role="purpose")


def validate_search_node(state: BrainState) -> dict:
    """
    Validate the most recent search result against the user's goal.
    Appends to reasoning_trace.
    """
    retry = state.get("retry_count", 0)

    # Retrieve last tool message (search result)
    last_msg = next(
        (m for m in reversed(state.get("messages", [])) if hasattr(m, "content")),
        None,
    )
    result_text = last_msg.content[:1200] if last_msg else ""

    resp = _llm.invoke(
        [
            SEARCH_VALIDATOR_SYSTEM,
            HumanMessage(
                content=(f"User goal: {state['goal']}\n\nSearch result:\n{result_text}")
            ),
        ]
    )

    valid_match = re.search(r"VALID:\s*(yes|no)", resp.content, re.IGNORECASE)
    fb_match = re.search(r"FEEDBACK:\s*(.+)", resp.content, re.IGNORECASE)

    valid = valid_match.group(1).lower() == "yes" if valid_match else False
    feedback = fb_match.group(1).strip() if fb_match else ""

    if valid:
        return {
            "search_valid": True,
            "search_feedback": feedback,
            "reasoning_trace": [f"search valid=True | {feedback[:60]}"],
        }

    # Check retry budget
    if retry >= MAX_SEARCH_RETRIES:
        return {
            "search_valid": False,
            "search_feedback": feedback,
            "retry_count": retry + 1,
            "reasoning_trace": [
                f"search budget exhausted after {retry} retries | {feedback[:60]}"
            ],
        }

    return {
        "search_valid": False,
        "search_feedback": feedback,
        "retry_count": retry + 1,
        "reasoning_trace": [f"search valid=False retry {retry + 1} | {feedback[:60]}"],
    }
