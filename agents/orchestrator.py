"""
Orchestrator — quality gate on QA output.

Scores the draft on a 1-10 scale across three criteria:
  - Accuracy     : does the answer stay within the provided knowledge?
  - Completeness : does it fully address the user's goal?
  - Clarity      : is the answer well-structured and easy to read?

Decision:
  score >= 7  -> approve  (qa_approved = True)
  score <  7  -> reject with actionable feedback (attempt budget permitting)
  budget exhausted -> force approve to avoid infinite loops
"""

import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from core.state import BrainState, MAX_QA_ATTEMPTS
from prompts import ORCHESTRATOR_SYSTEM


_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)


def _parse(text: str) -> tuple[int, str]:
    nums = re.findall(r"\d+", text[:120])
    scores = [int(n) for n in nums[:3] if 1 <= int(n) <= 10]
    avg = round(sum(scores) / len(scores)) if scores else 5
    fb_match = re.search(r"FEEDBACK:\s*(.+)", text, re.IGNORECASE)
    feedback = fb_match.group(1).strip() if fb_match else ""
    return avg, feedback


def validate_qa_node(state: BrainState) -> dict:
    """
    Evaluate the QA draft. Return approval decision and optional feedback.
    Appends to reasoning_trace so Memory Agent can log what happened.
    """
    attempts = state.get("qa_attempts", 0)

    # Budget exhausted — force approve to prevent infinite loops
    if attempts >= MAX_QA_ATTEMPTS:
        return {
            "qa_approved": True,
            "qa_feedback": "",
            "reasoning_trace": [
                f"qa budget exhausted at attempt {attempts} — force approved"
            ],
        }

    ctx = state.get("oriented_context", {})
    draft = state.get("qa_draft", "")
    knowledge = ctx.get("relevant_knowledge", [])
    knowledge_text = (
        "\n".join(f"- {c['desc']}: {c['raw'][:200]}" for c in knowledge)
        or "No knowledge chunks."
    )

    resp = _llm.invoke(
        [
            ORCHESTRATOR_SYSTEM,
            HumanMessage(
                content=(
                    f"User goal: {state['goal']}\n\n"
                    f"Provided knowledge:\n{knowledge_text}\n\n"
                    f"Draft answer:\n{draft}"
                )
            ),
        ]
    )

    score, feedback = _parse(resp.content)
    approved = score >= 7

    return {
        "qa_approved": approved,
        "qa_feedback": feedback if not approved else "",
        "qa_attempts": attempts + 1,
        "reasoning_trace": [f"qa scored {score}/10 | approved={approved}"],
    }
