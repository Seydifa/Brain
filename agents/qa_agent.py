"""
QA Agent - the brain's only output voice.

Receives ONLY the oriented_context from Memory Agent - never raw state,
never full history. This protects the context window and ensures answers
are always grounded in what Memory Agent judged relevant.

Two nodes:

  qa_draft_node   Generate a draft from oriented_context.
                  On retry: also receives orchestrator feedback.
                  Includes conversation_thread for follow-up continuity.

  qa_final_node   Promote the approved draft to state["response"].
                  No LLM call - orchestrator already validated the draft.
"""

from langchain_core.messages import HumanMessage, AIMessage

from core.config import get_llm
from core.state import BrainState
from core.prompts import QA_SYSTEM


_llm = get_llm(temperature=0.3, role="purpose")


def _format_oriented_context(
    ctx: dict, action_result: dict | None = None, action_scratch: list | None = None
) -> str:
    """
    Build the context block that QA sees.
    Includes: bridge sentence, action results (with retry history), conversation thread, knowledge chunks.
    """
    parts = []

    # Bridge sentence — only present on topic changes (from direction_agent)
    bridge = ctx.get("bridge_sentence", "")
    if bridge:
        parts.append(f"=== Topic transition ===\n{bridge}")
        parts.append("")

    # Action result — present when direction_agent detected needs_action=True
    if action_result and action_result.get("status") not in (None, "skipped"):
        action_type = action_result.get("action_type", "")
        summary = action_result.get("summary", "")
        facts = action_result.get("facts_verified", [])
        status = action_result.get("status", "")
        solution_code = action_result.get("solution_code", "")
        parts.append(f"=== Action results ({action_type} — {status}) ===")
        if summary:
            parts.append(summary)
        if facts:
            parts.append("\nVerified facts:")
            for f in facts:
                parts.append(
                    f"  • {f.get('fact', '')} → {f.get('verdict', '?')} — {f.get('reason', '')}"
                )
        # Show retry journey if there were multiple attempts
        if action_scratch and len(action_scratch) > 1:
            parts.append(f"\nRetry journey ({len(action_scratch)} attempts):")
            for entry in action_scratch:
                diag = entry.get("diagnosis", "")
                if diag:
                    parts.append(f"  Attempt {entry.get('attempt', '?')}: {diag}")
        # Show final solution code
        if solution_code and status == "success":
            parts.append(f"\nFinal working code:\n```python\n{solution_code}\n```")
            parts.append("\nSolution stored in long-term memory.")
        parts.append("")

    # Conversation thread - only present for follow-up/elaboration turns
    thread = ctx.get("conversation_thread", [])
    if thread:
        parts.append("=== Previous conversation ===")
        for t in thread:
            parts.append(f"User asked: {t['request']}")
            if t.get("response"):
                parts.append(f"Brain answered: {t['response'][:600]}...")
        parts.append("")

    # Relevant knowledge chunks
    chunks = ctx.get("relevant_knowledge", [])
    if chunks:
        parts.append("=== Relevant knowledge ===")
        for i, c in enumerate(chunks, 1):
            score = c.get("score", "?")
            parts.append(
                f"[{i}] (relevance: {score})\nSummary: {c['desc']}\nContent: {c['raw']}"
            )

    return "\n\n".join(parts) if parts else "No context available."


def qa_draft_node(state: BrainState) -> dict:
    """
    Generate a draft answer from the oriented context.

    First attempt: goal + oriented_context only.
    Retry:         also includes orchestrator feedback for targeted improvement.
    """
    ctx = state.get("oriented_context", {})
    feedback = state.get("qa_feedback", "")
    action_result = state.get("action_result") or None
    action_scratch = state.get("action_scratch") or None
    context_text = _format_oriented_context(ctx, action_result, action_scratch)
    turn_type = ctx.get("turn_type", "new_topic")

    user_content = (
        f"Turn type: {turn_type}\n"
        f"User goal: {state['goal']}\n\n"
        f"Context:\n{context_text}"
    )

    if feedback:
        user_content += (
            f"\n\n---\n"
            f"Previous draft was rejected.\n"
            f"Feedback: {feedback}\n"
            "Address the feedback specifically in your revised answer."
        )

    resp = _llm.invoke([QA_SYSTEM, HumanMessage(content=user_content)])

    return {
        "qa_draft": resp.content,
        "reasoning_trace": [
            f"qa draft generated | turn={turn_type} | feedback={'yes' if feedback else 'no'}"
        ],
    }


def qa_final_node(state: BrainState) -> dict:
    """
    Promote the approved draft as the final response.
    No LLM call - the draft is already validated by the orchestrator.
    """
    return {
        "response": state["qa_draft"],
        "messages": [AIMessage(content=state["qa_draft"])],
        "reasoning_trace": ["qa draft approved, promoted to final response"],
    }
