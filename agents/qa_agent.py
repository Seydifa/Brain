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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from core.state import BrainState


_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

_SYSTEM = SystemMessage(
    content=(
        "You are the brain's response agent. "
        "Write a clear, well-structured answer to the user's goal using ONLY "
        "the provided context.\n\n"
        "Format rules:\n"
        "  QUESTION   -> direct answer + numbered Sources section\n"
        "  HOW-TO     -> numbered steps\n"
        "  COMPARISON -> markdown table\n"
        "  FOLLOW-UP  -> reference the previous answer, then extend it\n"
        "Never invent facts not present in the context. "
        "If context is insufficient, say so explicitly."
    )
)


def _format_oriented_context(ctx: dict) -> str:
    """
    Build the context block that QA sees.
    Includes: conversation thread (for follow-ups) + relevant knowledge chunks.
    Nothing else.
    """
    parts = []

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
    context_text = _format_oriented_context(ctx)
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

    resp = _llm.invoke([_SYSTEM, HumanMessage(content=user_content)])

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
