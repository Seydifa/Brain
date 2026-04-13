"""
Brain state — the shared blackboard for the LangGraph graph.

All agents read from and write to this state. No agent communicates
directly with another: they only read what previous nodes wrote.

Key design: oriented_context
-----------------------------
The Memory Agent populates oriented_context at the start of every turn.
All downstream agents (Search, QA, Orchestrator) receive ONLY the slice
of knowledge that Memory Agent judged relevant — no raw history, no full
store access.

reasoning_trace uses operator.add as its reducer, which means:
  - Nodes return {"reasoning_trace": ["my step"]}
  - LangGraph concatenates, never replaces
  - The final trace is the full audit log of the turn
"""

from __future__ import annotations
from operator import add
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
MEMORY_SCORE_THRESHOLD = 0.65  # min cosine similarity to count as "known"
MAX_SEARCH_RETRIES = 3  # max search + validation loops per turn
MAX_QA_ATTEMPTS = 2  # max QA draft + orchestrator validation loops


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class BrainState(TypedDict):
    # ---- Core ---------------------------------------------------------------
    goal: str  # user's current request
    messages: Annotated[list[BaseMessage], add_messages]  # message history
    response: str  # final delivered answer
    status: str  # "empty" | "partial" | "found" | "done" | "needs_clarification"

    # ---- Direction Agent output (written before memory_classify) -----------
    # Schema: {turn_type, parent_id, bridge_sentence, semantic_sim, method}
    direction_result: dict

    # ---- Memory Agent output ------------------------------------------------
    # The full oriented context — the ONLY state other agents should trust.
    # Schema: {
    #   turn_type, relevant_knowledge, conversation_thread,
    #   coverage, weak_topics,
    #   current_episode_id, parent_episode_id, knowledge_confidence
    # }
    oriented_context: dict

    # ---- Reasoning audit trail (append-only via operator.add) ---------------
    reasoning_trace: Annotated[list[str], add]

    # ---- Search retry tracking ----------------------------------------------
    retry_count: int
    search_valid: bool
    search_feedback: str

    # ---- QA draft / orchestrator loop ---------------------------------------
    qa_draft: str
    qa_approved: bool
    qa_feedback: str
    qa_attempts: int

    # ---- Clarification flow -------------------------------------------------
    needs_clarification: bool
    clarification_reason: str  # one sentence: why the brain can't proceed
    clarification_questions: list[str]
