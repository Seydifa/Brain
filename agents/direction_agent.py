"""
Direction Agent — conversational direction intelligence.

Determines the structural direction of each turn using two layers:

Layer 1 — Hard signals (no LLM needed, instant):
  • Empty episode history                              → new_topic
  • Explicit disclaimer regex ("setting X aside", etc.)→ new_topic + bridge
  • Cosine similarity < NEW_TOPIC_THRESHOLD            → new_topic + bridge

Layer 2 — LLM disambiguation (similarity in ambiguous range 0.40–0.65):
  • LLM decides: follow_up / elaboration / clarification / correction / new_topic

Action classification (runs in parallel with Layer 1/2):
  • Regex + LLM detect whether the request requires code execution / validation.
  • Sets needs_action=True and action_type in direction_result.
  • Downstream action_node runs only when needs_action=True.

Action types
------------
  run_code    — user wants code executed, tested, or a script produced and run
  validate    — user wants a factual claim, output, or piece of code verified
  check_env   — user wants system / environment / dependency info
  none        — pure knowledge question, no execution needed

Public API
----------
direction_node(state, config)              LangGraph entry node.
classify_direction(goal, thread_id)        Pure function — usable outside graph.

direction_result schema
-----------------------
{
    "turn_type":       str,     # new_topic | follow_up | elaboration | ...
    "parent_id":       str | None,
    "bridge_sentence": str,     # "" if same topic or no prior episodes
    "semantic_sim":    float,   # cosine(goal_emb, best_recent_emb), 0–1
    "method":          str,     # "empty_history" | "disclaimer" | "semantic" | "llm"
    "needs_action":    bool,    # True if request needs code execution / validation
    "action_type":     str,     # "run_code" | "validate" | "check_env" | "none"
    "actionable_facts": list,   # extracted constraints / assertions from the goal
}
"""

from __future__ import annotations

import re
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from core.config import get_llm, get_embeddings
from core.state import BrainState
from memory.episodes import get_recent
from core.prompts import DIRECTION_SYSTEM, ACTION_DETECT_SYSTEM


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Cosine similarity below this → definitely new topic, no LLM needed
NEW_TOPIC_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Module-level singletons — built from config so backend-agnostic
# ---------------------------------------------------------------------------

_llm = get_llm(temperature=0, role="purpose")
_embeddings = get_embeddings()

# ---------------------------------------------------------------------------
# Explicit topic-change disclaimer patterns
# These phrases signal an intentional subject switch regardless of similarity
# ---------------------------------------------------------------------------

_DISCLAIMER_PATTERNS = [
    r"\bsetting .{2,50}? aside\b",
    r"\bswitching (back )?to\b",
    r"\b(completely |totally |entirely )?different (question|topic|subject)\b",
    r"\bchanging (the )?subject\b",
    r"\bforget(ting)? .{2,50}? (for now|for a moment|a moment)\b",
    r"\bleaving .{2,50}? (behind|aside|for now)\b",
    r"\bapart from .{2,60}?,\b",
    r"\binstead[,]? let'?s\b",
    r"\bnow (I want to |let'?s )(talk|ask|discuss|explore)\b",
    r"\bon (a )?different (note|topic|subject)\b",
    r"\bunrelated (question|to)\b",
    r"\bjump(ing)? (to|over to)\b",
    r"\bmov(ing|e) on to\b",
    r"\bshift(ing)? (to|our focus)\b",
    r"\bnew (line of |)?question\b",
]

_DISCLAIMER_RE = re.compile("|".join(_DISCLAIMER_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# Action-detection patterns (Layer 1, no LLM)
# Detect requests that require code execution, validation, or env checks
# ---------------------------------------------------------------------------

_RUN_CODE_PATTERNS = [
    r"\b(run|execute|eval(uate)?|test)\b.{0,40}\b(code|script|function|snippet|cell)\b",
    r"\b(write|create|generate)\b.{0,30}\b(script|program|class|function)\b.{0,30}\band (run|test|execute)\b",
    r"\bcan you (run|test|try|execute)\b",
    r"\b(implement|code).{0,20}\band (verify|check|test)\b",
]

_VALIDATE_PATTERNS = [
    r"\b(verify|validate|check|confirm|assert|is (this|that|it) correct)\b",
    r"\b(does this (work|compile|pass)|will this (work|run|compile))\b",
    r"\b(is (this|the) (output|result|answer) (correct|right|accurate))\b",
    r"\b(sanity.check|double.check|make sure)\b",
]

_CHECK_ENV_PATTERNS = [
    r"\b(what (version|packages?|modules?|dependencies|libs?)\b)",
    r"\b(check (my|the|our) (environment|setup|config|dependencies|versions?))\b",
    r"\b(is .{2,30} installed)\b",
    r"\b(show (me )?the installed|list (the )?packages)\b",
    r"\b(pip list|pip show|import .{2,30} as)\b",
    r"\b(which python|python --version|sys\.version)\b",
]

_ACTION_PATTERNS: dict[str, list[str]] = {
    "run_code": _RUN_CODE_PATTERNS,
    "validate": _VALIDATE_PATTERNS,
    "check_env": _CHECK_ENV_PATTERNS,
}

_ACTION_RE: dict[str, re.Pattern] = {
    atype: re.compile("|".join(patterns), re.IGNORECASE)
    for atype, patterns in _ACTION_PATTERNS.items()
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_disclaimer(goal: str) -> bool:
    """Return True if the goal contains an explicit topic-change signal."""
    return bool(_DISCLAIMER_RE.search(goal))


def _detect_action(goal: str) -> tuple[bool, str]:
    """
    Layer-1 action detection using regex.
    Returns (needs_action, action_type). action_type is 'none' when False.
    Priority order: run_code > validate > check_env.
    """
    for atype in ("run_code", "validate", "check_env"):
        if _ACTION_RE[atype].search(goal):
            return True, atype
    return False, "none"


def _extract_actionable_facts(goal: str) -> list[str]:
    """
    Extract concrete constraints, assertions, or checkable claims from the goal.
    These become the action_node's work items.

    Examples:
      "Run this and verify the output is 42" → ["output should be 42"]
      "Check if numpy >= 1.24 is installed"  → ["numpy >= 1.24 installed"]
    """
    facts = []

    # Numbers / version assertions
    version_m = re.findall(r"\b[\w\-]+\s*[><=!]+\s*[\d\.]+", goal)
    facts.extend(version_m)

    # "should be / must be / equals" claims
    claim_m = re.findall(
        r"(?:output|result|value|return(?:s)?)\s+(?:is|should be|equals?|==)\s+['\"]?[\w\.\-]+['\"]?",
        goal,
        re.IGNORECASE,
    )
    facts.extend(claim_m)

    # File / module names after "check if ... is installed"
    install_m = re.findall(r"(?:is\s+)?([\w\-\.]+)\s+installed", goal, re.IGNORECASE)
    facts.extend([f"{m} installed" for m in install_m if len(m) > 1])

    return list(dict.fromkeys(facts))  # deduplicate preserving order


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _embed(text: str) -> list[float]:
    """Embed a single text string."""
    return _embeddings.embed_query(text)


def _topic_label(episode: dict) -> str:
    """Short human-readable topic label for an episode."""
    cluster = (episode.get("topic_cluster") or "").strip()
    if cluster:
        return cluster
    req = episode.get("user_request") or ""
    words = req.split()[:6]
    return " ".join(words) + ("…" if len(req.split()) > 6 else "")


def _build_bridge_sentence(goal: str, prev_episode: dict, method: str) -> str:
    """
    Template-based transition sentence — no extra LLM call.
    Acknowledges what was being discussed and frames the topic change.
    """
    prev_label = _topic_label(prev_episode)
    new_words = goal.split()[:7]
    new_label = " ".join(new_words) + ("…" if len(goal.split()) > 7 else "")

    if method == "disclaimer":
        return (
            f"I see you're setting aside our discussion on **{prev_label}** — "
            f"your new question (*{new_label}*) starts a fresh topic."
        )
    return (
        f"We were discussing **{prev_label}**; "
        f"your question now shifts to a new topic — starting fresh."
    )


def _llm_classify(
    goal: str,
    recent_episodes: list[dict],
    semantic_sim: float,
) -> tuple[str, str | None]:
    """
    Ask the Direction LLM to disambiguate when the semantic similarity is
    in the ambiguous range (0.40–0.65). Returns (turn_type, parent_id | None).
    """
    history = "\n".join(
        f"[{ep['id']}] {ep['user_request'][:120]}" for ep in recent_episodes[:3]
    )
    resp = _llm.invoke(
        [
            DIRECTION_SYSTEM,
            HumanMessage(
                content=(
                    f"Current request: {goal}\n"
                    f"Semantic similarity to most recent episode: {semantic_sim:.3f}\n\n"
                    f"Recent episode history:\n{history}"
                )
            ),
        ]
    )

    content = resp.content if hasattr(resp, "content") else str(resp)
    type_m = re.search(r"TYPE:\s*(\w+)", content, re.IGNORECASE)
    parent_m = re.search(r"PARENT:\s*(\S+)", content, re.IGNORECASE)

    turn_type = type_m.group(1).lower() if type_m else "new_topic"
    raw_parent = parent_m.group(1) if parent_m else "null"
    parent_id = None if raw_parent.lower() == "null" else raw_parent
    return turn_type, parent_id


def _llm_detect_action(goal: str) -> tuple[bool, str]:
    """
    LLM fallback for action detection when regex gives no clear signal.
    Used only when the goal contains ambiguous action language.
    Returns (needs_action, action_type).
    """
    resp = _llm.invoke(
        [
            ACTION_DETECT_SYSTEM,
            HumanMessage(content=f"User request: {goal}"),
        ]
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    action_m = re.search(
        r"ACTION:\s*(run_code|validate|check_env|none)", content, re.IGNORECASE
    )
    action_type = action_m.group(1).lower() if action_m else "none"
    return action_type != "none", action_type


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_direction(goal: str, thread_id: str = "") -> dict:
    """
    Classify the conversational direction of the current goal.

    Two-layer topic classification + action detection.

    Returns a direction_result dict (see module docstring for schema).
    """
    recent_episodes = get_recent(n=5, thread_id=thread_id)

    # ── Action detection (runs independently, always) ────────────────────────
    needs_action, action_type = _detect_action(goal)
    actionable_facts = _extract_actionable_facts(goal) if needs_action else []

    # ── Layer 1a: empty history → new_topic ─────────────────────────────────
    if not recent_episodes:
        return {
            "turn_type": "new_topic",
            "parent_id": None,
            "bridge_sentence": "",
            "semantic_sim": 0.0,
            "method": "empty_history",
            "needs_action": needs_action,
            "action_type": action_type,
            "actionable_facts": actionable_facts,
        }

    # ── Layer 1b: explicit disclaimer regex → new_topic + bridge ────────────
    if _detect_disclaimer(goal):
        bridge = _build_bridge_sentence(goal, recent_episodes[0], method="disclaimer")
        return {
            "turn_type": "new_topic",
            "parent_id": None,
            "bridge_sentence": bridge,
            "semantic_sim": 0.0,
            "method": "disclaimer",
            "needs_action": needs_action,
            "action_type": action_type,
            "actionable_facts": actionable_facts,
        }

    # ── Layer 1c: semantic distance ──────────────────────────────────────────
    # Embed goal once and compare against up to 3 recent episodes.
    # Bug fix: always pick the best-matching episode, even if ep[0] has no request text.
    goal_emb = _embed(goal)

    best_sim = -1.0
    best_ep = recent_episodes[0]

    for ep in recent_episodes[:3]:
        req_text = ep.get("user_request", "").strip()
        if not req_text:
            continue
        sim = _cosine_similarity(goal_emb, _embed(req_text))
        if sim > best_sim:
            best_sim = sim
            best_ep = ep

    # If all recent episodes had empty request text, fall back
    if best_sim < 0:
        best_sim = 0.0

    # Hard cut: too far → new_topic, skip LLM
    if best_sim < NEW_TOPIC_THRESHOLD:
        bridge = _build_bridge_sentence(goal, recent_episodes[0], method="semantic")
        return {
            "turn_type": "new_topic",
            "parent_id": None,
            "bridge_sentence": bridge,
            "semantic_sim": best_sim,
            "method": "semantic",
            "needs_action": needs_action,
            "action_type": action_type,
            "actionable_facts": actionable_facts,
        }

    # ── Layer 2: LLM disambiguation (0.40–1.0 range) ────────────────────────
    turn_type, parent_id = _llm_classify(goal, recent_episodes, best_sim)
    bridge = (
        _build_bridge_sentence(goal, recent_episodes[0], method="semantic")
        if turn_type == "new_topic"
        else ""
    )
    return {
        "turn_type": turn_type,
        "parent_id": parent_id or (best_ep["id"] if turn_type != "new_topic" else None),
        "bridge_sentence": bridge,
        "semantic_sim": best_sim,
        "method": "llm",
        "needs_action": needs_action,
        "action_type": action_type,
        "actionable_facts": actionable_facts,
    }


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def direction_node(state: BrainState, config: RunnableConfig) -> dict:
    """
    LangGraph entry node. Runs before memory_classify every turn.
    Writes direction_result to state.
    Routes to action_node (via graph conditional) when needs_action=True.
    """
    goal = state["goal"]
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")

    result = classify_direction(goal, thread_id=thread_id)

    action_note = ""
    if result["needs_action"]:
        action_note = (
            f" | action={result['action_type']}"
            f" | facts={len(result['actionable_facts'])}"
        )

    return {
        "direction_result": result,
        "reasoning_trace": [
            f"direction: {result['turn_type']} | "
            f"method={result['method']} | "
            f"sim={result['semantic_sim']:.3f} | "
            f"bridge={'yes' if result['bridge_sentence'] else 'no'}"
            f"{action_note}"
        ],
    }
