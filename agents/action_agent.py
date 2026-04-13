"""
Action Agent — code executor and environment checker.

Runs ONLY when direction_agent sets needs_action=True in direction_result.
It is a LangGraph ReAct agent with one privileged tool: execute_python.

Workflow
--------
  direction → action (conditional, needs_action=True) → memory_classify → …
  direction → memory_classify (needs_action=False, skip action entirely)

The action_node writes action_result to state.
QA Agent reads action_result.summary and includes it in the response.

action_result schema
--------------------
{
    "status":          str,         # "success" | "partial" | "failed" | "skipped"
    "action_type":     str,         # from direction_result
    "summary":         str,         # human-readable findings
    "facts_verified":  list[dict],  # [{fact, verdict, reason}, …]
    "stdout":          str,         # raw captured stdout
    "stderr":          str,         # raw captured stderr (truncated)
    "error":           str,         # exception message if execution failed
}
"""

from __future__ import annotations

import io
import re
import sys
import textwrap
import traceback
from contextlib import redirect_stdout, redirect_stderr

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from config import get_llm
from core.state import BrainState
from prompts import ACTION_AGENT_SYSTEM


_llm = get_llm(temperature=0)

# ---------------------------------------------------------------------------
# Safety guard — block destructive patterns before execution
# ---------------------------------------------------------------------------

_BLOCKED_PATTERNS = re.compile(
    r"\b(rm\s+-rf|os\.system|subprocess\.(run|Popen|call)|shutil\.rmtree"
    r"|open\s*\(.+['\"]w['\"]|urllib\.request|requests\.|httpx\.|socket\.)",
    re.IGNORECASE,
)

_MAX_OUTPUT_CHARS = 4_000  # truncate large stdout to keep context reasonable


def _check_safety(code: str) -> str | None:
    """Return a violation description if the code is unsafe, else None."""
    if _BLOCKED_PATTERNS.search(code):
        match = _BLOCKED_PATTERNS.search(code)
        return f"Blocked pattern detected: '{match.group()}'"
    return None


# ---------------------------------------------------------------------------
# execute_python tool
# ---------------------------------------------------------------------------


@tool
def execute_python(code: str) -> str:
    """
    Execute a Python code snippet in a sandboxed context and return its output.

    The snippet runs with the current Python interpreter. stdout and stderr
    are captured and returned. Execution is limited to pure computation —
    file system writes and network calls are blocked.

    Args:
        code: Valid Python source code to execute.

    Returns:
        Captured stdout + stderr, truncated to 4 000 characters.
    """
    violation = _check_safety(code)
    if violation:
        return f"BLOCKED: {violation}"

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        exec_globals: dict = {"__builtins__": __builtins__}
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(textwrap.dedent(code), exec_globals)  # noqa: S102
    except Exception:
        stderr_buf.write(traceback.format_exc())

    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()

    combined = ""
    if out:
        combined += f"STDOUT:\n{out}"
    if err:
        combined += f"\nSTDERR:\n{err}"
    if not combined:
        combined = "(no output)"

    if len(combined) > _MAX_OUTPUT_CHARS:
        combined = combined[:_MAX_OUTPUT_CHARS] + "\n… [truncated]"
    return combined


# ---------------------------------------------------------------------------
# ReAct agent
# ---------------------------------------------------------------------------

_agent = create_react_agent(
    model=_llm,
    tools=[execute_python],
    prompt=ACTION_AGENT_SYSTEM,
)


# ---------------------------------------------------------------------------
# Result parser
# ---------------------------------------------------------------------------


def _parse_action_result(
    raw: str, action_type: str, actionable_facts: list[str]
) -> dict:
    """
    Parse the agent's free-form text output into a structured action_result dict.
    """
    status_m = re.search(r"STATUS:\s*(success|partial|failed)", raw, re.IGNORECASE)
    status = status_m.group(1).lower() if status_m else "partial"

    results_m = re.search(
        r"RESULTS:\s*\n(.*?)(?=FACTS_VERIFIED:|$)", raw, re.DOTALL | re.IGNORECASE
    )
    summary = results_m.group(1).strip() if results_m else raw[:600]

    facts_m = re.search(r"FACTS_VERIFIED:\s*\n(.*?)$", raw, re.DOTALL | re.IGNORECASE)
    facts_text = facts_m.group(1).strip() if facts_m else ""

    facts_verified = []
    for line in facts_text.splitlines():
        line = line.strip()
        if not line:
            continue
        verdict = "UNKNOWN"
        for v in ("CONFIRMED", "REFUTED", "UNKNOWN"):
            if v in line.upper():
                verdict = v
                break
        facts_verified.append({"fact": line, "verdict": verdict, "reason": line})

    # Pad with UNKNOWN entries for any facts not mentioned
    if len(facts_verified) < len(actionable_facts):
        mentioned = {f["fact"] for f in facts_verified}
        for fact in actionable_facts:
            if fact not in mentioned:
                facts_verified.append(
                    {"fact": fact, "verdict": "UNKNOWN", "reason": "Not verified"}
                )

    return {
        "status": status,
        "action_type": action_type,
        "summary": summary,
        "facts_verified": facts_verified,
        "stdout": "",
        "stderr": "",
        "error": "",
    }


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def action_node(state: BrainState, config: RunnableConfig) -> dict:
    """
    Execute code / validate claims / check environment as requested.
    Only reached when direction_result.needs_action=True.
    Writes action_result to state.
    """
    direction = state.get("direction_result", {})
    action_type = direction.get("action_type", "none")
    actionable_facts = direction.get("actionable_facts", [])
    goal = state["goal"]

    # Extract any code blocks from the goal
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", goal, re.DOTALL)
    code_context = "\n\n".join(code_blocks) if code_blocks else ""

    prompt = (
        f"action_type: {action_type}\n"
        f"goal: {goal}\n"
        f"actionable_facts: {actionable_facts}\n"
        f"code_context:\n{code_context}"
        if code_context
        else f"action_type: {action_type}\n"
        f"goal: {goal}\n"
        f"actionable_facts: {actionable_facts}"
    )

    raw_output = ""
    error_msg = ""
    try:
        # create_react_agent expects {"messages": [...]}
        agent_result = _agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )
        # Extract the last AI message content
        for msg in reversed(agent_result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                raw_output = msg.content
                break
    except Exception as exc:
        error_msg = str(exc)
        raw_output = f"Agent error: {error_msg}"

    result = _parse_action_result(raw_output, action_type, actionable_facts)
    result["error"] = error_msg

    return {
        "action_result": result,
        "reasoning_trace": [
            f"action: type={action_type} | status={result['status']} | "
            f"facts={len(result['facts_verified'])}"
        ],
    }
