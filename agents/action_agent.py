"""
Action Agent — code executor and environment checker with scratch-pad retry.

Runs ONLY when direction_agent sets needs_action=True in direction_result.
It is a LangGraph ReAct agent with one privileged tool: execute_python.

Retry protocol (scratch pad = short-term debug memory)
------------------------------------------------------
On first call, action_scratch is empty.
On failure: the node appends {attempt, code, stdout, stderr, error, diagnosis}
to action_scratch and increments action_attempts.  The graph routes back
to this node; the scratch pad is included in the LLM prompt so it never
repeats the same mistake.
On success: action_result includes solution_code and solution_desc for
long-term storage by the store_solution node downstream.

action_result schema
--------------------
{
    "status":          str,         # "success" | "partial" | "failed"
    "action_type":     str,         # from direction_result
    "summary":         str,         # human-readable findings
    "facts_verified":  list[dict],  # [{fact, verdict, reason}, …]
    "stdout":          str,         # raw captured stdout
    "stderr":          str,         # raw captured stderr (truncated)
    "error":           str,         # exception message if execution failed
    "solution_code":   str,         # final working code (only on success)
    "solution_desc":   str,         # one-line description of the solution
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

from core.config import get_llm
from core.state import BrainState
from core.prompts import ACTION_AGENT_SYSTEM


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
# Scratch pad formatter
# ---------------------------------------------------------------------------


def _format_scratch_pad(scratch: list[dict]) -> str:
    """Format previous attempts for the LLM prompt."""
    if not scratch:
        return ""
    lines = ["=== SCRATCH PAD (previous attempts) ==="]
    for entry in scratch:
        lines.append(f"\n--- Attempt {entry.get('attempt', '?')} ---")
        if entry.get("code"):
            lines.append(f"Code:\n```python\n{entry['code']}\n```")
        if entry.get("stdout"):
            lines.append(f"Stdout: {entry['stdout'][:500]}")
        if entry.get("stderr"):
            lines.append(f"Stderr: {entry['stderr'][:500]}")
        if entry.get("error"):
            lines.append(f"Error: {entry['error'][:300]}")
        if entry.get("diagnosis"):
            lines.append(f"Diagnosis: {entry['diagnosis']}")
    lines.append("\n=== END SCRATCH PAD ===")
    lines.append("You MUST fix the issues above. Do NOT repeat the same code.")
    return "\n".join(lines)


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

    # Extract diagnosis
    diag_m = re.search(
        r"DIAGNOSIS:\s*\n?(.*?)(?=CODE:|RESULTS:|FACTS_VERIFIED:|$)",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    diagnosis = diag_m.group(1).strip() if diag_m else ""

    # Extract code block
    code_m = re.search(
        r"CODE:\s*\n```(?:python)?\n(.*?)```", raw, re.DOTALL | re.IGNORECASE
    )
    code_block = code_m.group(1).strip() if code_m else ""

    results_m = re.search(
        r"RESULTS:\s*\n(.*?)(?=FACTS_VERIFIED:|SOLUTION:|$)",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    summary = results_m.group(1).strip() if results_m else raw[:600]

    # Extract solution (only on success)
    solution_m = re.search(r"SOLUTION:\s*\n(.*?)$", raw, re.DOTALL | re.IGNORECASE)
    solution_text = solution_m.group(1).strip() if solution_m else ""
    # Extract code from solution if it contains a code block
    sol_code_m = re.search(r"```(?:python)?\n(.*?)```", solution_text, re.DOTALL)
    solution_code = sol_code_m.group(1).strip() if sol_code_m else solution_text

    facts_m = re.search(
        r"FACTS_VERIFIED:\s*\n(.*?)(?=SOLUTION:|$)", raw, re.DOTALL | re.IGNORECASE
    )
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
        "solution_code": solution_code if status == "success" else "",
        "solution_desc": "",
        "diagnosis": diagnosis,
        "code": code_block,
    }


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def action_node(state: BrainState, config: RunnableConfig) -> dict:
    """
    Execute code / validate claims / check environment.

    On each call:
      - Reads scratch_pad from state (previous failed attempts).
      - Builds prompt with scratch context so LLM learns from errors.
      - On failure: appends attempt to action_scratch for next retry.
      - On success: writes solution_code + solution_desc for long-term storage.

    The graph routes back here on failure (up to MAX_ACTION_RETRIES).
    """
    direction = state.get("direction_result", {})
    action_type = direction.get("action_type", "none")
    actionable_facts = direction.get("actionable_facts", [])
    goal = state["goal"]
    scratch = state.get("action_scratch", [])
    attempt = state.get("action_attempts", 0) + 1

    # Extract any code blocks from the goal
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", goal, re.DOTALL)
    code_context = "\n\n".join(code_blocks) if code_blocks else ""

    # Build prompt
    parts = [
        f"action_type: {action_type}",
        f"goal: {goal}",
        f"actionable_facts: {actionable_facts}",
        f"attempt: {attempt}",
    ]
    if code_context:
        parts.append(f"code_context:\n{code_context}")

    scratch_text = _format_scratch_pad(scratch)
    if scratch_text:
        parts.append(scratch_text)

    prompt = "\n".join(parts)

    raw_output = ""
    error_msg = ""
    try:
        agent_result = _agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )
        for msg in reversed(agent_result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                raw_output = msg.content
                break
    except Exception as exc:
        error_msg = str(exc)
        raw_output = f"Agent error: {error_msg}"

    result = _parse_action_result(raw_output, action_type, actionable_facts)
    result["error"] = error_msg or result.get("error", "")

    # Build scratch entry for this attempt
    scratch_entry = {
        "attempt": attempt,
        "code": result.get("code", ""),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
        "error": result["error"],
        "diagnosis": result.get("diagnosis", ""),
    }

    # Generate solution description on success
    if result["status"] == "success" and result.get("solution_code"):
        result["solution_desc"] = (
            f"Solution for: {goal[:100]}. Action: {action_type}. Attempts: {attempt}."
        )

    return {
        "action_result": result,
        "action_scratch": [scratch_entry],
        "action_attempts": attempt,
        "reasoning_trace": [
            f"action: attempt={attempt} | type={action_type} | "
            f"status={result['status']} | facts={len(result['facts_verified'])}"
        ],
    }
