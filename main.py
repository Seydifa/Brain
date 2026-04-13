"""
Brain — entry point.

Runs the multi-agent brain in a conversational REPL loop.
Every turn is a fresh graph invocation on the same thread_id so LangGraph
checkpointing preserves message history across turns.

On clarification turns: the brain cannot answer because search failed.
The user is prompted to answer the clarifying questions and the loop
continues with the enriched goal.

Debug mode: set BRAIN_DEBUG=1 in the environment to see the full
reasoning_trace and oriented_context metadata after each turn.
"""

import os
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from core.graph import get_graph

_EMPTY_STATE = {
    "goal": "",
    "messages": [],
    "response": "",
    "status": "empty",
    "direction_result": {},
    "action_result": {},
    "action_scratch": [],
    "action_attempts": 0,
    "oriented_context": {},
    "reasoning_trace": [],
    "retry_count": 0,
    "search_valid": False,
    "search_feedback": "",
    "qa_draft": "",
    "qa_approved": False,
    "qa_feedback": "",
    "qa_attempts": 0,
    "needs_clarification": False,
    "clarification_reason": "",
    "clarification_questions": [],
}

DEBUG = os.getenv("BRAIN_DEBUG", "0") == "1"


def _print_debug(result: dict) -> None:
    ctx = result.get("oriented_context", {})
    trace = result.get("reasoning_trace", [])
    print("\n--- DEBUG ---")
    print(f"Turn type : {ctx.get('turn_type', '?')}")
    print(f"Coverage  : {ctx.get('coverage', '?')}")
    print(f"Confidence: {ctx.get('knowledge_confidence', 0):.2f}")
    print(f"Episode   : {ctx.get('current_episode_id', '?')}")
    print("Reasoning trace:")
    for step in trace:
        print(f"  [{step}]")
    print("-------------\n")


def run():
    graph = get_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    Path("data").mkdir(exist_ok=True)
    print("Brain is ready. Type 'exit' to quit.\n")

    while True:
        goal = input("You: ").strip()
        if not goal or goal.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        state = {**_EMPTY_STATE, "goal": goal}
        for attempt in range(4):
            try:
                result = graph.invoke(state, config=config)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 10 * (attempt + 1)
                    print(f"  [rate limit, retrying in {wait}s...]")
                    time.sleep(wait)
                else:
                    raise
        else:
            print("\nBrain: (rate limit exceeded, please try again in a moment)\n")
            continue

        if DEBUG:
            _print_debug(result)

        # Clarification loop — search exhausted, brain needs more info
        while result.get("needs_clarification"):
            reason = result.get("clarification_reason", "")
            questions = result.get("clarification_questions", [])
            print("\nBrain needs clarification:")
            if reason:
                print(f"  Why: {reason}")
            for q in questions:
                print(f"  - {q}")

            clarification = input("\nYour clarification: ").strip()
            if not clarification:
                break

            enriched_goal = f"{goal} — clarification: {clarification}"
            state = {**_EMPTY_STATE, "goal": enriched_goal}
            for attempt in range(4):
                try:
                    result = graph.invoke(state, config=config)
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait = 10 * (attempt + 1)
                        print(f"  [rate limit, retrying in {wait}s...]")
                        time.sleep(wait)
                    else:
                        raise

            if DEBUG:
                _print_debug(result)

        response = result.get("response", "")
        if response:
            print(f"\nBrain: {response}\n")
        else:
            print("\nBrain: (no response produced)\n")


if __name__ == "__main__":
    run()
