"""
Search Agent — the brain's information gatherer.

Reads the oriented_context produced by Memory Agent to decide HOW to search:

  coverage=none    -> search the full goal
  coverage=partial -> search only weak_topics (to fill gaps)
  retry (retry_count > 0) -> reformulate query using search_feedback

Two tools available:
  web_search_tool       DuckDuckGo — fast, broad coverage
  academic_search_tool  Semantic Scholar — peer-reviewed papers

The agent returns the combined result as a single message for the
search_validator to evaluate.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

from core.state import BrainState
from prompts import SEARCH_REACT_SYSTEM


_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def web_search_tool(query: str) -> str:
    """Search the web using DuckDuckGo. Returns a summary of top results."""
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found."
        return "\n\n".join(
            f"[{r['title']}]\n{r['body']}\nURL: {r['href']}" for r in results
        )
    except Exception as e:
        return f"Web search error: {e}"


@tool
def academic_search_tool(query: str) -> str:
    """Search Semantic Scholar for academic papers. Returns titles + abstracts."""
    import httpx

    try:
        params = {
            "query": query,
            "limit": 5,
            "fields": "title,abstract,year,authors",
        }
        resp = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            timeout=10,
        )
        papers = resp.json().get("data", [])
        if not papers:
            return "No academic papers found."
        return "\n\n".join(
            f"[{p['title']} ({p.get('year', '?')})] "
            f"{p.get('abstract', 'No abstract.')[:400]}"
            for p in papers
        )
    except Exception as e:
        return f"Academic search error: {e}"


_react_agent = create_react_agent(
    _llm, [web_search_tool, academic_search_tool], prompt=SEARCH_REACT_SYSTEM
)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def search_node(state: BrainState) -> dict:
    """
    Formulate and execute a search based on oriented_context.

    Routing logic:
    - Retry turn    -> prepend feedback to query for reformulation
    - Partial cover -> search only weak_topics from oriented_context
    - New/none      -> search the full goal
    """
    goal = state["goal"]
    retry = state.get("retry_count", 0)
    feedback = state.get("search_feedback", "")
    ctx = state.get("oriented_context", {})
    coverage = ctx.get("coverage", "none")
    weak = ctx.get("weak_topics", [])

    # Determine search query
    if retry > 0 and feedback:
        query = f"{goal} — focusing on: {feedback}"
    elif coverage == "partial" and weak:
        query = f"{goal} — specifically about: {', '.join(weak)}"
    else:
        query = goal

    result = _react_agent.invoke({"messages": [("user", query)]})

    last_msg = result["messages"][-1]
    search_output = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    return {
        "messages": [AIMessage(content=search_output)],
        "status": "partial",
        "reasoning_trace": [
            f"search attempt {retry + 1} | coverage={coverage} | "
            f"query={'feedback-reformulated' if retry > 0 else 'fresh'}"
        ],
    }
