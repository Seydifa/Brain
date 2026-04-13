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

from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

from config import get_llm
from core.state import BrainState
from prompts import SEARCH_REACT_SYSTEM


_llm = get_llm(temperature=0)


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


@tool
def wikipedia_tool(query: str) -> str:
    """Retrieve encyclopedic background from Wikipedia. Best for definitions,
    mechanisms, history, and well-established foundational concepts."""
    import httpx, urllib.parse

    try:
        slug = urllib.parse.quote(query.strip().replace(" ", "_"))
        resp = httpx.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
            timeout=8,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            d = resp.json()
            return (
                f"[Wikipedia: {d.get('title', query)}]\n"
                f"{d.get('extract', 'No extract.')[:1000]}\n"
                f"URL: {d.get('content_urls', {}).get('desktop', {}).get('page', 'N/A')}"
            )
        # Fallback: full-text Wikipedia search
        r2 = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            timeout=8,
        )
        results = r2.json().get("query", {}).get("search", [])
        if results:
            slug2 = urllib.parse.quote(results[0]["title"].replace(" ", "_"))
            r3 = httpx.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug2}",
                timeout=8,
                follow_redirects=True,
            )
            if r3.status_code == 200:
                d = r3.json()
                return (
                    f"[Wikipedia: {d.get('title')}]\n"
                    f"{d.get('extract', 'No extract.')[:1000]}\n"
                    f"URL: {d.get('content_urls', {}).get('desktop', {}).get('page', 'N/A')}"
                )
        return f"Wikipedia: no page found for '{query}'."
    except Exception as e:
        return f"Wikipedia error: {e}"


@tool
def arxiv_tool(query: str) -> str:
    """Search arXiv for cutting-edge preprints. Best for AI/ML, quantum
    computing, physics, mathematics, and CS research from 2020 onwards."""
    import httpx, re

    try:
        resp = httpx.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "max_results": 5,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=12,
        )
        entries = re.findall(r"<entry>(.*?)</entry>", resp.text, re.DOTALL)
        if not entries:
            return "No arXiv papers found."
        out = []
        for e in entries[:4]:
            title_m = re.search(r"<title>(.*?)</title>", e, re.DOTALL)
            summ_m = re.search(r"<summary>(.*?)</summary>", e, re.DOTALL)
            pub_m = re.search(r"<published>(.*?)</published>", e)
            t = title_m.group(1).strip().replace("\n", " ") if title_m else "Unknown"
            s = summ_m.group(1).strip()[:350] if summ_m else "No abstract."
            y = pub_m.group(1)[:4] if pub_m else "?"
            out.append(f"[arXiv {y}] {t}\n{s}")
        return "\n\n".join(out)
    except Exception as e:
        return f"arXiv error: {e}"


_react_agent = create_react_agent(
    _llm,
    [web_search_tool, academic_search_tool, wikipedia_tool, arxiv_tool],
    prompt=SEARCH_REACT_SYSTEM,
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
