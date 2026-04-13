"""
Search tools — free, no extra API keys required.

    web_search      → DuckDuckGo (general web)
    academic_search → Semantic Scholar API (papers + abstracts)
"""

import httpx
from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    Returns a plain-text block of title + snippet per result.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return "No web results found."

    return "\n\n".join(f"Title: {r['title']}\n{r['body']}" for r in results)


def academic_search(query: str, max_results: int = 3) -> str:
    """
    Search Semantic Scholar for academic papers (free, no key needed).
    Returns title, authors, year and a short abstract per paper.
    """
    try:
        resp = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "fields": "title,authors,year,abstract",
                "limit": max_results,
            },
            timeout=15,
        )
        resp.raise_for_status()
        papers = resp.json().get("data", [])
    except Exception as exc:
        return f"Academic search failed: {exc}"

    if not papers:
        return "No academic results found."

    lines = []
    for p in papers:
        authors = ", ".join(a["name"] for a in p.get("authors", [])[:2])
        abstract = (p.get("abstract") or "No abstract available.")[:400]
        lines.append(
            f"Authors: {authors} ({p.get('year', '?')})\n"
            f"Title:   {p.get('title', 'Unknown')}\n"
            f"Abstract: {abstract}..."
        )

    return "\n\n".join(lines)
