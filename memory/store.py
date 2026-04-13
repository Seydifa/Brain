"""
Memory store — the brain's long-term memory layer.

Write path
----------
raw text  →  split into chunks (if > CHUNK_SIZE chars)
          →  each chunk described in one sentence by the model
          →  description embedded + stored in ChromaDB with raw chunk as metadata

Read path
---------
query  →  semantic search on description embeddings
       →  return top-K {desc, raw, score} where score ∈ [0, 1]
          (higher = more semantically similar)
"""

import hashlib
from datetime import datetime, UTC

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

CHUNK_SIZE = 800  # characters per chunk

_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

_db = Chroma(
    collection_name="brain",
    embedding_function=_embeddings,
    persist_directory="data/memory",
    collection_metadata={"hnsw:space": "cosine"},  # cosine → scores in [0, 1]
)

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _describe(text: str) -> str:
    """
    One-sentence description of the text.
    This is what gets embedded and searched — not the raw text itself.
    Keeps the semantic index clean and precise.
    """
    return _llm.invoke(
        f"Write a single sentence describing what this text is about:\n\n{text}"
    ).content.strip()


def _split_into_chunks(text: str) -> list[str]:
    """Split text into fixed-size character chunks."""
    return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]


def _make_id(text: str, chunk_index: int) -> str:
    """Stable deterministic document id to avoid duplicates on re-store."""
    digest = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"{digest}_c{chunk_index}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def store(raw: str, source: str = "unknown") -> dict:
    """
    Persist raw text into long-term memory.

    Short text (≤ CHUNK_SIZE) → 1 document.
    Long text               → N chunked documents, each described separately.

    Returns: {"status": "stored", "chunks": int}
    """
    parts = _split_into_chunks(raw) if len(raw) > CHUNK_SIZE else [raw]
    docs = []

    for i, chunk in enumerate(parts):
        description = _describe(chunk)
        docs.append(
            Document(
                page_content=description,  # embedded + searched
                metadata={
                    "raw": chunk,  # returned as context
                    "source": source,
                    "created_at": datetime.now(UTC).isoformat(),
                    "chunk_index": i,
                    "total_chunks": len(parts),
                },
                id=_make_id(chunk, i),
            )
        )

    _db.add_documents(docs)
    return {"status": "stored", "chunks": len(docs)}


def recall(query: str, n: int = 4) -> dict:
    """
    Retrieve the top-N memory chunks most relevant to the query.
    Scores are in [0.0, 1.0] — higher means more semantically similar.

    Returns:
        {"status": "found",  "context": [{"desc", "raw", "score"}, ...]}
        {"status": "empty",  "context": []}
    """
    results = _db.similarity_search_with_relevance_scores(query, k=n)

    if not results:
        return {"status": "empty", "context": []}

    return {
        "status": "found",
        "context": [
            {
                "desc": doc.page_content,
                "raw": doc.metadata["raw"],
                "score": round(score, 3),
            }
            for doc, score in results
        ],
    }
