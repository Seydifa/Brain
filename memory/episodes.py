"""
Episode Store — the brain's diary.

Every completed conversational turn is stored as a structured episode.
Episodes form a chain: follow-up turns reference their parent via follow_up_of.
This gives Memory Agent a full conversation timeline without requiring
a separate chat history structure.

Schema
------
id              TEXT   — unique episode id  (ep_<md5[:8]>_<yyyymmddHHMMSS>)
timestamp       TEXT   — UTC ISO-8601
user_request    TEXT   — original goal for this turn
chosen_response TEXT   — final response delivered
reasoning_trace TEXT   — JSON array of reasoning steps collected during the turn
flags           TEXT   — JSON array of strings (e.g. "follow_up", "partial_coverage", "in_progress")
topic_cluster   TEXT   — primary topic label (first 3 words of goal)
follow_up_of    TEXT   — parent episode id (NULL for new topics)
turn_type       TEXT   — "new_topic" | "follow_up" | "elaboration" | "clarification" | "correction"
knowledge_conf  REAL   — best knowledge confidence score at recall time
"""

import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = "data/episodes.db"


def _conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id              TEXT PRIMARY KEY,
            timestamp       TEXT NOT NULL,
            user_request    TEXT NOT NULL,
            chosen_response TEXT,
            reasoning_trace TEXT,
            flags           TEXT,
            topic_cluster   TEXT,
            follow_up_of    TEXT,
            turn_type       TEXT,
            knowledge_conf  REAL
        )
    """)
    conn.commit()
    return conn


def make_episode_id(user_request: str) -> str:
    """Generate a deterministic-enough but unique episode id."""
    digest = hashlib.md5(user_request.encode()).hexdigest()[:8]
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"ep_{digest}_{ts}"


def save_episode(
    episode_id: str,
    user_request: str,
    chosen_response: str = "",
    reasoning_trace: list[str] | None = None,
    flags: list[str] | None = None,
    topic_cluster: str = "",
    follow_up_of: str | None = None,
    turn_type: str = "new_topic",
    knowledge_conf: float = 0.0,
) -> str:
    """
    Persist or update an episode.

    Call with chosen_response="" and flags=["in_progress"] to register a
    placeholder at the start of a turn. Call again with the same id once
    the turn is complete to finalize it (UPSERT semantics).
    """
    conn = _conn()
    conn.execute(
        """
        INSERT INTO episodes
            (id, timestamp, user_request, chosen_response,
             reasoning_trace, flags, topic_cluster,
             follow_up_of, turn_type, knowledge_conf)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            chosen_response = excluded.chosen_response,
            reasoning_trace = excluded.reasoning_trace,
            flags           = excluded.flags,
            topic_cluster   = excluded.topic_cluster,
            follow_up_of    = excluded.follow_up_of,
            turn_type       = excluded.turn_type,
            knowledge_conf  = excluded.knowledge_conf
        """,
        (
            episode_id,
            datetime.utcnow().isoformat(),
            user_request,
            chosen_response,
            json.dumps(reasoning_trace or []),
            json.dumps(flags or []),
            topic_cluster,
            follow_up_of,
            turn_type,
            knowledge_conf,
        ),
    )
    conn.commit()
    conn.close()
    return episode_id


def get_recent(n: int = 5) -> list[dict]:
    """Fetch the N most recent episodes, newest first."""
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_by_id(episode_id: str) -> dict | None:
    """Fetch a single episode by id. Returns None if not found."""
    conn = _conn()
    row = conn.execute(
        "SELECT * FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None
