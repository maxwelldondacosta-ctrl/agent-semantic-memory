#!/usr/bin/env python3
"""
Project-local semantic indexer for Claude Code session transcripts.

Reads this project's Claude Code JSONL transcripts from
~/.claude/projects/<encoded-project-path>/*.jsonl, embeds user/assistant
text turns with fastembed (local ONNX, no server), stores in a local
SQLite DB for semantic retrieval by retrieval_hook.py.

Usage:
  python3 indexer.py              # index all unindexed messages
  python3 indexer.py --reindex    # force re-index everything
"""

import json
import os
import re
import sys
import time
import sqlite3
import glob

from fastembed import TextEmbedding

# ── Config ──────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DB = os.path.join(SCRIPT_DIR, "semantic_memory.db")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, fast, ~33MB via ONNX
BATCH_SIZE = 16
MAX_TEXT_CHARS = 4000  # truncate very long turns before embedding


def encode_project_path(path):
    """Match Claude Code's project-dir encoding: non-alphanumeric -> '-'."""
    return re.sub(r"[^a-zA-Z0-9]", "-", path)


TRANSCRIPTS_DIR = os.path.join(
    os.path.expanduser("~/.claude/projects"), encode_project_path(PROJECT_DIR)
)

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def ensure_db():
    db = sqlite3.connect(INDEX_DB)
    db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_uuid    TEXT UNIQUE,
            session_id  TEXT,
            role        TEXT,
            timestamp   TEXT,
            text        TEXT,
            vector      BLOB,
            indexed_at  REAL
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
    db.commit()
    return db


def fastembed_embed(texts):
    embedder = get_embedder()
    return [vec.tolist() for vec in embedder.embed(texts)]


def extract_text_from_content(content):
    """Claude Code message.content is either a plain string (user turns)
    or a list of content blocks (assistant turns, sometimes user turns
    with attachments). Extract only text blocks — skip tool_use/tool_result
    noise to keep the index focused on conversational content."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()
    return ""


def extract_messages(transcript_path):
    """Parse a Claude Code transcript JSONL file and yield
    (uuid, session_id, role, timestamp, text) for user/assistant text turns."""
    session_id = os.path.basename(transcript_path).replace(".jsonl", "")
    with open(transcript_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            obj_type = obj.get("type")
            if obj_type not in ("user", "assistant"):
                continue

            message = obj.get("message", {})
            role = message.get("role", obj_type)
            text = extract_text_from_content(message.get("content", ""))
            if not text:
                continue

            yield (
                obj.get("uuid"),
                obj.get("sessionId", session_id),
                role,
                obj.get("timestamp", ""),
                text[:MAX_TEXT_CHARS],
            )


def get_unindexed_messages(db, all_messages):
    indexed = {row[0] for row in db.execute("SELECT msg_uuid FROM messages WHERE vector IS NOT NULL")}
    return [m for m in all_messages if m[0] not in indexed]


def index_messages(db, messages):
    if not messages:
        return 0
    texts = [m[4] for m in messages]
    vectors = fastembed_embed(texts)
    now = time.time()
    cur = db.cursor()
    for msg, vec in zip(messages, vectors):
        msg_uuid, session_id, role, ts, text = msg
        vec_blob = json.dumps(vec).encode("utf-8")
        cur.execute(
            """INSERT OR REPLACE INTO messages
               (msg_uuid, session_id, role, timestamp, text, vector, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (msg_uuid, session_id, role, ts, text, vec_blob, now),
        )
    db.commit()
    return len(messages)


def main():
    reindex = "--reindex" in sys.argv

    print(f"Project transcripts dir: {TRANSCRIPTS_DIR}")
    if not os.path.isdir(TRANSCRIPTS_DIR):
        print("No transcripts directory found for this project yet — nothing to index.")
        return

    transcript_files = sorted(glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.jsonl")))
    print(f"Found {len(transcript_files)} transcript file(s)")

    db = ensure_db()

    all_messages = []
    for tf in transcript_files:
        count = 0
        for msg in extract_messages(tf):
            all_messages.append(msg)
            count += 1
        print(f"  {os.path.basename(tf)}: {count} text turns")

    print(f"\nTotal text turns extracted: {len(all_messages)}")

    if reindex:
        db.execute("DELETE FROM messages")
        db.commit()
        print("Re-index: cleared existing index")
        to_index = all_messages
    else:
        to_index = get_unindexed_messages(db, all_messages)

    print(f"To index: {len(to_index)}")

    if not to_index:
        print("Already up to date.")
        db.close()
        return

    total_indexed = 0
    for i in range(0, len(to_index), BATCH_SIZE):
        batch = to_index[i:i + BATCH_SIZE]
        total_indexed += index_messages(db, batch)
        print(f"  Progress: {total_indexed}/{len(to_index)}")

    db.execute("VACUUM")
    db.close()

    db2 = sqlite3.connect(INDEX_DB)
    total = db2.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    sessions = db2.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()[0]
    db2.close()
    print(f"\nDone. Total indexed: {total} turns across {sessions} sessions -> {INDEX_DB}")


if __name__ == "__main__":
    main()
