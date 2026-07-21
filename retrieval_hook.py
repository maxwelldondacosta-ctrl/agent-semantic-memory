#!/usr/bin/env python3
"""
UserPromptSubmit hook: incrementally indexes the current session's new
turns, then injects retrieved context from this project's semantic memory
index before each turn.

Claude Code contract:
  stdin  -> JSON with at least {"user_prompt": "...", "transcript_path": "...", ...}
  stdout -> {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                     "additionalContext": "..."}}
  exit 0 on success (even a no-op). Never blocks the prompt.

Runs on every message. Only the current session's transcript file is
re-scanned each time (fast — indexer.py's dedup by msg_uuid makes repeat
scans of already-indexed turns a no-op), not the whole project history.

Fails silent and fast on any error — this must never break a real prompt.
"""

import json
import math
import os
import sqlite3
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

TOP_K = 4
SCORE_THRESHOLD = 0.45
MIN_PROMPT_LEN = 8
CONTEXT_WINDOW = 2
MAX_INJECTED_CHARS = 3000

# Transcript-size nudge: this hook only exists in projects with semantic
# indexing wired up, so a /clear suggestion here is always safe — retrieval
# backfills anything relevant on the next message. No auto-compact threshold
# knob exists in Claude Code itself, so this is a soft nudge, not a hard cap.
NUDGE_THRESHOLD_BYTES = 2_000_000  # ~2MB transcript


def build_nudge(transcript_path):
    try:
        size = os.path.getsize(transcript_path)
    except OSError:
        return None
    if size < NUDGE_THRESHOLD_BYTES:
        return None
    return (
        f"[System note: this session's transcript is large ({size / 1_000_000:.1f}MB). "
        "Semantic memory indexing is active for this project, so running /clear now "
        "would be safe — relevant history stays retrievable on demand. If a natural "
        "task boundary has been reached, consider proactively suggesting /clear to "
        "the user rather than carrying the full transcript forward.]"
    )


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(db, query_vec):
    rows = db.execute(
        "SELECT id, session_id, role, text, timestamp, vector FROM messages WHERE vector IS NOT NULL"
    ).fetchall()
    if not rows:
        return None

    scored = []
    for row in rows:
        vec = json.loads(row[5])
        sim = cosine_similarity(query_vec, vec)
        if sim >= SCORE_THRESHOLD:
            scored.append((sim, row[0], row[1], row[2], row[3]))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOP_K]
    if not top:
        return None

    seen = set()
    context_msgs = []
    for sim, vid, session_id, role, text in top:
        before = db.execute(
            "SELECT role, text, id FROM messages WHERE session_id=? AND id<? ORDER BY id DESC LIMIT ?",
            (session_id, vid, CONTEXT_WINDOW),
        ).fetchall()
        after = db.execute(
            "SELECT role, text, id FROM messages WHERE session_id=? AND id>? ORDER BY id ASC LIMIT ?",
            (session_id, vid, CONTEXT_WINDOW),
        ).fetchall()
        window = list(reversed(before)) + [(role, text, vid)] + list(after)
        for r_role, r_text, r_id in window:
            if r_id not in seen:
                seen.add(r_id)
                context_msgs.append((r_id, r_role, r_text))

    context_msgs.sort(key=lambda x: x[0])
    parts = [f"[{role}] {text[:500]}" for _id, role, text in context_msgs]
    return "\n\n".join(parts)[:MAX_INJECTED_CHARS]


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)
        payload = json.loads(raw)
        prompt = (payload.get("user_prompt") or "").strip()
        transcript_path = payload.get("transcript_path")

        import indexer  # local module, same dir

        db = indexer.ensure_db()

        # Incrementally index just the current session's transcript — fast,
        # dedup'd by msg_uuid, keeps the index fresh every single turn.
        if transcript_path and os.path.exists(transcript_path):
            current_msgs = list(indexer.extract_messages(transcript_path))
            to_index = indexer.get_unindexed_messages(db, current_msgs)
            if to_index:
                indexer.index_messages(db, to_index)

        nudge = build_nudge(transcript_path) if transcript_path else None

        context_text = None
        if len(prompt) >= MIN_PROMPT_LEN:
            query_vec = indexer.fastembed_embed([prompt])[0]
            context_text = retrieve(db, query_vec)
        db.close()

        pieces = []
        if nudge:
            pieces.append(nudge)
        if context_text:
            pieces.append(f"[Project Semantic Memory — relevant past context]\n{context_text}")

        if not pieces:
            sys.exit(0)

        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "\n\n".join(pieces),
            }
        }))
        sys.exit(0)
    except Exception:
        # Never break the user's prompt over a retrieval/indexing failure.
        sys.exit(0)


if __name__ == "__main__":
    main()
