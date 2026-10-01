#!/usr/bin/env python3
"""
UserPromptSubmit hook: incrementally indexes the current session, then
injects raw turns that are *not* already in the model's live context.

Claude Code keeps everything after the last compact_boundary in the prompt
and replaces everything before it with a summary. Re-injecting the hot zone
only burns the context budget on text the model just saw. This hook embeds
the new prompt and retrieves pre-compaction turns from this session, plus
turns from other sessions, and passes a short excerpt as additionalContext.

Claude Code contract:
  stdin  -> JSON with at least {"user_prompt": "...", "transcript_path": "...", ...}
  stdout -> {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                     "additionalContext": "..."}}
  exit 0 on success (even a no-op). Never blocks the prompt.

Only the current session file is re-scanned. Dedup is by message UUID and
chunk index. Failures are logged and swallowed — a retrieval bug must never
break a real prompt.
"""

import json
import os
import re
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

TOP_K = 4
# Measured on bge-small-en-v1.5 with the query prefix: related turns land
# around 0.65-0.80, unrelated ones up to ~0.59.
SCORE_THRESHOLD = 0.62
RELATIVE_MARGIN = 0.10
KEYWORD_FLOOR = 0.55
MIN_PROMPT_LEN = 8
CONTEXT_WINDOW = 2
MAX_INJECTED_CHARS = 3000
LOG_PATH = os.path.join(SCRIPT_DIR, "semantic_memory.log")

# Short or anaphoric prompts ("do that again") embed poorly on their own.
# Fold in a little of the live tail so the query points at the real topic,
# while retrieval still refuses to echo that tail back.
_ANAPHORA = re.compile(
    r"\b(it|that|this|those|them|same|again|previous|above|continue)\b",
    re.IGNORECASE,
)


def log_failure(exc):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            handle.write("\n")
    except Exception:
        pass


def build_nudge(transcript_path, threshold):
    try:
        size = os.path.getsize(transcript_path)
    except OSError:
        return None
    if size < threshold:
        return None
    return (
        f"[System note: this session's transcript is large ({size / 1_000_000:.1f}MB). "
        "Semantic memory is active, so running /clear is safe — turns that fall out "
        "of the live prompt stay retrievable. If a task boundary has been reached, "
        "suggest /clear rather than carrying the full transcript forward.]"
    )


def _hot_zone_tail(transcript_path, exclude_uuids, limit=2, chars=300):
    """Recent live turns, used only to disambiguate the query — never injected."""
    if not transcript_path or not os.path.exists(transcript_path):
        return ""
    import indexer
    texts = []
    for _uuid, _session, role, _ts, text in indexer.extract_messages(transcript_path):
        if _uuid not in exclude_uuids:
            continue
        snippet = " ".join(text.split())
        if len(snippet) > chars:
            snippet = snippet[:chars]
        texts.append(f"[{role}] {snippet}")
    return "\n".join(texts[-limit:])


def build_query(prompt, transcript_path, exclude_uuids):
    import indexer
    body = prompt
    if len(prompt) < 80 or _ANAPHORA.search(prompt):
        tail = _hot_zone_tail(transcript_path, exclude_uuids)
        if tail and tail not in prompt:
            body = "Recent live context:\n" + tail + "\n\nCurrent request:\n" + prompt
    return indexer.QUERY_PREFIX + body


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)
        payload = json.loads(raw)
    except Exception as exc:
        log_failure(exc)
        sys.exit(0)

    prompt = (payload.get("user_prompt") or "").strip()
    transcript_path = payload.get("transcript_path")

    try:
        import indexer

        db = indexer.ensure_db()
        if transcript_path and os.path.exists(transcript_path):
            current = list(indexer.extract_messages(transcript_path))
            pending = indexer.get_unindexed_messages(db, current)
            if pending:
                indexer.index_messages(db, pending)
        live, epoch = indexer.live_context(transcript_path) if transcript_path else (set(), 0)
        session_key = payload.get("session_id") or (
            os.path.basename(transcript_path).replace(".jsonl", "") if transcript_path else ""
        )
        nudge = build_nudge(transcript_path, indexer.LARGE_TRANSCRIPT_BYTES) if transcript_path else None

        context_text = None
        if len(prompt) >= MIN_PROMPT_LEN:
            # Earlier injections this epoch are still in the prompt as hook context.
            exclude = live | indexer.already_injected(db, session_key, epoch)
            query = build_query(prompt, transcript_path, live)
            query_vec = indexer.fastembed_embed([query])[0]
            context_text, injected = indexer.retrieve_with_ids(
                db,
                query_vec,
                exclude_uuids=exclude,
                top_k=TOP_K,
                threshold=SCORE_THRESHOLD,
                context_window=CONTEXT_WINDOW,
                max_chars=MAX_INJECTED_CHARS,
                query_text=prompt,
                keyword_floor=KEYWORD_FLOOR,
                relative_margin=RELATIVE_MARGIN,
            )
            if injected and session_key:
                indexer.record_injections(db, session_key, epoch, injected)
        db.close()
    except Exception as exc:
        log_failure(exc)
        sys.exit(0)

    pieces = []
    if nudge:
        pieces.append(nudge)
    if context_text:
        pieces.append(context_text)
    if not pieces:
        sys.exit(0)

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n\n".join(pieces),
        }
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()
