#!/usr/bin/env python3
"""
Stop hook: check the reply against memory when it leans on the past.

Models rarely say "I don't remember". They say "as we decided, the boss
has 300 HP" and carry on. This hook reads the finished reply, picks out
sentences that refer back to earlier work or hedge ("I think", "probably",
"if I recall", "I don't have that"), and looks those sentences up in
project memory. When memory holds relevant records the model has not seen
this context window, the hook hands them back once and asks it to check
its reply against them. If everything matches, the model says so in one
line; if not, it corrects itself to the user.

Claude Code contract (Stop):
  stdin  -> {"last_assistant_message", "stop_hook_active", "transcript_path", "session_id", ...}
  stdout -> {"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": "..."}}
Never fires twice in a row (stop_hook_active), and stays silent on errors.
"""

import json
import os
import re
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MAX_CLAIMS = 4
CLAIM_CHARS = 300
TOP_K_PER_CLAIM = 2
SCORE_THRESHOLD = 0.62
RELATIVE_MARGIN = 0.10
MAX_CHECK_CHARS = 2500
LOG_PATH = os.path.join(SCRIPT_DIR, "semantic_memory.log")

_PAST_REFERENCE = re.compile(
    r"\b(?:we|you|i)\s+(?:had\s+)?(?:decided|agreed|chose|settled on|established|said|mentioned|"
    r"wanted|planned|named|set|picked|discussed|went with|defined)\b"
    r"|\b(?:as (?:we )?(?:discussed|decided|agreed|planned|established|mentioned)|earlier|"
    r"previously|originally|last time|last session|before we|already (?:have|has|did|added|"
    r"decided|set|built|made|named)|remember|recall|established|canon(?:ically)?|"
    r"the lore|in the lore|we've been using|existing)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(?:i think|i believe|if i recall|iirc|i'm not sure|i am not sure|not certain|"
    r"i don't (?:remember|recall|have|know)|i do not (?:remember|recall|have)|"
    r"don't have (?:a )?record|no record|probably|likely (?:was|is|were)|might have been|"
    r"i assume|assuming|i'd guess|my guess|should be|presumably)\b",
    re.IGNORECASE,
)
# A sentence that proposes ("should", "let's", "maybe") is an opinion, not a
# recalled fact, so having no record of it says nothing.
_PROPOSAL = re.compile(
    r"\b(?:should|could|would|let's|let us|maybe|perhaps|how about|consider|suggest|"
    r"propose|recommend|might want|i'd)\b",
    re.IGNORECASE,
)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def log_failure(exc):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            handle.write("\n")
    except Exception:
        pass


def memory_claims(reply, limit=MAX_CLAIMS):
    """Sentences of `reply` that depend on remembered context."""
    text = _CODE_FENCE.sub(" ", reply or "")
    claims = []
    for sentence in _SENTENCE.split(text):
        sentence = " ".join(sentence.split()).strip(" -*>#")
        if len(sentence) < 15:
            continue
        if not (_PAST_REFERENCE.search(sentence) or _HEDGE.search(sentence)):
            continue
        claims.append(sentence[:CLAIM_CHARS])
        if len(claims) >= limit:
            break
    return claims


def _has_any_record(db, vec, claim):
    """Whether memory holds anything at all on this claim, live context included."""
    import indexer
    return bool(indexer.search_hits(
        db, vec, top_k=1, threshold=SCORE_THRESHOLD, relative_margin=RELATIVE_MARGIN,
        query_text=claim,
    ))


def build_check(db, claims, exclude, embed):
    import indexer
    vectors = embed([indexer.QUERY_PREFIX + claim for claim in claims])
    blocks = []
    shown = []
    unsupported = []
    seen = set(exclude)
    budget = MAX_CHECK_CHARS
    for claim, vec in zip(claims, vectors):
        if not _has_any_record(db, vec, claim):
            if not _PROPOSAL.search(claim):
                unsupported.append(claim)
            continue
        text, ids = indexer.retrieve_with_ids(
            db,
            vec,
            exclude_uuids=seen,
            top_k=TOP_K_PER_CLAIM,
            threshold=SCORE_THRESHOLD,
            relative_margin=RELATIVE_MARGIN,
            max_chars=budget,
            query_text=claim,
            header=f'You wrote: "{claim}"',
        )
        if not ids:
            continue
        blocks.append(text)
        shown.extend(ids)
        seen.update(ids)
        budget -= len(text)
        if budget <= 200:
            break
    if not blocks and not unsupported:
        return None, []
    parts = [f"{indexer.MEMORY_MARK} · check] Your last reply relied on remembered context."]
    if blocks:
        parts.append(
            "Project memory holds records you have not seen in this context window. "
            "Compare them with what you told the user. If anything you said conflicts "
            "with a record (CANON wins, then the most recent turn), tell the user plainly "
            "what you got wrong and give the corrected answer."
        )
    if unsupported:
        parts.append(
            "Project memory has no record at all for: "
            + "; ".join(f'"{claim}"' for claim in unsupported[:2])
            + ". If that came from a file you read or from the user in this turn, ignore "
            "this. Otherwise tell the user it is your suggestion, not something established."
        )
    parts.append("If everything holds up, reply with one short line saying it was checked against memory.")
    message = " ".join(parts)
    if blocks:
        message += "\n\n" + "\n\n".join(blocks)
    return message, shown


def main():
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception as exc:
        log_failure(exc)
        sys.exit(0)

    if payload.get("stop_hook_active"):
        sys.exit(0)
    claims = memory_claims(payload.get("last_assistant_message") or "")
    if not claims:
        sys.exit(0)

    transcript_path = payload.get("transcript_path")
    try:
        import indexer

        db = indexer.ensure_db()
        indexer.refresh_transcript(db, transcript_path)
        live, epoch = indexer.live_context(transcript_path) if transcript_path else (set(), 0)
        session_key = payload.get("session_id") or (
            os.path.basename(transcript_path).replace(".jsonl", "") if transcript_path else ""
        )
        exclude = live | indexer.already_injected(db, session_key, epoch) | indexer.recently_recalled(db)
        message, shown = build_check(db, claims, exclude, indexer.fastembed_embed)
        if shown and session_key:
            indexer.record_injections(db, session_key, epoch, shown)
        db.close()
    except Exception as exc:
        log_failure(exc)
        sys.exit(0)

    if not message:
        sys.exit(0)
    print(json.dumps({
        "hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": message}
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()
