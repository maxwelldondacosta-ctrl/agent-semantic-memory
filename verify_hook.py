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


def canon_entities(db):
    """[(key, topic, fact, [(alias, case_sensitive)])] for labelled canon topics.

    The full topic matches case-insensitively. When a topic is one proper
    noun plus lowercase descriptors ("Vargan stats", "Elara backstory"),
    that noun alone also matches, with its capital. A multi-word name such
    as "Frost Caverns" must match in full, so "Frost spells" does not.
    """
    entities = []
    for key, topic, fact in db.execute("SELECT topic_key, topic, fact FROM canon WHERE topic IS NOT NULL"):
        aliases = [(topic, False)]
        words = re.findall(r"[A-Za-z][\w'-]+", topic)
        proper = [word for word in words if word[0].isupper()]
        if len(words) > 1 and len(proper) == 1 and len(proper[0]) >= 3:
            aliases.append((proper[0], True))
        entities.append((key, topic, fact, aliases))
    return entities


def _mentions(sentence, aliases):
    for alias, case_sensitive in aliases:
        flags = 0 if case_sensitive else re.IGNORECASE
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", sentence, flags):
            return True
    return False


def find_claims(reply, entities=(), limit=MAX_CLAIMS):
    """Sentences worth checking: past references, hedges, and canon mentions.

    Each claim is {"text", "cue", "entities"}. `cue` is True for sentences
    flagged by wording; canon mentions need no particular wording, which is
    what catches a flat, confident "Vargan has 300 HP."
    """
    text = _CODE_FENCE.sub(" ", reply or "")
    claims = []
    for sentence in _SENTENCE.split(text):
        sentence = " ".join(sentence.split()).strip(" -*>#")
        if len(sentence) < 12:
            continue
        cue = bool(_PAST_REFERENCE.search(sentence) or _HEDGE.search(sentence))
        mentioned = [entity for entity in entities if _mentions(sentence, entity[3])]
        if not cue and not mentioned:
            continue
        claims.append({"text": sentence[:CLAIM_CHARS], "cue": cue, "entities": mentioned})
        if len(claims) >= limit:
            break
    return claims


def memory_claims(reply, limit=MAX_CLAIMS):
    """Wording-flagged sentences only (no canon lookup)."""
    return [claim["text"] for claim in find_claims(reply, (), limit)]


_NUM_UNIT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|[A-Za-z][A-Za-z-]*)")
_UNIT_NUM = re.compile(
    r"\b(level|lvl|tier|act|chapter|wave|stage|rank|floor|phase|zone|world|room)\s+(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_ATTRIBUTE = re.compile(
    r"\b(weak(?:ness)?|vulnerab(?:le|ility)|immun(?:e|ity)|resist(?:ant|ance)?)"
    r"\s+(?:is\s+|are\s+)?(?:to\s+|against\s+)?([a-z]+)",
    re.IGNORECASE,
)
_NOT_UNITS = frozenset("""
a an and or the to of in on at for by with from as is are was were be it its this that
than then times x per but so if into over under after before about vs
""".split())


def _normalize_unit(unit):
    unit = unit.lower()
    if unit in ("lvl",):
        return "level"
    if len(unit) > 3 and unit.endswith("s") and not unit.endswith("ss"):
        unit = unit[:-1]
    return unit


def _attribute_kind(word):
    word = word.lower()
    if word.startswith(("weak", "vulnerab")):
        return "weak to"
    if word.startswith("immun"):
        return "immune to"
    return "resistant to"


def extract_values(text):
    """{unit or attribute: {values}} for concrete, checkable details."""
    values = {}
    for number, unit in _NUM_UNIT.findall(text or ""):
        unit = _normalize_unit(unit)
        if unit in _NOT_UNITS or unit in ("level", "tier", "act", "chapter", "wave", "stage", "rank"):
            continue
        values.setdefault(unit, set()).add(float(number))
    for unit, number in _UNIT_NUM.findall(text or ""):
        values.setdefault(_normalize_unit(unit), set()).add(float(number))
    for kind, target in _ATTRIBUTE.findall(text or ""):
        if target.lower() in _NOT_UNITS:
            continue
        values.setdefault(_attribute_kind(kind), set()).add(target.lower())
    return values


def _show(value):
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    return value


def find_conflicts(claims):
    """Concrete contradictions between a sentence and canon about a named topic.

    A sentence that also states the canon value is describing a change
    ("raised from 450 to 600 HP"), and a proposal is not a claim; neither
    counts. Conflicts are reported even when the canon entry is already in
    the live context: contradicting visible canon is still drift.
    """
    conflicts = []
    for claim in claims:
        if not claim["entities"] or _PROPOSAL.search(claim["text"]):
            continue
        said = extract_values(claim["text"])
        if not said:
            continue
        for _key, topic, fact, _aliases in claim["entities"]:
            canon = extract_values(fact)
            for unit, values in said.items():
                if unit in canon and not (values & canon[unit]):
                    conflicts.append({
                        "claim": claim["text"],
                        "topic": topic,
                        "fact": fact,
                        "unit": unit,
                        "said": sorted(_show(v) for v in values),
                        "canon": sorted(_show(v) for v in canon[unit]),
                    })
    return conflicts


def _has_any_record(db, vec, claim):
    """Whether memory holds anything at all on this claim, live context included."""
    import indexer
    return bool(indexer.search_hits(
        db, vec, top_k=1, threshold=SCORE_THRESHOLD, relative_margin=RELATIVE_MARGIN,
        query_text=claim,
    ))


def build_check(db, claims, exclude, embed):
    """(message or None, uuids shown). `claims` come from find_claims."""
    import indexer
    if not claims:
        return None, []
    conflicts = find_conflicts(claims)
    texts = [claim["text"] for claim in claims]
    vectors = embed([indexer.QUERY_PREFIX + text for text in texts])
    blocks = []
    shown = []
    unsupported = []
    seen = set(exclude)
    budget = MAX_CHECK_CHARS
    for claim, vec in zip(claims, vectors):
        text = claim["text"]
        if not claim["entities"] and not _has_any_record(db, vec, text):
            if claim["cue"] and not _PROPOSAL.search(text):
                unsupported.append(text)
            continue
        excerpt, ids = indexer.retrieve_with_ids(
            db,
            vec,
            exclude_uuids=seen,
            top_k=TOP_K_PER_CLAIM,
            threshold=SCORE_THRESHOLD,
            relative_margin=RELATIVE_MARGIN,
            max_chars=budget,
            query_text=text,
            header=f'You wrote: "{text}"',
        )
        if not ids:
            continue
        blocks.append(excerpt)
        shown.extend(ids)
        seen.update(ids)
        budget -= len(excerpt)
        if budget <= 200:
            break
    if not blocks and not unsupported and not conflicts:
        return None, []

    parts = [f"{indexer.MEMORY_MARK} · check]"]
    if conflicts:
        lines = ["Your last reply contradicts recorded canon:"]
        for item in conflicts[:4]:
            lines.append(
                f'- You wrote "{item["claim"]}", but canon "{item["topic"]}" says '
                f'"{item["fact"]}" ({item["unit"]}: you said {", ".join(map(str, item["said"]))}, '
                f'canon says {", ".join(map(str, item["canon"]))}).'
            )
        lines.append(
            "Tell the user plainly which detail was wrong and give the canon value. If the "
            "user asked for this change in this conversation, it is not a mistake: say so "
            "and update canon with `remember` so memory matches."
        )
        parts.append("\n".join(lines))
    if blocks:
        parts.append(
            "Your reply relied on remembered context, and project memory holds records you "
            "have not seen in this context window. Compare them with what you told the "
            "user. If anything conflicts (CANON wins, then the most recent turn), tell the "
            "user plainly what you got wrong and give the corrected answer."
        )
    if unsupported:
        parts.append(
            "Project memory has no record at all for: "
            + "; ".join(f'"{claim}"' for claim in unsupported[:2])
            + ". If that came from a file you read or from the user in this turn, ignore "
            "this. Otherwise tell the user it is your suggestion, not something established."
        )
    if not conflicts:
        parts.append("If everything holds up, reply with one short line saying it was checked against memory.")
    message = " ".join(parts[:1]) + " " + "\n\n".join(parts[1:])
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
    reply = payload.get("last_assistant_message") or ""
    if not reply.strip():
        sys.exit(0)

    transcript_path = payload.get("transcript_path")
    try:
        import indexer

        db = indexer.ensure_db()
        claims = find_claims(reply, canon_entities(db))
        if not claims:
            db.close()
            sys.exit(0)
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
