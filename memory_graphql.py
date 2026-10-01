#!/usr/bin/env python3
"""
GraphQL view of project memory.

Plain `recall` answers "is there anything about X". This answers questions
with structure: everything linked to a character, the conversation where a
canon fact was decided, what was said just before a turn, one session's
last few turns. The data is the same SQLite store the hooks use.

Usage:
  python3 memory_graphql.py '{ canon(contains: "Vargan") { topic fact links { relation target { topic fact } } } }'
  python3 memory_graphql.py --schema

Requires graphql-core (pip3 install --user graphql-core).
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import indexer  # noqa: E402

SDL = '''
type Query {
  "Hybrid semantic + keyword search over past turns and canon. Empty list = no record."
  search(query: String!, limit: Int = 5, canonOnly: Boolean = false, includeCanon: Boolean = true, session: ID): [Hit!]!
  "Canon facts. topic matches one topic; contains filters topic and fact text."
  canon(topic: String, contains: String, limit: Int = 20): [CanonFact!]!
  "Most recent sessions first."
  sessions(limit: Int = 10): [Session!]!
  session(id: ID!): Session
  turn(uuid: ID!): Turn
}

type Mutation {
  "Record or replace canon for a topic. Links may point at topics not recorded yet."
  remember(fact: String!, topic: String, links: [LinkInput!]): RememberResult!
  link(source: String!, target: String!, relation: String = "related"): Boolean!
  forget(topic: String!): Boolean!
}

input LinkInput {
  topic: String!
  relation: String = "related"
}

type Hit {
  score: Float!
  keywords: [String!]!
  excerpt(maxChars: Int = 480): String!
  "Set when the hit is a canon fact."
  canon: CanonFact
  "Set when the hit is a conversation turn."
  turn: Turn
}

type Turn {
  uuid: ID!
  role: String!
  timestamp: String
  text(maxChars: Int = 600): String!
  session: Session!
  before(count: Int = 2): [Turn!]!
  after(count: Int = 2): [Turn!]!
}

type Session {
  id: ID!
  startedAt: String
  endedAt: String
  turnCount: Int!
  turns(limit: Int = 20, offset: Int = 0, fromEnd: Boolean = false): [Turn!]!
}

type CanonFact {
  key: ID!
  topic: String
  "Null when another fact links to this topic but it was never recorded."
  fact: String
  recordedAt: String
  recorded: Boolean!
  links: [CanonLink!]!
  "Past turns about this fact, e.g. where it was decided or changed."
  discussedIn(limit: Int = 3): [Hit!]!
}

type CanonLink {
  relation: String!
  "out = this fact links to target; in = target links to this fact."
  direction: String!
  target: CanonFact!
}

type RememberResult {
  fact: CanonFact!
  replaced: String
}
'''

MAX_LIMIT = 25
MAX_TEXT_CHARS = 4000
MAX_DEPTH = 8
MAX_RESULT_CHARS = 16000
SEARCH_THRESHOLD = 0.60
SEARCH_MARGIN = 0.12

_schema = None


def get_schema():
    global _schema
    if _schema is None:
        from graphql import build_schema
        _schema = build_schema(SDL)
    return _schema


def _clamp(value, low, high):
    return max(low, min(int(value), high))


def _cut(text, max_chars):
    text = text or ""
    limit = _clamp(max_chars, 1, MAX_TEXT_CHARS)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


class Context:
    def __init__(self, db, embed):
        self.db = db
        self.embed = embed
        self.shown = set()
        self._canon = {}
        self._turns = {}

    def turn(self, msg_uuid):
        if msg_uuid not in self._turns:
            rows = self.db.execute(
                """SELECT id, role, timestamp, text, session_id FROM messages
                   WHERE msg_uuid=? ORDER BY chunk_index""",
                (msg_uuid,),
            ).fetchall()
            self._turns[msg_uuid] = Turn(self, msg_uuid, rows) if rows else None
        return self._turns[msg_uuid]

    def canon_fact(self, key, label=None):
        if key not in self._canon:
            self._canon[key] = CanonFact(self, key, label)
        return self._canon[key]

    def hits(self, query, limit, **filters):
        vec = self.embed([indexer.QUERY_PREFIX + query])[0]
        found = indexer.search_hits(
            self.db, vec, top_k=_clamp(limit, 1, MAX_LIMIT), threshold=SEARCH_THRESHOLD,
            relative_margin=SEARCH_MARGIN, query_text=query, **filters,
        )
        hits = [Hit(self, item) for item in found]
        self.shown.update(item["uuid"] for item in found)
        return hits


class Hit:
    def __init__(self, ctx, item):
        self.ctx = ctx
        self.item = item
        self.score = round(item["score"], 4)
        self.keywords = item["matched"]

    def excerpt(self, info, maxChars=480):
        return indexer._focused_snippet(
            self.item["text"], self.item["matched"], _clamp(maxChars, 40, MAX_TEXT_CHARS)
        )

    def canon(self, info):
        if self.item["session_id"] != indexer.CANON_SESSION:
            return None
        return self.ctx.canon_fact(self.item["uuid"].split(":", 1)[1])

    def turn(self, info):
        if self.item["session_id"] == indexer.CANON_SESSION:
            return None
        return self.ctx.turn(self.item["uuid"])


class Turn:
    def __init__(self, ctx, msg_uuid, rows):
        self.ctx = ctx
        self.uuid = msg_uuid
        self.first_id = rows[0][0]
        self.role = rows[0][1]
        self.timestamp = rows[0][2] or None
        self.session_id = rows[0][4]
        full = rows[0][3] or ""
        for row in rows[1:]:
            full += (row[3] or "")[indexer.CHUNK_OVERLAP:]
        self.full_text = full

    def text(self, info, maxChars=600):
        return _cut(self.full_text, maxChars)

    def session(self, info):
        return Session(self.ctx, self.session_id)

    def _neighbors(self, count, later):
        op, order = (">", "ASC") if later else ("<", "DESC")
        rows = self.ctx.db.execute(
            f"""SELECT msg_uuid, MIN(id) AS first FROM messages
                WHERE session_id=? GROUP BY msg_uuid HAVING first {op} ?
                ORDER BY first {order} LIMIT ?""",
            (self.session_id, self.first_id, _clamp(count, 0, MAX_LIMIT)),
        ).fetchall()
        turns = [self.ctx.turn(row[0]) for row in rows]
        turns = [turn for turn in turns if turn]
        return turns if later else list(reversed(turns))

    def before(self, info, count=2):
        return self._neighbors(count, later=False)

    def after(self, info, count=2):
        return self._neighbors(count, later=True)


class Session:
    def __init__(self, ctx, session_id):
        self.ctx = ctx
        self.id = session_id
        row = ctx.db.execute(
            """SELECT MIN(NULLIF(timestamp, '')), MAX(NULLIF(timestamp, '')), COUNT(DISTINCT msg_uuid)
               FROM messages WHERE session_id=?""",
            (session_id,),
        ).fetchone()
        self.startedAt, self.endedAt, self.turnCount = row[0], row[1], row[2] or 0

    def turns(self, info, limit=20, offset=0, fromEnd=False):
        order = "DESC" if fromEnd else "ASC"
        rows = self.ctx.db.execute(
            f"""SELECT msg_uuid, MIN(id) AS first FROM messages WHERE session_id=?
                GROUP BY msg_uuid ORDER BY first {order} LIMIT ? OFFSET ?""",
            (self.id, _clamp(limit, 1, MAX_LIMIT), max(0, int(offset))),
        ).fetchall()
        turns = [self.ctx.turn(row[0]) for row in rows]
        turns = [turn for turn in turns if turn]
        return list(reversed(turns)) if fromEnd else turns


class CanonFact:
    def __init__(self, ctx, key, label=None):
        self.ctx = ctx
        self.key = key
        row = ctx.db.execute(
            "SELECT topic, fact, recorded_at FROM canon WHERE topic_key=?", (key,)
        ).fetchone()
        self.recorded = bool(row)
        self.topic = (row[0] if row else None) or label or key
        self.fact = row[1] if row else None
        self.recordedAt = row[2] if row else None
        if self.recorded:
            ctx.shown.add("canon:" + key)

    def links(self, info):
        db = self.ctx.db
        out = [
            {"relation": rel, "direction": "out", "target": self.ctx.canon_fact(dst, label)}
            for dst, rel, label in db.execute(
                "SELECT dst, relation, dst_topic FROM canon_links WHERE src=? ORDER BY relation, dst",
                (self.key,),
            )
        ]
        incoming = [
            {"relation": rel, "direction": "in", "target": self.ctx.canon_fact(src)}
            for src, rel in db.execute(
                "SELECT src, relation FROM canon_links WHERE dst=? ORDER BY relation, src", (self.key,)
            )
        ]
        return out + incoming

    def discussedIn(self, info, limit=3):
        if not self.fact:
            return []
        return self.ctx.hits(
            f"{self.topic}: {self.fact}", limit, skip_sessions={indexer.CANON_SESSION}
        )


class Root:
    """Query and mutation resolvers. graphql-core calls methods as (info, **args)."""

    def __init__(self, ctx):
        self.ctx = ctx

    def search(self, info, query, limit=5, canonOnly=False, includeCanon=True, session=None):
        filters = {}
        if canonOnly:
            filters["sessions"] = {indexer.CANON_SESSION}
        elif session:
            filters["sessions"] = {session}
        if not includeCanon and not canonOnly:
            filters["skip_sessions"] = {indexer.CANON_SESSION}
        return self.ctx.hits(query, limit, **filters)

    def canon(self, info, topic=None, contains=None, limit=20):
        db = self.ctx.db
        if topic:
            rows = db.execute("SELECT topic_key FROM canon WHERE topic_key=?", (indexer.canon_key(topic),))
        elif contains:
            pattern = f"%{contains}%"
            rows = db.execute(
                """SELECT topic_key FROM canon WHERE topic LIKE ? OR fact LIKE ?
                   ORDER BY recorded_at DESC LIMIT ?""",
                (pattern, pattern, _clamp(limit, 1, MAX_LIMIT)),
            )
        else:
            rows = db.execute(
                "SELECT topic_key FROM canon ORDER BY recorded_at DESC LIMIT ?",
                (_clamp(limit, 1, MAX_LIMIT),),
            )
        return [self.ctx.canon_fact(row[0]) for row in rows.fetchall()]

    def sessions(self, info, limit=10):
        rows = self.ctx.db.execute(
            """SELECT session_id, MAX(id) AS last FROM messages WHERE session_id IS NOT ?
               GROUP BY session_id ORDER BY last DESC LIMIT ?""",
            (indexer.CANON_SESSION, _clamp(limit, 1, MAX_LIMIT)),
        ).fetchall()
        return [Session(self.ctx, row[0]) for row in rows]

    def session(self, info, id):
        exists = self.ctx.db.execute("SELECT 1 FROM messages WHERE session_id=? LIMIT 1", (id,)).fetchone()
        return Session(self.ctx, id) if exists else None

    def turn(self, info, uuid):
        return self.ctx.turn(uuid)

    def remember(self, info, fact, topic=None, links=None):
        pairs = [(link["topic"], link.get("relation")) for link in links or ()]
        msg_uuid, replaced = indexer.remember_fact(self.ctx.db, fact, topic, pairs, self.ctx.embed)
        key = msg_uuid.split(":", 1)[1]
        self.ctx._canon.pop(key, None)
        return {"fact": self.ctx.canon_fact(key), "replaced": replaced}

    def link(self, info, source, target, relation="related"):
        src, dst = indexer.canon_key(source), indexer.canon_key(target)
        if not src or not dst or src == dst:
            return False
        indexer.link_canon(self.ctx.db, src, dst, relation, target.strip())
        self.ctx.db.commit()
        return True

    def forget(self, info, topic):
        self.ctx._canon.pop(indexer.canon_key(topic), None)
        return indexer.forget_fact(self.ctx.db, topic)


def _depth(node, fragments, seen=()):
    selection_set = getattr(node, "selection_set", None)
    if not selection_set:
        return 0
    deepest = 0
    for selection in selection_set.selections:
        kind = selection.kind
        if kind == "fragment_spread":
            name = selection.name.value
            if name in seen or name not in fragments:
                continue
            deepest = max(deepest, _depth(fragments[name], fragments, seen + (name,)))
        elif kind == "inline_fragment":
            deepest = max(deepest, _depth(selection, fragments, seen))
        else:
            deepest = max(deepest, 1 + _depth(selection, fragments, seen))
    return deepest


def execute(db, source, variables=None, embed=None):
    """Run one GraphQL document. Returns a JSON-able {"data", "errors"} dict."""
    from graphql import graphql_sync, parse, GraphQLError

    try:
        document = parse(source)
    except GraphQLError as exc:
        return {"data": None, "errors": [{"message": exc.message}]}
    fragments = {
        definition.name.value: definition
        for definition in document.definitions
        if definition.kind == "fragment_definition"
    }
    for definition in document.definitions:
        if definition.kind == "operation_definition" and _depth(definition, fragments) > MAX_DEPTH:
            return {"data": None, "errors": [{"message": f"Query nests deeper than {MAX_DEPTH} levels."}]}

    ctx = Context(db, embed or indexer.fastembed_embed)
    result = graphql_sync(
        get_schema(), source, root_value=Root(ctx), context_value=ctx,
        variable_values=variables or None,
    )
    if ctx.shown:
        indexer.record_recalls(db, sorted(ctx.shown))
    payload = {"data": result.data}
    if result.errors:
        payload["errors"] = [{"message": err.message, "path": err.path} for err in result.errors]
    return payload


def render(payload):
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if len(text) > MAX_RESULT_CHARS:
        return json.dumps({
            "data": None,
            "errors": [{
                "message": (
                    f"Result is {len(text)} characters, over the {MAX_RESULT_CHARS} limit. "
                    "Ask for fewer fields, lower limits, or smaller maxChars."
                )
            }],
        })
    return text


def main(argv):
    if "--schema" in argv:
        print(SDL.strip())
        return 0
    args = [arg for arg in argv if not arg.startswith("--")]
    if not args:
        print(__doc__.strip())
        return 2
    variables = json.loads(args[1]) if len(args) > 1 else None
    db = indexer.ensure_db()
    try:
        print(render(execute(db, args[0], variables)))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
