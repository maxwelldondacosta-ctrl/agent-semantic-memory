#!/usr/bin/env python3
"""
MCP stdio server that lets the model query project memory itself.

The UserPromptSubmit hook only guesses what is relevant from the user's
words. The model is the one who notices, mid-answer, that it needs a name,
a stat or a decision from three sessions ago. These tools let it look:

  recall(query)            hybrid search over every indexed turn + canon
  remember(fact, topic)    record an authoritative fact; same topic replaces
  forget(topic)            drop a canon entry that is no longer true

A recall that finds nothing says so plainly. That answer is the point: it
gives the model a cheap, honest alternative to inventing a detail.

No MCP SDK dependency — newline-delimited JSON-RPC 2.0 on stdin/stdout.
Register it with:
  claude mcp add semantic-memory --scope project -- \
      python3 "$PROJECT_ROOT/.claude/hooks/semantic-memory/memory_server.py"
"""

import json
import os
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import indexer  # noqa: E402

SERVER_NAME = "semantic-memory"
SERVER_VERSION = "0.3.0"
DEFAULT_PROTOCOL = "2025-06-18"

# An explicit question deserves a slightly wider net than a passive
# injection, but the background for bge-small sits near 0.55–0.59.
RECALL_THRESHOLD = 0.60
RECALL_MARGIN = 0.12
RECALL_TOP_K = 6
RECALL_MAX_CHARS = 5000

TOOLS = [
    {
        "name": "recall",
        "description": (
            "Search this project's long-term memory: every earlier conversation turn "
            "(including parts of this session that were compacted away) and recorded "
            "canon facts. Call this BEFORE stating any project detail you cannot see "
            "in the current context — names, lore, stats, rules, file or function "
            "names, earlier decisions, things the user said before. Use it whenever "
            "you are less than certain, or about to write 'I think', 'probably' or "
            "'as we decided'. If it returns no record, tell the user you have no record "
            "instead of guessing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you need to know, phrased as a question or the specific names involved.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember",
        "description": (
            "Record an authoritative project fact (canon) so it survives compaction and "
            "new sessions: a character's name or backstory, a game rule, a balance "
            "number, a design decision. Use a short stable topic; remembering the same "
            "topic again replaces the old fact. Only record what the user stated or "
            "approved."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The fact, stated fully on its own."},
                "topic": {"type": "string", "description": "Stable key, e.g. 'boss: Vargan stats'."},
            },
            "required": ["fact"],
        },
    },
    {
        "name": "forget",
        "description": "Remove a canon fact by topic when the user says it is no longer true.",
        "inputSchema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
]


class MemoryServer:
    def __init__(self, db_path=None):
        self.db_path = db_path

    def _db(self):
        db = indexer.ensure_db(self.db_path)
        for path in indexer.recent_transcripts():
            try:
                indexer.refresh_transcript(db, path)
            except Exception:
                traceback.print_exc(file=sys.stderr)
        return db

    def recall(self, query, limit=None):
        query = (query or "").strip()
        if not query:
            return "recall needs a query.", True
        db = self._db()
        try:
            vec = indexer.fastembed_embed([indexer.QUERY_PREFIX + query])[0]
            text, shown = indexer.retrieve_with_ids(
                db,
                vec,
                top_k=max(1, min(int(limit or RECALL_TOP_K), 10)),
                threshold=RECALL_THRESHOLD,
                relative_margin=RECALL_MARGIN,
                max_chars=RECALL_MAX_CHARS,
                query_text=query,
                header=(
                    f"{indexer.MEMORY_MARK} · recall] Results for: {query}\n"
                    "CANON entries are authoritative. Other entries are raw past turns; "
                    "later turns can supersede earlier ones, so check timestamps."
                ),
            )
            if shown:
                indexer.record_recalls(db, shown)
        finally:
            db.close()
        if not text:
            return (
                f"{indexer.MEMORY_MARK} · recall] No record found for: {query}\n"
                "Nothing in project memory matches. Do not fill the gap with a guess: "
                "tell the user you have no record of this, and ask or propose a value "
                "explicitly as new."
            ), False
        return text, False

    def remember(self, fact, topic=None):
        db = self._db()
        try:
            msg_uuid, replaced = indexer.remember_fact(db, fact, topic)
        finally:
            db.close()
        note = f"{indexer.MEMORY_MARK} · remember] Recorded as canon ({msg_uuid})."
        if replaced:
            note += f"\nReplaced previous entry: {replaced}"
        return note, False

    def forget(self, topic):
        db = self._db()
        try:
            removed = indexer.forget_fact(db, topic)
        finally:
            db.close()
        if removed:
            return f"{indexer.MEMORY_MARK} · forget] Removed canon topic: {topic}", False
        return f"{indexer.MEMORY_MARK} · forget] No canon entry for topic: {topic}", False

    def call(self, name, arguments):
        arguments = arguments or {}
        if name == "recall":
            return self.recall(arguments.get("query"), arguments.get("limit"))
        if name == "remember":
            return self.remember(arguments.get("fact"), arguments.get("topic"))
        if name == "forget":
            return self.forget(arguments.get("topic"))
        return f"Unknown tool: {name}", True

    def handle(self, message):
        """Return the JSON-RPC response for one message, or None for notifications."""
        method = message.get("method")
        msg_id = message.get("id")
        if msg_id is None:
            return None
        if method == "initialize":
            params = message.get("params") or {}
            result = {
                "protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = message.get("params") or {}
            try:
                text, is_error = self.call(params.get("name"), params.get("arguments"))
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                text, is_error = f"Memory error: {exc}", True
            result = {"content": [{"type": "text", "text": text}], "isError": is_error}
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def serve(stdin=sys.stdin, stdout=sys.stdout):
    server = MemoryServer()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
        else:
            response = server.handle(message)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
