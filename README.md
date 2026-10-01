# agent-semantic-memory

Local, no-network semantic memory for [Claude Code](https://claude.com/claude-code) sessions.

## The problem

Claude Code has no configurable auto-compaction threshold — once a session
runs long, the entire raw transcript gets resent to the model on every
single turn. That causes two related problems:

1. **Token cost/latency blows up.** A long session can balloon to megabytes
   of transcript, and you pay to re-read all of it just to ask one more
   question.
2. **Memory drift.** When context finally does get compacted/summarized,
   detail gets lost or garbled — the model starts "forgetting" specifics
   from earlier in the session (a decision you made, a constraint you
   mentioned once and never repeated).

The usual fix is "summarize periodically," but summarization is lossy by
construction — you're betting that whatever gets thrown away won't matter
later. It usually does.

## The approach

A `UserPromptSubmit` hook that fires on every message and:

1. Incrementally embeds and indexes any new turns from the current
   session's transcript into a local SQLite store, deduped by message UUID
   plus chunk index (only the current session file gets rescanned, not
   full project history).
2. Figures out which turns the model can **already see**, and refuses to
   spend the injection budget on them.
3. Embeds the new prompt and does cosine-similarity retrieval — via
   [fastembed](https://github.com/qdrant/fastembed)
   (`BAAI/bge-small-en-v1.5`, fully local ONNX, no server, no API cost) —
   against everything else: earlier sessions, and the part of *this*
   session that compaction already replaced with a summary.
4. Injects a few diverse matches above a similarity threshold, with a
   little surrounding context, capped at a few thousand characters, as
   `additionalContext`.
5. Once the transcript file crosses a size threshold, nudges toward
   `/clear` — relevant history stays retrievable, so the raw transcript
   does not have to keep riding along.

No LLM summarization step anywhere in the pipeline. Store the raw turns,
retrieve a small precise slice of what the live prompt no longer contains.

### What "already seen" means

Claude Code appends a `compact_boundary` record when it summarizes a
session. Messages after the last boundary (the hot zone) are still in the
prompt; messages before it survive in the JSONL but the model only has the
lossy summary. Retrieval excludes the hot zone and searches the forgotten
prefix plus every other session.

Until a boundary exists, the whole current file is the live prompt, so it
is excluded too — otherwise the top hits are just the conversation you are
already having, and older sessions never fit in the budget. If the file is
huge and the marker is missing, only the recent tail is treated as live.

The compact summary itself is not indexed. It is already in the hot zone,
and storing the paraphrase would compete with the raw turns it replaced.

Earlier injections count as seen too. Hook context stays in the prompt
until the next compaction, so every excerpt the hook shows is recorded
per session and compaction epoch and skipped until a new
`compact_boundary` appears. Without that, a topic you keep working on
would re-inject the same excerpt on every turn.

### Meaning and exact names

Embeddings are good at "how do jobs avoid running twice" and bad at
`export_rows`, `E1042` or `src/net/http.py`. Those exact identifiers are
often what the question is about. Every chunk is also in a SQLite FTS5
keyword index, and the two rankings are merged with reciprocal rank
fusion. A keyword-only hit needs a moderate semantic score, unless the
matching term looks like an identifier, so one shared common word is not
enough.

The same design generalizes to any agent framework with a pre-prompt /
pre-LLM-call hook — the only framework-specific parts are the input/output
contract (what shape the hook receives on stdin, what shape it must return)
and where that framework stores its own transcripts.

## Letting the model look things up itself

Passive injection only knows what the user typed. The model is the one who
finds out, halfway through an answer, that it needs a boss's HP or a rule
decided three sessions ago, and models rarely admit "I don't remember".
They fill the gap. Three pieces cover that:

1. **A `recall` tool** (`memory_server.py`, an MCP server). The model can
   search all of memory whenever it is unsure. When nothing matches, the
   tool says "No record found" and tells the model not to guess. That gives
   it an honest answer that is cheaper than inventing one. Once per session
   and compaction, the prompt hook reminds the model that the tool exists
   and that memory exists beyond what it can see.
2. **Canon** (`remember` / `forget`). Settled facts such as lore, names,
   stats and rules are stored as one authoritative entry per topic.
   Remembering a topic again replaces the old fact, so a rebalance can't
   leave two conflicting HP values side by side. Results label canon as
   `CANON`.
3. **A check after every reply** (`verify_hook.py`, a `Stop` hook). This
   covers the cases where the model won't admit it is unsure. It reads the
   finished reply and picks out sentences that lean on the past ("as we
   decided", "earlier", "we named") or hedge ("I think", "if I recall",
   "I don't remember"). It then looks those sentences up:
   - If memory holds records the model has not seen in this context, it
     hands them back once and asks the model to correct itself to the user
     if anything conflicts.
   - If memory has no record at all of something the reply asserts, it
     asks the model to present it as a suggestion, not established fact.
     Sentences that propose things ("should", "let's", "maybe") are left
     alone.

   It never fires twice in a row (`stop_hook_active`). Records that the
   model just recalled, or that the hook already showed, are not repeated.

### Structured queries (GraphQL)

Game canon is a graph: a boss lives somewhere, drops something, belongs to a
faction. Each fact also has a history, meaning the conversation where it was
decided. `query_memory` runs GraphQL (`memory_graphql.py`) over the same
store, so the model can follow those links in one call:

```graphql
{
  canon(topic: "Vargan") {
    fact
    links { relation direction target { topic recorded fact } }
    discussedIn(limit: 2) {
      excerpt
      turn { timestamp before(count: 1) { text } }
    }
  }
}
```

`remember` takes links (`"lives in: Frost Caverns"`), and GraphQL's
`remember(..., links: [{topic, relation}])` and `link(source, target,
relation)` mutations do the same. Links may point at topics that are not
recorded yet; those show up with `recorded: false`, which marks a hole in
the lore. Other entry points are `search`, `sessions`, `session(id)` and
`turn(uuid)`. A turn can walk `before` / `after` within its session. Run
`python3 memory_graphql.py --schema` for the full schema.

GraphQL is for exploration. For "do we have anything about X?", `recall`
is simpler and leaves the model less room to get the query wrong. The
server caps nesting depth, list sizes and result size.

## Install

```bash
pip3 install --user fastembed graphql-core   # graphql-core is only needed for query_memory

mkdir -p "$PROJECT_ROOT/.claude/hooks/semantic-memory"
cp indexer.py retrieval_hook.py verify_hook.py memory_server.py memory_graphql.py \
   "$PROJECT_ROOT/.claude/hooks/semantic-memory/"
chmod +x "$PROJECT_ROOT/.claude/hooks/semantic-memory/"*.py

# Backfill from existing session history for this project
python3 "$PROJECT_ROOT/.claude/hooks/semantic-memory/indexer.py"

# Give the model its recall / remember / query_memory tools
cd "$PROJECT_ROOT" && claude mcp add semantic-memory --scope project -- \
  python3 "$PROJECT_ROOT/.claude/hooks/semantic-memory/memory_server.py"
```

Then register both hooks in `$PROJECT_ROOT/.claude/settings.local.json`
(merge into any existing config, don't overwrite it):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/semantic-memory/retrieval_hook.py\""
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/semantic-memory/verify_hook.py\""
          }
        ]
      }
    ]
  }
}
```

To let the tools run without a permission prompt each time, add
`"permissions": {"allow": ["mcp__semantic-memory__recall",
"mcp__semantic-memory__query_memory"]}`. Leave `remember` and `forget`
behind a prompt if you want to approve every canon change.

Verify the pieces standalone before trusting them live:

```bash
H="$PROJECT_ROOT/.claude/hooks/semantic-memory"
T="<a real transcript path from ~/.claude/projects/<encoded-project-path>/>"
echo "{\"prompt\": \"something from this project history\", \"transcript_path\": \"$T\"}" | python3 "$H/retrieval_hook.py"
echo "{\"last_assistant_message\": \"As we decided earlier, ...\", \"transcript_path\": \"$T\"}" | python3 "$H/verify_hook.py"
python3 "$H/memory_graphql.py" '{ sessions(limit: 3) { id turnCount } }'
```

Each hook exits 0 and prints `{"hookSpecificOutput": {...}}` when it has something to
add, or nothing otherwise.

## How it works

- **Storage**: `semantic_memory.db` — SQLite (WAL). `messages` holds
  uuid, chunk index, session id, role, timestamp, text, float32 vector and
  indexed_at. `messages_fts` is the trigger-synced keyword index, and
  `injections` records what was shown per session and compaction epoch.
  `canon` and `canon_links` hold remembered facts and the links between
  them, and `recalls` notes what the model looked up recently, so the
  Stop hook doesn't repeat it. Project-local, not shared across projects. An older database is migrated
  and backfilled on open.
- **Transcript source**: Claude Code's own session transcripts at
  `~/.claude/projects/<encoded-project-path>/*.jsonl`. The extractor reads
  `type: "user"` / `"assistant"` text, plus short tool results (errors and
  statuses). Long tool output is skipped — it is usually a file dump.
  Compact summaries and `isMeta` records are skipped.
- **Chunking**: turns longer than 1500 characters are embedded in
  overlapping windows so the tail of a long message is still findable.
  The stored text is the raw slice, not a paraphrase.
- **Retrieval**: normalized cosine similarity over the float32 matrix
  (numpy when it is installed, which it is via fastembed), fused with BM25
  from the FTS5 index. A semantic hit must score at least 0.62 and come
  within 0.10 of the query's best match. A keyword hit must score at least
  0.55, unless the term looks like an identifier. Hits are diversified so
  near-copies of one turn do not fill the budget, shown with ±2 neighboring
  messages from the *same* session, and capped at ~3000 characters. A
  snippet is centered on the matched keyword when that keyword sits deep
  in a long chunk. Queries get the BGE retrieval prefix. A short or
  anaphoric prompt ("do that again") is embedded together with the live
  tail so the query has a topic, but that tail is not injected.
- **Thresholds**: bge-small rarely scores unrelated text below 0.4, and
  in a sample project unrelated turns reached 0.59 while relevant ones
  scored 0.65–0.80. The old 0.45 cutoff injected noise on nearly every
  prompt. If you change the model, re-measure these values
  (`SCORE_THRESHOLD`, `RELATIVE_MARGIN`, `KEYWORD_FLOOR` in
  `retrieval_hook.py`).
- **Latency**: each hook takes about 0.65 s on a small index, almost all of
  it loading the ONNX model in a fresh process. The Stop hook only loads it
  when the reply contains a past reference or a hedge. The MCP server loads
  the model once; after that, recall and GraphQL calls took 0.01–0.09 s.
- **Nudge threshold**: 2MB transcript file size (`LARGE_TRANSCRIPT_BYTES`
  in `indexer.py`). The same threshold is the fallback for "no compact
  boundary, but this file is too big to treat as fully live."
- **Contract**: Claude Code's `UserPromptSubmit` hook — stdin JSON has
  `prompt`, `transcript_path`, `session_id`, `cwd`, etc. (earlier versions of
  this repo read `user_prompt`, which Claude Code never sends, so live
  retrieval silently did nothing). The `Stop` hook reads
  `last_assistant_message` and `stop_hook_active`. stdout must
  be `{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
  "additionalContext": "..."}}` on exit 0 to inject context. The hook exits
  0 with no output on any error and appends the traceback to
  `semantic_memory.log` — it must never break a real prompt.

## Gotchas

- Don't assume this hook contract carries over to other agent frameworks —
  each one has its own shape (some expect a `systemMessage` field via a
  YAML-configured pre-call hook, others expect the `hookSpecificOutput`
  shape used here). Check your framework's actual hook docs before
  porting.
- `transcript_path` in the hook payload is the *current session's own
  file* — indexing just that file per call (not a full project rescan) is
  what keeps this fast enough for a hook that fires on every message.
- `message.content` is a plain string for simple user turns but a list of
  content blocks (mixed text/tool_use/tool_result) for assistant turns and
  some user turns. Text blocks are indexed. Tool results are indexed only
  when they are short (≤600 characters); longer ones are dropped.
- After changing chunk size, run `indexer.py --reindex`. Ordinary new turns
  are picked up incrementally, and a message is re-embedded when its stored
  chunk count no longer matches.
- Skipping repeat injections assumes Claude Code keeps earlier
  `additionalContext` in the conversation until compaction. If you start a
  new session with `/clear`, its session id changes, so recall starts fresh.
  To reset by hand, `DELETE FROM injections`.
- Everything this project writes into the conversation (hook context,
  tool results) starts with `[Project semantic memory`, and the indexer
  skips it. Memory never indexes its own output.
- The Stop hook's cue lists are English regexes in `verify_hook.py`. If
  your sessions have phrasings it misses, add them there.
- Injected excerpts are framed as reference. They are past transcript text,
  not new instructions, and the hook says so in the context header.
- If `python3` on `PATH` doesn't have `fastembed` installed (e.g. system
  Python vs. a separate install), the hook command needs an explicit
  interpreter path or it will silently fail every call — test standalone
  before trusting it live.

## Tests

```bash
python3 -m unittest test_semantic_memory.py
```

The tests stub the embedder, so they do not download the model. The
GraphQL tests are skipped when `graphql-core` is not installed.

## License

MIT
