# agent-semantic-memory

Local, no-server semantic memory for [Claude Code](https://claude.com/claude-code) sessions.

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

The same design generalizes to any agent framework with a pre-prompt /
pre-LLM-call hook — the only framework-specific parts are the input/output
contract (what shape the hook receives on stdin, what shape it must return)
and where that framework stores its own transcripts.

## Install

```bash
pip3 install --user fastembed

mkdir -p "$PROJECT_ROOT/.claude/hooks/semantic-memory"
cp indexer.py retrieval_hook.py "$PROJECT_ROOT/.claude/hooks/semantic-memory/"
chmod +x "$PROJECT_ROOT/.claude/hooks/semantic-memory/"*.py

# Backfill from existing session history for this project
python3 "$PROJECT_ROOT/.claude/hooks/semantic-memory/indexer.py"
```

Then register the hook in `$PROJECT_ROOT/.claude/settings.local.json`
(merge into any existing config, don't overwrite it):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/semantic-memory/retrieval_hook.py\""
          }
        ]
      }
    ]
  }
}
```

Verify it standalone before trusting it live:

```bash
echo '{"user_prompt": "test query about something from this project history", "transcript_path": "<a real transcript path from ~/.claude/projects/<encoded-project-path>/>"}' \
  | python3 "$PROJECT_ROOT/.claude/hooks/semantic-memory/retrieval_hook.py"
```

Should exit 0. When the index has a match outside the live transcript, stdout is
`{"hookSpecificOutput": {...}}`. No match prints nothing.

## How it works

- **Storage**: `semantic_memory.db` — SQLite (WAL), one `messages` table
  (uuid, chunk index, session id, role, timestamp, text, float32 vector,
  indexed_at). Project-local, not shared across projects. An older JSON-vector
  database is migrated on open.
- **Transcript source**: Claude Code's own session transcripts at
  `~/.claude/projects/<encoded-project-path>/*.jsonl`. The extractor reads
  `type: "user"` / `"assistant"` text, plus short tool results (errors and
  statuses). Long tool output is skipped — it is usually a file dump.
  Compact summaries and `isMeta` records are skipped.
- **Chunking**: turns longer than 1500 characters are embedded in
  overlapping windows so the tail of a long message is still findable.
  The stored text is the raw slice, not a paraphrase.
- **Retrieval**: normalized cosine similarity over the float32 matrix
  (numpy when it is installed, which it is via fastembed). Top matches
  above 0.45, diversified so four near-copies of the same turn do not fill
  the budget, ±2 neighboring messages from the *same* session, capped at
  ~3000 characters. Queries get the BGE retrieval prefix. A short or
  anaphoric prompt ("do that again") is embedded together with the live
  tail so the query has a topic, but that tail is not injected.
- **Nudge threshold**: 2MB transcript file size (`LARGE_TRANSCRIPT_BYTES`
  in `indexer.py`). The same threshold is the fallback for "no compact
  boundary, but this file is too big to treat as fully live."
- **Contract**: Claude Code's `UserPromptSubmit` hook — stdin JSON has
  `user_prompt`, `transcript_path`, `session_id`, `cwd`, etc.; stdout must
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

The tests stub the embedder, so they do not download the model.

## License

MIT
