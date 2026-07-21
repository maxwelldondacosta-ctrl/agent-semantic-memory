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
   (cheap — only the current session file gets rescanned, not full project
   history).
2. Embeds the new prompt and does cosine-similarity retrieval — via
   [fastembed](https://github.com/qdrant/fastembed)
   (`BAAI/bge-small-en-v1.5`, fully local ONNX, no server, no API cost) —
   against the *entire* project's indexed conversation history.
3. Injects the top few matches above a similarity threshold, with a little
   surrounding context, capped at a few thousand characters, as
   `additionalContext`.
4. Once the live transcript crosses a size threshold, nudges toward
   `/clear` — since anything relevant is now retrievable on demand, you
   don't need to keep dragging the raw history forward.

No LLM summarization step anywhere in the pipeline. Store everything raw,
retrieve a small precise slice when it's actually relevant. That sidesteps
drift because nothing is ever paraphrased or dropped — it's lossless
storage + cheap, targeted retrieval, instead of lossy storage.

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

Should print valid `{"hookSpecificOutput": {...}}` JSON and exit 0.

## How it works

- **Storage**: `semantic_memory.db` — SQLite, one `messages` table (uuid,
  session_id, role, timestamp, text, vector blob, indexed_at).
  Project-local, not shared across projects.
- **Transcript source**: Claude Code's own session transcripts at
  `~/.claude/projects/<encoded-project-path>/*.jsonl`. The extractor only
  reads `type: "user"`/`"assistant"` entries and their text content,
  skipping tool-call plumbing, to stay resilient to schema drift elsewhere
  in the transcript format.
- **Retrieval**: cosine similarity, top 4 matches above a 0.45 similarity
  threshold, ±2 messages of surrounding context per match, capped at ~3000
  injected characters.
- **Nudge threshold**: 2MB transcript file size, adjustable via
  `NUDGE_THRESHOLD_BYTES` in `retrieval_hook.py`.
- **Contract**: Claude Code's `UserPromptSubmit` hook — stdin JSON has
  `user_prompt`, `transcript_path`, `session_id`, `cwd`, etc.; stdout must
  be `{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
  "additionalContext": "..."}}` on exit 0 to inject context. The hook fails
  silent (exit 0, no output) on any error — it must never break a real
  prompt.

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
  some user turns. Only `text` blocks get indexed.
- If `python3` on `PATH` doesn't have `fastembed` installed (e.g. system
  Python vs. a separate install), the hook command needs an explicit
  interpreter path or it will silently fail every call — test standalone
  before trusting it live.

## License

MIT
