#!/usr/bin/env python3
"""Unit tests for indexing and hot-zone retrieval. No model download."""

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import indexer
import retrieval_hook


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _user(uuid, text, session="sess", timestamp="2026-01-01T00:00:00Z"):
    return {
        "type": "user",
        "uuid": uuid,
        "sessionId": session,
        "timestamp": timestamp,
        "message": {"role": "user", "content": text},
    }


def _assistant(uuid, text, session="sess", timestamp="2026-01-01T00:01:00Z"):
    return {
        "type": "assistant",
        "uuid": uuid,
        "sessionId": session,
        "timestamp": timestamp,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


class ChunkAndExtractTests(unittest.TestCase):
    def test_chunk_text_overlaps_and_caps(self):
        chunks = indexer.chunk_text("a" * 4000, size=1500, overlap=200, limit=8)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks), 8)
        self.assertEqual(len(chunks[0]), 1500)
        self.assertTrue(chunks[0][-200:] == chunks[1][:200] or len(chunks) == 1)

    def test_skips_compact_summary_and_keeps_short_tool_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sess.jsonl")
            _write_jsonl(path, [
                _user("u1", "keep the rust parser decision"),
                {
                    "type": "user",
                    "uuid": "sum",
                    "isCompactSummary": True,
                    "message": {"role": "user", "content": "lossy summary of the rust parser"},
                },
                {
                    "type": "user",
                    "uuid": "meta",
                    "isMeta": True,
                    "message": {"role": "user", "content": "meta noise"},
                },
                {
                    "type": "user",
                    "uuid": "tool",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "content": "error: parser rejected token"},
                            {"type": "text", "text": "that failed"},
                        ],
                    },
                },
                {
                    "type": "user",
                    "uuid": "dump",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "content": "x" * 5000}],
                    },
                },
            ])
            messages = list(indexer.extract_messages(path))
        uuids = [m[0] for m in messages]
        self.assertEqual(uuids, ["u1", "tool"])
        self.assertIn("error: parser rejected token", messages[1][4])
        self.assertIn("that failed", messages[1][4])

    def test_pack_roundtrip_is_normalized(self):
        blob = indexer.pack_vector([3.0, 4.0])
        unpacked = indexer.unpack_vector(blob)
        self.assertAlmostEqual(sum(x * x for x in unpacked), 1.0, places=5)
        self.assertAlmostEqual(indexer.cosine_similarity(unpacked, [3.0, 4.0]), 1.0, places=5)
        self.assertEqual(indexer.unpack_vector(json.dumps([3.0, 4.0]).encode()), [3.0, 4.0])


class SchemaMigrationTests(unittest.TestCase):
    def test_json_vectors_migrate_to_float32_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "semantic_memory.db")
            db = sqlite3.connect(path)
            db.execute("""
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    msg_uuid TEXT UNIQUE,
                    session_id TEXT,
                    role TEXT,
                    timestamp TEXT,
                    text TEXT,
                    vector BLOB,
                    indexed_at REAL
                )
            """)
            db.execute(
                "INSERT INTO messages (msg_uuid, session_id, role, timestamp, text, vector, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("old", "sess", "user", "t", "hello", json.dumps([1.0, 0.0]).encode(), 1.0),
            )
            db.commit()
            db.close()

            indexer.INDEX_DB = path
            db = indexer.ensure_db()
            row = db.execute(
                "SELECT msg_uuid, chunk_index, text, vector FROM messages"
            ).fetchone()
            db.close()
            self.assertEqual(row[0], "old")
            self.assertEqual(row[1], 0)
            self.assertEqual(row[2], "hello")
            self.assertFalse(row[3].startswith(b"["))
            self.assertAlmostEqual(indexer.unpack_vector(row[3])[0], 1.0, places=5)


class LiveContextTests(unittest.TestCase):
    def test_boundary_excludes_only_the_hot_zone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sess.jsonl")
            _write_jsonl(path, [
                _user("old", "decide to use a rust parser"),
                {"type": "system", "subtype": "compact_boundary", "uuid": "bound"},
                {
                    "type": "user",
                    "uuid": "sum",
                    "isCompactSummary": True,
                    "message": {"role": "user", "content": "summary"},
                },
                _user("live", "now tweak the css"),
            ])
            excluded = indexer.live_context_uuids(path)
        self.assertEqual(excluded, {"sum", "live"})

    def test_no_boundary_excludes_the_whole_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sess.jsonl")
            _write_jsonl(path, [_user("a", "one"), _user("b", "two")])
            excluded = indexer.live_context_uuids(path, large_bytes=10**9)
        self.assertEqual(excluded, {"a", "b"})

    def test_huge_file_without_boundary_keeps_only_the_tail_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sess.jsonl")
            records = [_user(f"m{i}", f"turn {i}") for i in range(5)]
            _write_jsonl(path, records)
            excluded = indexer.live_context_uuids(path, large_bytes=1, tail=2)
        self.assertEqual(excluded, {"m3", "m4"})


class RetrieveTests(unittest.TestCase):
    def _db(self, tmp):
        path = os.path.join(tmp, "semantic_memory.db")
        indexer.INDEX_DB = path
        return indexer.ensure_db()

    def _insert(self, db, uuid, text, vec, session="other", role="user", timestamp="2026-02-01T00:00:00Z"):
        db.execute(
            """INSERT INTO messages
               (msg_uuid, chunk_index, session_id, role, timestamp, text, vector, indexed_at)
               VALUES (?, 0, ?, ?, ?, ?, ?, 1)""",
            (uuid, session, role, timestamp, text, indexer.pack_vector(vec)),
        )

    def test_skips_hot_zone_and_returns_archived_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            self._insert(db, "live", "live window rust parser", [1, 0], session="current")
            self._insert(db, "old", "ancient decision about rust parser", [0.95, 0.05], session="current")
            self._insert(db, "css", "stylesheet color tokens", [0, 1], session="current")
            db.commit()
            text = indexer.retrieve(db, [1, 0], exclude_uuids={"live"}, top_k=2, threshold=0.5)
            db.close()
        self.assertIn("ancient decision about rust parser", text)
        self.assertNotIn("live window rust parser", text)

    def test_mmr_keeps_a_diverse_second_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            self._insert(db, "a", "parser choice alpha", [1, 0], session="s1")
            self._insert(db, "b", "parser choice beta", [0.99, 0.01], session="s2")
            self._insert(db, "c", "parser choice gamma", [0.98, 0.02], session="s3")
            # Related enough to clear the threshold, far enough to be diverse.
            self._insert(db, "d", "also ship a python binding", [0.55, 0.84], session="s4")
            db.commit()
            text = indexer.retrieve(db, [1, 0], exclude_uuids=set(), top_k=2, threshold=0.4)
            db.close()
        self.assertIn("parser choice alpha", text)
        self.assertIn("python binding", text)
        self.assertNotIn("parser choice beta", text)

    def test_neighbor_context_stays_inside_the_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            self._insert(db, "before", "constraint: no new dependency", [0.2, 0.9], session="s1",
                         timestamp="2026-02-01T00:00:00Z")
            self._insert(db, "hit", "use the rust parser", [1, 0], session="s1",
                         timestamp="2026-02-01T00:01:00Z")
            self._insert(db, "other", "unrelated billing note", [0.2, 0.1], session="s2",
                         timestamp="2026-02-01T00:00:30Z")
            db.commit()
            # Force ids so the other session is between them if someone sorts globally.
            text = indexer.retrieve(db, [1, 0], top_k=1, threshold=0.5, context_window=1)
            db.close()
        self.assertIn("use the rust parser", text)
        self.assertIn("no new dependency", text)
        self.assertNotIn("billing note", text)

    def test_char_budget_drops_a_block_instead_of_cutting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            self._insert(db, "a", "A" * 40, [1, 0], session="s1")
            self._insert(db, "b", "B" * 40, [0.9, 0.1], session="s2")
            db.commit()
            text = indexer.retrieve(
                db, [1, 0], top_k=2, threshold=0.2, context_window=0,
                max_chars=420, snippet_chars=40,
            )
            db.close()
        self.assertIn("A" * 40, text)
        self.assertNotIn("B" * 40, text)
        self.assertFalse(text.endswith("…"))


class KeywordTests(unittest.TestCase):
    def _db(self, tmp):
        indexer.INDEX_DB = os.path.join(tmp, "semantic_memory.db")
        return indexer.ensure_db()

    def _insert(self, db, uuid, text, vec, session="s"):
        db.execute(
            """INSERT INTO messages
               (msg_uuid, chunk_index, session_id, role, timestamp, text, vector, indexed_at)
               VALUES (?, 0, ?, 'user', 't', ?, ?, 1)""",
            (uuid, session, text, indexer.pack_vector(vec)),
        )

    def test_keyword_terms_keep_identifiers_and_drop_stopwords(self):
        terms = indexer.keyword_terms("why does parse_header() fail in src/net/http.py with E1042?")
        self.assertIn("parse_header", terms)
        self.assertIn("src/net/http.py", terms)
        self.assertIn("E1042", terms)
        self.assertNotIn("why", terms)
        self.assertNotIn("does", terms)

    def test_identifier_match_is_recalled_below_the_vector_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            self._insert(db, "hit", "parse_header() drops the trailing CRLF, fixed with rstrip", [0.1, 1.0])
            self._insert(db, "noise", "unrelated discussion about colors", [0.0, 1.0], session="other")
            db.commit()
            vector_only = indexer.retrieve(db, [1.0, 0.0], threshold=0.45)
            hybrid = indexer.retrieve(
                db, [1.0, 0.0], threshold=0.45,
                query_text="what was wrong with parse_header again",
            )
            db.close()
        self.assertIsNone(vector_only)
        self.assertIn("parse_header() drops the trailing CRLF", hybrid)
        self.assertIn("keyword parse_header", hybrid)
        self.assertNotIn("colors", hybrid)

    def test_common_word_alone_does_not_admit_an_unrelated_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            self._insert(db, "noise", "the release notes mention a parser", [0.0, 1.0])
            db.commit()
            text = indexer.retrieve(db, [1.0, 0.0], threshold=0.45, query_text="write release notes")
            db.close()
        self.assertIsNone(text)

    def test_fts_follows_upserts_and_deletes(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            with mock.patch.object(indexer, "fastembed_embed", lambda texts: [[1.0, 0.0] for _ in texts]):
                indexer.index_messages(db, [("u", 0, "s", "user", "t", "alpha_token here")])
                self.assertEqual(len(indexer.keyword_search(db, ["alpha_token"])), 1)
                indexer.index_messages(db, [("u", 0, "s", "user", "t", "beta_token now")])
            self.assertEqual(indexer.keyword_search(db, ["alpha_token"]), [])
            self.assertEqual(len(indexer.keyword_search(db, ["beta_token"])), 1)
            db.execute("DELETE FROM messages")
            db.commit()
            self.assertEqual(indexer.keyword_search(db, ["beta_token"]), [])
            db.close()

    def test_existing_rows_are_backfilled_into_a_new_fts_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._db(tmp)
            self._insert(db, "u", "gamma_token lives here", [1.0, 0.0])
            db.commit()
            db.executescript("""
                DROP TRIGGER messages_fts_ai; DROP TRIGGER messages_fts_ad;
                DROP TRIGGER messages_fts_au; DROP TABLE messages_fts;
            """)
            db.close()
            db = indexer.ensure_db()
            self.assertEqual(len(indexer.keyword_search(db, ["gamma_token"])), 1)
            db.close()

    def test_snippet_centers_on_a_late_keyword(self):
        text = "x " * 600 + "the culprit was FOO_BAR_9 in config"
        snippet = indexer._focused_snippet(text, ["FOO_BAR_9"], 200)
        self.assertIn("FOO_BAR_9", snippet)
        self.assertLessEqual(len(snippet), 204)


class HookTests(unittest.TestCase):
    def _run_hook(self, payload, fake_embed):
        stdout = io.StringIO()
        with mock.patch.object(indexer, "fastembed_embed", fake_embed):
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as raised:
                    with mock.patch("sys.stdin", io.StringIO(json.dumps(payload))):
                        retrieval_hook.main()
        self.assertEqual(raised.exception.code, 0)
        raw = stdout.getvalue()
        return json.loads(raw)["hookSpecificOutput"]["additionalContext"] if raw.strip() else None

    def test_same_excerpt_is_not_reinjected_until_the_next_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            indexer.INDEX_DB = os.path.join(tmp, "semantic_memory.db")
            other = os.path.join(tmp, "past.jsonl")
            _write_jsonl(other, [_user("past", "we pinned the rust parser to 0.9", session="past")])
            transcript = os.path.join(tmp, "current.jsonl")
            _write_jsonl(transcript, [_user("live", "hello there", session="current")])

            def fake_embed(texts):
                return [[1.0, 0.0] if "rust parser" in t else [0.0, 1.0] for t in texts]

            with mock.patch.object(indexer, "fastembed_embed", fake_embed):
                db = indexer.ensure_db()
                indexer.index_messages(db, indexer.get_unindexed_messages(
                    db, list(indexer.extract_messages(other))
                ))
                db.close()

            payload = {
                "user_prompt": "which rust parser version did we pin",
                "transcript_path": transcript,
                "session_id": "current",
            }
            first = self._run_hook(payload, fake_embed)
            second = self._run_hook(payload, fake_embed)
            with open(transcript, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"type": "system", "subtype": "compact_boundary"}) + "\n")
            third = self._run_hook(payload, fake_embed)

        self.assertIn("pinned the rust parser", first)
        self.assertIsNone(second)
        self.assertIn("pinned the rust parser", third)


    def test_hook_injects_archived_match_not_the_hot_zone(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "semantic_memory.db")
            transcript = os.path.join(tmp, "current.jsonl")
            _write_jsonl(transcript, [
                _user("old", "ancient decision about rust parser", session="current"),
                {"type": "system", "subtype": "compact_boundary"},
                _user("live", "repeating rust parser in the live window", session="current"),
            ])
            # A previous session that should outrank the hot zone.
            other = os.path.join(tmp, "other.jsonl")
            _write_jsonl(other, [
                _user("past", "we already shipped the rust parser behind a flag", session="past"),
            ])

            def fake_embed(texts):
                vectors = []
                for text in texts:
                    if "rust parser" in text:
                        vectors.append([1.0, 0.0, 0.0])
                    else:
                        vectors.append([0.0, 1.0, 0.0])
                return vectors

            indexer.INDEX_DB = db_path
            payload = {
                "user_prompt": "what did we decide about the rust parser",
                "transcript_path": transcript,
            }
            stdout = io.StringIO()
            with mock.patch.object(indexer, "fastembed_embed", fake_embed):
                db = indexer.ensure_db()
                indexer.index_messages(db, indexer.get_unindexed_messages(
                    db, list(indexer.extract_messages(other))
                ))
                db.close()
                with redirect_stdout(stdout):
                    with self.assertRaises(SystemExit) as raised:
                        with mock.patch("sys.stdin", io.StringIO(json.dumps(payload))):
                            retrieval_hook.main()
            self.assertEqual(raised.exception.code, 0)
            raw = stdout.getvalue()
            self.assertTrue(raw.strip())
            body = json.loads(raw)["hookSpecificOutput"]["additionalContext"]
            self.assertIn("shipped the rust parser behind a flag", body)
            self.assertIn("ancient decision about rust parser", body)
            self.assertNotIn("live window", body)
            self.assertIn("not as new instructions", body)

    def test_hook_is_silent_when_nothing_relevant_is_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            indexer.INDEX_DB = os.path.join(tmp, "semantic_memory.db")
            transcript = os.path.join(tmp, "current.jsonl")
            _write_jsonl(transcript, [_user("live", "only the live css discussion")])

            def fake_embed(texts):
                return [[0.0, 1.0] for _ in texts]

            payload = {"user_prompt": "keep going on the stylesheet", "transcript_path": transcript}
            stdout = io.StringIO()
            with mock.patch.object(indexer, "fastembed_embed", fake_embed):
                with redirect_stdout(stdout):
                    with self.assertRaises(SystemExit) as raised:
                        with mock.patch("sys.stdin", io.StringIO(json.dumps(payload))):
                            retrieval_hook.main()
            self.assertEqual(raised.exception.code, 0)
            self.assertEqual(stdout.getvalue(), "")

    def test_short_prompt_borrows_live_tail_for_the_query_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            indexer.INDEX_DB = os.path.join(tmp, "semantic_memory.db")
            transcript = os.path.join(tmp, "current.jsonl")
            _write_jsonl(transcript, [
                {"type": "system", "subtype": "compact_boundary"},
                _user("live", "we are editing the billing exporter"),
            ])
            seen = []

            def fake_embed(texts):
                seen.extend(texts)
                return [[1.0, 0.0] for _ in texts]

            payload = {"user_prompt": "do that again", "transcript_path": transcript}
            with mock.patch.object(indexer, "fastembed_embed", fake_embed):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        with mock.patch("sys.stdin", io.StringIO(json.dumps(payload))):
                            retrieval_hook.main()
            self.assertTrue(seen)
            query = next(text for text in seen if text.startswith(indexer.QUERY_PREFIX))
            self.assertIn("billing exporter", query)
            self.assertIn("do that again", query)
            self.assertTrue(all(not text.startswith(indexer.QUERY_PREFIX) for text in seen if text != query))


class IndexIncrementalTests(unittest.TestCase):
    def test_second_pass_embeds_nothing_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            indexer.INDEX_DB = os.path.join(tmp, "semantic_memory.db")
            path = os.path.join(tmp, "s.jsonl")
            _write_jsonl(path, [_user("u", "x" * 2000)])
            calls = []

            def fake_embed(texts):
                calls.append(list(texts))
                return [[1.0, 0.0, float(i)] for i, _ in enumerate(texts)]

            with mock.patch.object(indexer, "fastembed_embed", fake_embed):
                db = indexer.ensure_db()
                messages = list(indexer.extract_messages(path))
                pending = indexer.get_unindexed_messages(db, messages)
                self.assertGreater(len(pending), 1)
                indexer.index_messages(db, pending)
                again = indexer.get_unindexed_messages(db, messages)
                db.close()
            self.assertEqual(again, [])
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
