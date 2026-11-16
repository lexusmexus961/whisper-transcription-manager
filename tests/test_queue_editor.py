"""Queue Editor: bulk operations on the queue — core and API.

The editor changes the settings of every entry currently in the queue (e.g.
enable low-VRAM mode) or of selected entries, and removes whole folders of
entries in one call. Before it, waiting entries with a hardcoded
``low_vram: false`` could not be touched, and a folder's entries could only
be removed one call per entry.

The guarantee that everything here revolves around: **the number the dry run shows
is the number the click hits.** That's why exactly one function (``editor_where``)
builds the WHERE for counts, lists, folders, dry runs, and mutations;
that's why mutation locks with BEGIN IMMEDIATE before counting; that's why
``expected ≠ affected`` answers with 409 and writes nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import whisper_tm.extensions as ext
from whisper_tm.core import queue as q
from whisper_tm.database import Database

ROOT = "/m"


@pytest.fixture
def db(tmp_path, monkeypatch):
    q.stop_worker()
    d = Database(tmp_path / "q.db")
    d.initialize()
    monkeypatch.setattr(ext, "db", d)
    q._current = None
    q._abort = False
    q._paused = False
    q._folder_cache["gen"] = -1
    q._folder_cache["data"] = {}
    return d


def _fill(db, lines):
    """rows: (filename, status, settings|None[, base_dir]) → ids in order."""
    ids = []
    with db.connection() as conn:
        for i, z in enumerate(lines, start=1):
            fn, status, settings = z[0], z[1], z[2]
            bd = z[3] if len(z) > 3 else ROOT
            sj = settings if isinstance(settings, str) else json.dumps(settings or {})
            conn.execute(
                "INSERT INTO queue_items (filename, base_dir, settings_json, status, position)"
                " VALUES (?, ?, ?, ?, ?)",
                (fn, bd, sj, status, i),
            )
            ids.append(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    q._bump()
    return ids


def _settings(db, item_id):
    with db.connection() as conn:
        return json.loads(
            conn.execute(
                "SELECT settings_json FROM queue_items WHERE id = ?", (item_id,)
            ).fetchone()[0]
        )


def _order(db):
    with db.connection() as conn:
        return [
            r[0]
            for r in conn.execute(
                "SELECT id FROM queue_items WHERE status='pending' ORDER BY position, id"
            )
        ]


INVENTORY = [
    ("videos/Shows/S01/E01.mkv", "pending", {"low_vram": False}),
    ("videos/Shows/S01/E02.mkv", "pending", {"low_vram": False, "model": "large-v3"}),
    ("videos/Shows/film.mkv", "pending", {}),
    ("videos/yt/a.mp4", "pending", {"low_vram": True}),
    ("videos/yt/b.mp4", "pending", "broken{"),  # invalid JSON
    ("videos/yt/c.mp4", "processing", {}),
    ("videos/yt/d.mp4", "error", {}),
    ("videos/yt/e.mp4", "done", {}),
    ("audio/x.mp3", "pending", {"low_vram": False}, "/other"),
]


# ── Filters and WHERE ─────────────────────────────────────────────────────


class TestFilterNormalization:
    def test_the_default_is_pending(self):
        f = q.editor_normalize_filter(None)
        assert f == {
            "status": ["pending"],
            "dir": None,
            "prefix": "",
            "recursive": True,
            "q": "",
            "name": "",
            "setting": None,
        }

    def test_prefix_gets_a_trailing_slash_and_stays_relative(self):
        assert q.editor_normalize_filter({"prefix": " videos/Shows "})["prefix"] == "videos/Shows/"
        with pytest.raises(q.TargetError, match="relative"):
            q.editor_normalize_filter({"prefix": "/etc"})
        with pytest.raises(q.TargetError, match="relative"):
            q.editor_normalize_filter({"prefix": "a/../b"})

    def test_status_rules(self):
        assert q.editor_normalize_filter({"status": "pending,error"})["status"] == [
            "pending",
            "error",
        ]
        with pytest.raises(q.TargetError, match="processing"):
            q.editor_normalize_filter({"status": ["processing"]})
        with pytest.raises(q.TargetError, match="Unknown status"):
            q.editor_normalize_filter({"status": ["nonsense"]})

    def test_dir_must_be_absolute_empty_allowed(self):
        assert q.editor_normalize_filter({"dir": ""})["dir"] == ""
        assert q.editor_normalize_filter({"dir": "/m/"})["dir"] == "/m"
        with pytest.raises(q.TargetError, match="absolute"):
            q.editor_normalize_filter({"dir": "relative/x"})

    def test_recursive_and_q(self):
        f = q.editor_normalize_filter({"recursive": "0", "q": "x" * 300})
        assert f["recursive"] is False and len(f["q"]) == 200

    def test_name(self):
        assert q.editor_normalize_filter({"name": "  intro "})["name"] == "intro"
        assert len(q.editor_normalize_filter({"name": "y" * 300})["name"]) == 200

    def test_setting(self):
        f = q.editor_normalize_filter({"setting": {"key": "low_vram", "value": True}})
        assert f["setting"] == {"key": "low_vram", "value": True}
        with pytest.raises(q.TargetError):
            q.editor_normalize_filter({"setting": {"key": "x;drop", "value": 1}})
        with pytest.raises(q.TargetError, match="an object"):
            q.editor_normalize_filter("nothing")


class TestEditorWhere:
    def test_ids_as_one_parameter(self):
        w, p = q.editor_where({"ids": [3, 1, 2]}, fixed_status="pending")
        assert w == "status = ? AND id IN (SELECT value FROM json_each(?))"
        assert p == ["pending", "[3, 1, 2]"]

    def test_ids_limits(self):
        with pytest.raises(q.TargetError, match="empty"):
            q.editor_where({"ids": []}, fixed_status="pending")
        with pytest.raises(q.TargetError, match="must be numbers"):
            q.editor_where({"ids": ["a"]}, fixed_status="pending")
        with pytest.raises(q.TargetError, match="Too many"):
            q.editor_where({"ids": list(range(q.EDITOR_IDS_MAX + 1))}, fixed_status="pending")

    def test_prefix_is_a_range_comparison(self):
        w, p = q.editor_where({"filter": {"prefix": "videos/Shows"}}, fixed_status="pending")
        assert "filename >= ? AND filename < ?" in w
        assert p[1:3] == ["videos/Shows/", "videos/Shows0"]
        assert "instr" not in w

    def test_not_recursive(self):
        w, p = q.editor_where({"filter": {"prefix": "a/", "recursive": False}}, fixed_status=None)
        assert "instr(substr(filename, ?), '/') = 0" in w and p[-1] == 3

    def test_not_recursive_at_the_root(self):
        """Without a prefix, "subfolders off" once did not apply at all."""
        w, p = q.editor_where({"filter": {"recursive": False}}, fixed_status=None)
        assert "instr(substr(filename, ?), '/') = 0" in w and p[-1] == 1

    def test_a_fixed_status_tuple_intersects_with_the_filter(self):
        w, p = q.editor_where(
            {"filter": {"status": ["error"]}}, fixed_status=("error", "cancelled")
        )
        assert w.startswith("status = ?") and p[0] == "error"
        w, p = q.editor_where(
            {"filter": {"status": ["error", "cancelled"]}}, fixed_status=("error", "cancelled")
        )
        assert w.startswith("status IN (?,?)") and p[:2] == ["error", "cancelled"]
        w, p = q.editor_where({"ids": [1, 2]}, fixed_status=("error", "cancelled"))
        assert w.startswith("status IN (?,?)") and p[:2] == ["error", "cancelled"]
        with pytest.raises(q.TargetError, match="error or cancelled"):
            q.editor_where({"filter": {"status": ["pending"]}}, fixed_status=("error", "cancelled"))

    def test_q_is_escaped_and_the_star(self):
        w, p = q.editor_where({"filter": {"q": "50%_*x"}}, fixed_status=None)
        assert "LIKE ? ESCAPE" in w and p[-1] == "%50\\%\\_%x%"

    def test_setting_with_the_json_valid_guard(self):
        w, p = q.editor_where(
            {"filter": {"setting": {"key": "low_vram", "value": True}}}, fixed_status=None
        )
        assert "json_valid(settings_json)" in w and p[-2:] == ["$.low_vram", 1]
        w, p = q.editor_where(
            {"filter": {"setting": {"key": "low_vram", "value": None}}}, fixed_status=None
        )
        assert w.endswith("IS NULL") and p[-1] == "$.low_vram"

    def test_prefix_uses_the_index_not_a_scan(self, db):
        _fill(db, INVENTORY)
        w, p = q.editor_where({"filter": {"prefix": "videos/Shows/"}}, fixed_status="pending")
        with db.connection() as conn:
            plan = " ".join(
                r[3]
                for r in conn.execute(
                    f"EXPLAIN QUERY PLAN SELECT COUNT(*) FROM queue_items WHERE {w}", p
                )
            )
        assert "SCAN" not in plan, plan

    def test_there_is_only_one_where_builder(self):
        """Code contract: no second WHERE builder for the editor."""
        source = Path("whisper_tm/core/queue.py").read_text()
        block = source[
            source.index("# ── Queue editor (bulk operations)") : source.index("def requeue_stale")
        ]
        hits = re.findall(r"WHERE \{(\w+)\}", block)
        assert hits
        # Every variable inserted into a WHERE comes from editor_where(...)
        for name in set(hits):
            assert re.search(rf"\b{name}, \w+ = editor_where\(", block), name


# ── List and folders ─────────────────────────────────────────────────────


class TestTheList:
    def test_the_listing_counts_and_paginates(self, db):
        _fill(db, INVENTORY)
        r = q.editor_list({}, offset=0, limit=2)
        assert r["total"] == 6 and len(r["items"]) == 2 and r["limit"] == 2
        assert r["items"][0]["filename"] == "videos/Shows/S01/E01.mkv"
        assert r["processing"]["filename"] == "videos/yt/c.mp4"
        assert {x["dir"]: x["count"] for x in r["roots"]} == {"/m": 5, "/other": 1}
        r2 = q.editor_list({}, offset=4, limit=2)
        assert [i["filename"] for i in r2["items"]] == ["videos/yt/b.mp4", "audio/x.mp3"]

    def test_filter_prefix_dir_q_setting(self, db):
        _fill(db, INVENTORY)
        assert q.editor_list({"prefix": "videos/Shows/"})["total"] == 3
        assert q.editor_list({"prefix": "videos/Shows/", "recursive": False})["total"] == 1
        assert q.editor_list({"recursive": False})["total"] == 0  # nothing directly at the root
        assert q.editor_list({"dir": "/other"})["total"] == 1
        assert q.editor_list({"q": "E0*mkv"})["total"] == 2
        assert q.editor_list({"setting": {"key": "low_vram", "value": True}})["total"] == 1
        # null = key missing OR JSON broken
        assert q.editor_list({"setting": {"key": "low_vram", "value": None}})["total"] == 2
        assert q.editor_list({"status": ["error"]})["total"] == 1

    def test_the_file_name_alone(self, db):
        """Narrowing down by file name: the path search finds "Shows" in every
        file of the Shows folder — the name filter only in a name."""
        _fill(
            db,
            [
                *INVENTORY,
                ("Shows-Collection.txt", "pending", {}),
                ("music/Élan Vital.mp3", "pending", {}),
            ],
        )
        assert q.editor_list({"q": "Shows"})["total"] == 4
        assert [i["filename"] for i in q.editor_list({"name": "shows"})["items"]] == [
            "Shows-Collection.txt"
        ]
        assert q.editor_list({"name": "E0*mkv"})["total"] == 2
        assert q.editor_list({"name": "élan"})["total"] == 1  # case ignored beyond ASCII
        assert q.editor_list({"q": "videos/yt", "name": "a.mp4"})["total"] == 1  # both narrow
        assert q.editor_list({"name": "50%"})["total"] == 0  # the wildcards are escaped
        # the tree counts by it too, and keeps its own cache entry
        assert q.editor_folders({"name": "shows"})["files_here"] == 1
        assert q.editor_folders({})["files_here"] == 1 and q.editor_folders({})["total"] == 8
        assert "Name 'shows'" in q._editor_description({"filter": {"name": "shows"}})

    def test_broken_json_does_not_break_the_listing(self, db):
        _fill(db, INVENTORY)
        r = q.editor_list({"q": "b.mp4"})
        assert r["total"] == 1 and r["items"][0]["settings"] == {}

    def test_limit_clamped(self, db):
        _fill(db, INVENTORY)
        assert q.editor_list({}, limit=99999)["limit"] == q.EDITOR_LIMIT_MAX
        assert q.editor_list({}, offset=-5)["offset"] == 0


class TestFolders:
    def test_children_at_depth_one_per_root(self, db):
        _fill(db, INVENTORY)
        r = q.editor_folders({})
        assert r["total"] == 6 and r["files_here"] == 0
        assert [(f["name"], f["dir"], f["count"], f["has_children"]) for f in r["folders"]] == [
            ("videos", "/m", 5, True),
            ("audio", "/other", 1, False),
        ]
        r2 = q.editor_folders({"prefix": "videos/Shows/"})
        assert r2["files_here"] == 1
        assert [(f["name"], f["count"]) for f in r2["folders"]] == [("S01", 2)]
        assert r2["folders"][0]["prefix"] == "videos/Shows/S01/"

    def test_roots_stay_when_a_root_and_folder_are_chosen(self, db):
        """Root selected → dropdown must not clear itself."""
        _fill(db, INVENTORY)
        all_items = q.editor_folders({})["roots"]
        assert all_items == [{"dir": "/m", "count": 5}, {"dir": "/other", "count": 1}]
        assert q.editor_folders({"dir": "/other"})["roots"] == all_items
        assert q.editor_folders({"prefix": "videos/Shows/", "dir": "/m"})["roots"] == all_items
        assert q.editor_folders({"q": "audio"})["roots"] == all_items
        # but status matters: in error view only error roots
        assert q.editor_folders({"status": ["error"]})["roots"] == [{"dir": "/m", "count": 1}]

    def test_cache_per_generation(self, db):
        _fill(db, INVENTORY)
        a = q.editor_folders({})
        with db.connection() as conn:
            conn.execute("DELETE FROM queue_items WHERE filename = 'audio/x.mp3'")
        assert q.editor_folders({})["total"] == 6  # still cached (no _bump)
        q._bump()
        assert q.editor_folders({})["total"] == 5
        assert a["generation"] != q.editor_folders({})["generation"]


# ── Mutations ────────────────────────────────────────────────────────────


class TestSettings:
    def test_low_vram_for_everything_pending(self, db):
        """The main use case: low-VRAM mode for everything pending."""
        ids = _fill(db, INVENTORY)
        dry = q.editor_settings({"filter": {}}, {"low_vram": True}, dry_run=True)
        assert dry["dry_run"] is True and dry["affected"] == 6
        assert dry["skipped_processing"] == 1
        assert _settings(db, ids[0]) == {"low_vram": False}  # nothing happened yet
        r = q.editor_settings({"filter": {}}, {"low_vram": True}, expected=6)
        assert r["affected"] == 6 and r["dry_run"] is False
        assert _settings(db, ids[0]) == {"low_vram": True}
        assert _settings(db, ids[1]) == {"low_vram": True, "model": "large-v3"}  # foreign key stays
        assert _settings(db, ids[4]) == {"low_vram": True}  # broken JSON repaired
        assert _settings(db, ids[5]) == {}  # processing untouched
        assert _settings(db, ids[6]) == {}  # error untouched

    def test_null_removes_the_key(self, db):
        ids = _fill(db, INVENTORY)
        r = q.editor_settings({"ids": [ids[1]]}, {"model": None})
        assert r["affected"] == 1 and _settings(db, ids[1]) == {"low_vram": False}

    def test_an_ids_target_reports_requested(self, db):
        ids = _fill(db, INVENTORY)
        r = q.editor_settings({"ids": [ids[0], ids[5], 999999]}, {"low_vram": True})
        assert r["requested"] == 3 and r["affected"] == 1 and r["skipped_processing"] == 1

    def test_except_ids(self, db):
        ids = _fill(db, INVENTORY)
        r = q.editor_settings(
            {"filter": {"prefix": "videos/Shows/"}, "except_ids": [ids[2]]}, {"low_vram": True}
        )
        assert r["affected"] == 2 and _settings(db, ids[2]) == {}
        assert "minus 1" in r["description"]

    def test_the_expectation_protects(self, db):
        ids = _fill(db, INVENTORY)
        with pytest.raises(q.HitCountChanged) as e:
            q.editor_settings({"filter": {}}, {"low_vram": True}, expected=5)
        assert e.value.actual == 6
        assert _settings(db, ids[0]) == {"low_vram": False}  # nothing written

    def test_generation_and_qlog(self, db):
        _fill(db, INVENTORY)
        before = q._gen()
        q._log.clear()
        r = q.editor_settings({"filter": {}}, {"low_vram": True, "vocab_preset": None})
        assert r["generation"] > before
        assert any(
            "✎ Queue editor: 6" in z and "low_vram=true, vocab_preset→default" in z for z in q._log
        )

    def test_an_empty_patch(self, db):
        _fill(db, INVENTORY)
        with pytest.raises(q.TargetError):
            q.editor_settings({"filter": {}}, {})


class TestRemove:
    def test_only_pending(self, db):
        _fill(db, INVENTORY)
        r = q.editor_remove({"filter": {"prefix": "videos/"}})
        assert r["affected"] == 5  # 3 Shows + yt/a + yt/b (pending)
        with db.connection() as conn:
            remainder = {
                r[0]: r[1] for r in conn.execute("SELECT filename, status FROM queue_items")
            }
        # processing, error, done under videos/ survive; other root untouched
        assert remainder == {
            "videos/yt/c.mp4": "processing",
            "videos/yt/d.mp4": "error",
            "videos/yt/e.mp4": "done",
            "audio/x.mp3": "pending",
        }

    def test_a_dry_run_deletes_nothing(self, db):
        _fill(db, INVENTORY)
        r = q.editor_remove({"filter": {}}, dry_run=True)
        assert r["affected"] == 6 and q.editor_list({})["total"] == 6
        assert len(r["sample"]) == 5


class TestMove:
    def test_moving_to_the_front_keeps_the_order(self, db):
        ids = _fill(db, INVENTORY)
        before = _order(db)
        r = q.editor_move({"ids": [ids[3], ids[8], ids[1]]}, "top")
        assert r["affected"] == 3 and r["to"] == "top"
        after = _order(db)
        assert after[:3] == [ids[1], ids[3], ids[8]]  # in their previous order
        assert after[3:] == [i for i in before if i not in (ids[1], ids[3], ids[8])]

    def test_moving_to_the_back_goes_behind_everything(self, db):
        ids = _fill(db, INVENTORY)
        r = q.editor_move({"filter": {"prefix": "videos/Shows/"}}, "bottom")
        assert r["affected"] == 3
        assert _order(db)[-3:] == ids[:3]
        with db.connection() as conn:
            mx = conn.execute(
                "SELECT MAX(position) FROM queue_items WHERE status != 'pending'"
            ).fetchone()[0]
            mn = conn.execute(
                "SELECT MIN(position) FROM queue_items WHERE id = ?", (ids[0],)
            ).fetchone()[0]
        assert mn > mx

    def test_invalid_direction(self, db):
        with pytest.raises(q.TargetError, match="top"):
            q.editor_move({"filter": {}}, "left")

    def test_processing_is_never_moved(self, db):
        ids = _fill(db, INVENTORY)
        r = q.editor_move({"ids": [ids[5]]}, "top")
        assert r["affected"] == 0 and r["skipped_processing"] == 1


class TestSnapshotGeneration:
    def test_the_snapshot_carries_the_generation(self, db):
        g = q.snapshot()["generation"]
        q._bump()
        assert q.snapshot()["generation"] == g + 1


class TestRetrying:
    def test_the_filter_hits_only_the_displayed_status(self, db):
        """Error view shows 2 → retry hits 2, not the cancelled ones too."""
        _fill(
            db, INVENTORY + [("videos/yt/f.mp4", "cancelled", {}), ("videos/yt/g.mp4", "error", {})]
        )
        dry = q.editor_retry({"filter": {"status": ["error"]}}, dry_run=True)
        assert dry["affected"] == 2 and dry["dry_run"] is True
        # the running c.mp4 is in the same folder — it was never a retry candidate
        assert dry["skipped_processing"] == 0
        assert (
            q.editor_settings(
                {"filter": {"prefix": "videos/yt/"}}, {"low_vram": True}, dry_run=True
            )["skipped_processing"]
            == 1
        )
        r = q.editor_retry({"filter": {"status": ["error"]}}, expected=2)
        assert r["affected"] == 2
        with db.connection() as conn:
            st = {
                row[0]: row[1] for row in conn.execute("SELECT filename, status FROM queue_items")
            }
        assert st["videos/yt/d.mp4"] == "pending" and st["videos/yt/g.mp4"] == "pending"
        assert st["videos/yt/f.mp4"] == "cancelled"
        assert st["videos/yt/c.mp4"] == "processing" and st["videos/yt/e.mp4"] == "done"

    def test_ids_hit_only_error_and_cancelled(self, db):
        ids = _fill(db, INVENTORY + [("videos/yt/f.mp4", "cancelled", {})])
        r = q.editor_retry({"ids": [ids[0], ids[5], ids[6], ids[7], ids[9]]})
        assert r["affected"] == 2 and r["requested"] == 5
        with db.connection() as conn:
            assert (
                conn.execute("SELECT COUNT(*) FROM queue_items WHERE status='pending'").fetchone()[
                    0
                ]
                == 8
            )

    def test_a_pending_filter_is_400_and_expected_protects(self, db):
        ids = _fill(db, INVENTORY)
        with pytest.raises(q.TargetError):
            q.editor_retry({"filter": {"status": ["pending"]}}, dry_run=True)
        with pytest.raises(q.HitCountChanged):
            q.editor_retry({"filter": {"status": ["error"]}}, expected=5)
        with db.connection() as conn:
            assert (
                conn.execute("SELECT status FROM queue_items WHERE id=?", (ids[6],)).fetchone()[0]
                == "error"
            )


# ── API ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client(db, monkeypatch):
    from flask import Flask

    from whisper_tm.api import queue as api

    app = Flask(__name__)
    app.register_blueprint(api.queue_bp)
    return app.test_client()


class TestApi:
    def test_items_and_folders(self, client, db):
        _fill(db, INVENTORY)
        r = client.get("/api/queue/editor/items?prefix=videos/Shows&limit=2").get_json()
        assert r["ok"] is True and r["total"] == 3 and len(r["items"]) == 2
        assert r["filter"]["prefix"] == "videos/Shows/"
        assert "generation" in r
        r = client.get(
            "/api/queue/editor/items?setting_key=low_vram&setting_value=false"
        ).get_json()
        assert r["total"] == 3
        r = client.get("/api/queue/editor/items?status=pending,error").get_json()
        assert r["total"] == 7
        f = client.get("/api/queue/editor/folders").get_json()
        assert f["ok"] is True and f["folders"][0]["name"] == "videos"
        r = client.get("/api/queue/editor/items?name=E0").get_json()
        assert r["total"] == 2 and r["filter"]["name"] == "E0"

    def test_filter_errors_are_400(self, client, db):
        assert client.get("/api/queue/editor/items?status=processing").status_code == 400
        assert client.get("/api/queue/editor/items?prefix=/etc").status_code == 400
        assert client.get("/api/queue/editor/items?setting_key=nothing").status_code == 400
        assert client.get("/api/queue/editor/items?offset=x").status_code == 200

    def test_settings_dry_run_then_for_real(self, client, db):
        ids = _fill(db, INVENTORY)
        r = client.post(
            "/api/queue/editor/settings",
            json={"all": True, "settings": {"low_vram": True, "nonsense": 1}, "dry_run": True},
        ).get_json()
        assert r["dry_run"] is True and r["affected"] == 6 and r["ignored_keys"] == ["nonsense"]
        assert _settings(db, ids[0]) == {"low_vram": False}
        r = client.post(
            "/api/queue/editor/settings",
            json={"all": True, "settings": {"low_vram": True}, "expected": 6},
        ).get_json()
        assert r["affected"] == 6 and r["patch"] == {"low_vram": True} and r["hint"] is None
        assert _settings(db, ids[0]) == {"low_vram": True}

    def test_low_vram_false_needs_no_hint(self, client, db):
        """The entry's choice wins over hardware.low_vram
        (core/vram.low_vram) — the old warning that false could not beat
        the global setting would now be wrong."""
        _fill(db, INVENTORY)
        r = client.post(
            "/api/queue/editor/settings", json={"all": True, "settings": {"low_vram": False}}
        ).get_json()
        assert r["ok"] is True and r["hint"] is None

    def test_409_when_the_count_changed(self, client, db):
        _fill(db, INVENTORY)
        r = client.post("/api/queue/editor/remove", json={"all": True, "expected": 5})
        assert r.status_code == 409
        d = r.get_json()
        assert d["affected"] == 6 and "expected 5, now 6" in d["error"]
        assert q.editor_list({})["total"] == 6

    def test_target_errors(self, client, db):
        p = "/api/queue/editor/remove"
        assert client.post(p, json={}).status_code == 400
        assert client.post(p, json={"ids": [1], "all": True}).status_code == 400
        assert client.post(p, json={"ids": "1,2"}).status_code == 400
        assert client.post(p, json={"ids": []}).status_code == 400
        assert client.post(p, json={"filter": "x"}).status_code == 400
        assert client.post(p, json={"all": True, "except_ids": "x"}).status_code == 400
        assert client.post(p, json={"all": True, "expected": "many"}).status_code == 400
        # Garbage must never be 500
        assert client.post(p, json={"ids": [10**30]}).status_code < 500
        assert client.post(p, json={"filter": {"status": 5}}).status_code == 400

    def test_settings_types(self, client, db):
        p = "/api/queue/editor/settings"
        assert (
            client.post(p, json={"all": True, "settings": {"low_vram": "yes"}}).status_code == 400
        )
        assert (
            client.post(p, json={"all": True, "settings": {"model": {"a": 1}}}).status_code == 400
        )
        assert client.post(p, json={"all": True, "settings": {"nonsense": 1}}).status_code == 400
        assert client.post(p, json={"all": True, "settings": "x"}).status_code == 400
        assert client.post(p, json={"all": True}).status_code == 400

    def test_move(self, client, db):
        ids = _fill(db, INVENTORY)
        assert client.post("/api/queue/editor/move", json={"ids": [ids[3]]}).status_code == 400
        r = client.post("/api/queue/editor/move", json={"ids": [ids[3]], "to": "top"}).get_json()
        assert r["affected"] == 1 and r["to"] == "top"
        assert _order(db)[0] == ids[3]

    def test_remove_with_except(self, client, db):
        ids = _fill(db, INVENTORY)
        r = client.post(
            "/api/queue/editor/remove",
            json={"filter": {"prefix": "videos/Shows/"}, "except_ids": [ids[0]]},
        ).get_json()
        assert r["affected"] == 2 and q.editor_list({"prefix": "videos/Shows/"})["total"] == 1

    def test_retry_route(self, client, db):
        ids = _fill(db, INVENTORY + [("videos/yt/f.mp4", "cancelled", {})])
        p = "/api/queue/editor/retry"
        assert (
            client.post(p, json={"all": True, "dry_run": True}).status_code == 400
        )  # all = pending
        r = client.post(p, json={"filter": {"status": ["cancelled"]}, "dry_run": True}).get_json()
        assert r["affected"] == 1 and r["dry_run"] is True
        r = client.post(p, json={"filter": {"status": ["cancelled"]}, "expected": 1}).get_json()
        assert r["affected"] == 1
        assert client.post(p, json={"ids": [ids[6]], "expected": 5}).status_code == 409
        r = client.post(p, json={"ids": [ids[6]]}).get_json()
        assert r["affected"] == 1
        assert q.editor_list({"status": ["error", "cancelled"]})["total"] == 0

    def test_the_queue_route_carries_the_generation(self, client, db):
        r = client.get("/api/queue").get_json()
        assert isinstance(r.get("generation"), int)

    def test_existing_routes_unchanged(self, client, db):
        """The single-entry routes: remove/move still take exactly one id."""
        ids = _fill(db, INVENTORY)
        assert client.post("/api/queue/remove", json={"id": ids[0]}).get_json()["ok"] is True
        assert client.post("/api/queue/remove", json={}).status_code == 400
