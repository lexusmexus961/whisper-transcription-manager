"""
Path remap after a move to another machine (database/path_remap.py).

What must hold: every stored path under the old root ends up under the new
one — in every database, every text column and config.yaml — while paths
that merely look similar stay untouched, a collision writes nothing, and a
second run changes nothing.
"""

import sqlite3

import pytest
import yaml

from whisper_tm.database import Database
from whisper_tm.database import path_remap as pr

OLD = "/mnt/oldnas"


def _remapper(new, *more):
    return pr.Remapper(pr.parse_mappings([(OLD, str(new)), *more]))


# ── Mappings ─────────────────────────────────────────────────────────────


class TestParseMappings:
    def test_canonical_form(self):
        assert pr.parse_mappings([("/mnt/oldnas/", "/srv//nas")]) == [("/mnt/oldnas", "/srv/nas")]

    def test_longest_prefix_first(self):
        got = pr.parse_mappings([("/mnt", "/a"), ("/mnt/nas", "/b")])
        assert [old for old, _ in got] == ["/mnt/nas", "/mnt"]

    @pytest.mark.parametrize(
        "old, new",
        [
            ("relative/path", "/srv"),
            ("/mnt/nas", "relative"),
            ("/", "/srv"),
            ("C:/", "/srv"),
            ("/mnt/nas", "/mnt/nas/"),
        ],
    )
    def test_refused(self, old, new):
        with pytest.raises(ValueError):
            pr.parse_mappings([(old, new)])

    def test_old_path_twice_is_refused(self):
        with pytest.raises(ValueError, match="twice"):
            pr.parse_mappings([("/a", "/b"), ("/a/", "/c")])


class TestRemapper:
    r = pr.Remapper([(OLD, "/srv/nas")])

    @pytest.mark.parametrize(
        "before, after",
        [
            (OLD, "/srv/nas"),
            (f"{OLD}/a b/c.mp4", "/srv/nas/a b/c.mp4"),
            (f"sync_mtime:{OLD}/.transcripts", "sync_mtime:/srv/nas/.transcripts"),
            (
                f"[Errno 13] Permission denied: '{OLD}/x'",
                "[Errno 13] Permission denied: '/srv/nas/x'",
            ),
            (
                f'{{"base": "{OLD}", "file": "{OLD}/a.mp4"}}',
                '{"base": "/srv/nas", "file": "/srv/nas/a.mp4"}',
            ),
        ],
    )
    def test_translates(self, before, after):
        assert self.r.hits(before)
        assert self.r(before) == after

    @pytest.mark.parametrize(
        "text",
        [
            "/mnt/oldnas2/a.mp4",  # a different share with a longer name
            "/data/mnt/oldnas/a.mp4",  # the old root further down a path
            "/mnt/OLDNAS/a.mp4",  # different case is a different path
            "/mnt/old",
            "",
        ],
    )
    def test_leaves_alone(self, text):
        assert not self.r.hits(text)
        assert self.r(text) == text

    def test_non_text_passes_through(self):
        assert self.r(None) is None and self.r(42) == 42

    def test_mappings_do_not_chain(self):
        r = pr.Remapper(pr.parse_mappings([("/a", "/b"), ("/b", "/c")]))
        assert r("/a/x") == "/b/x"
        assert r("/b/x") == "/c/x"

    def test_new_inside_old_is_idempotent(self):
        r = pr.Remapper(pr.parse_mappings([("/mnt", "/mnt/nas")]))
        once = r("/mnt/a.mp4")
        assert once == "/mnt/nas/a.mp4"
        assert not r.hits(once)
        assert r(once) == once
        # A sibling whose name only starts like the new root still moves.
        assert r("/mnt/nas2/a.mp4") == "/mnt/nas/nas2/a.mp4"

    def test_windows_old_root(self):
        r = pr.Remapper(pr.parse_mappings([("D:\\Media", "/srv/media")]))
        assert r("D:/Media/a.mp4") == "/srv/media/a.mp4"


# ── Databases ────────────────────────────────────────────────────────────

SAMPLE = {
    "filename": "a.mp4",
    "complete": True,
    "language": "de",
    "model": "turbo",
    "speakers": ["SPEAKER_00"],
    "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "Hello world"}],
}


@pytest.fixture
def app_db(tmp_path):
    """A main database with the real schema and rows under OLD."""
    path = tmp_path / "library.db"
    db = Database(path)
    db.initialize()
    db.upsert_media_file("a.mp4", abs_path=f"{OLD}/a.mp4")
    db.upsert_transcript("a.mp4", SAMPLE, media_dir=OLD, abs_path=f"{OLD}/a.mp4")
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO schema_info (key, value) VALUES (?, '1')",
            (f"sync_mtime:{OLD}/.transcripts",),
        )
    db.close_all()
    return path


def _values(path, sql):
    with sqlite3.connect(path) as con:
        return [r[0] for r in con.execute(sql)]


class TestRemapDatabase:
    def test_dry_run_counts_and_writes_nothing(self, app_db):
        rep = pr.remap_database(app_db, _remapper("/srv/nas"))
        found = {(h.table, h.column) for h in rep.hits}
        assert {
            ("media_files", "abs_path"),
            ("transcripts", "abs_path"),
            ("transcripts", "media_dir"),
            ("schema_info", "key"),
        } <= found
        assert not rep.applied
        assert _values(app_db, "SELECT abs_path FROM transcripts") == [f"{OLD}/a.mp4"]

    def test_apply_moves_every_row_and_the_app_finds_it(self, app_db):
        rep = pr.remap_database(app_db, _remapper("/srv/nas"), apply=True)
        assert rep.applied and not rep.conflicts
        assert _values(app_db, "SELECT abs_path FROM media_files") == ["/srv/nas/a.mp4"]
        assert _values(app_db, "SELECT media_dir FROM transcripts") == ["/srv/nas"]
        assert "sync_mtime:/srv/nas/.transcripts" in _values(app_db, "SELECT key FROM schema_info")

        db = Database(app_db)
        db.initialize()
        try:
            got = db.get_transcript("a.mp4", media_dir="/srv/nas")
            assert got and got["segments"][0]["text"] == "Hello world"
        finally:
            db.close_all()

    def test_second_run_finds_nothing(self, app_db):
        r = _remapper("/srv/nas")
        pr.remap_database(app_db, r, apply=True)
        assert pr.remap_database(app_db, r).rows == 0

    def test_collision_writes_nothing(self, app_db):
        """The server already ran on the new machine and indexed the media
        under the new path: moving the old rows there would violate the
        identity index."""
        db = Database(app_db)
        db.initialize()
        db.upsert_media_file("a.mp4", abs_path="/srv/nas/a.mp4")
        db.close_all()

        rep = pr.remap_database(app_db, _remapper("/srv/nas"), apply=True)
        assert not rep.applied
        assert [(c.table, c.column) for c in rep.conflicts] == [("media_files", "abs_path")]
        assert "/srv/nas/a.mp4" in rep.conflicts[0].examples[0]
        assert sorted(_values(app_db, "SELECT abs_path FROM media_files")) == [
            f"{OLD}/a.mp4",
            "/srv/nas/a.mp4",
        ]
        assert _values(app_db, "SELECT abs_path FROM transcripts") == [f"{OLD}/a.mp4"]

    def test_full_text_index_stays_consistent(self, tmp_path):
        """Like rag.db: an external-content FTS table fed by triggers. The
        remap writes the base table only; the triggers keep the index."""
        path = tmp_path / "rag.db"
        with sqlite3.connect(path) as con:
            con.executescript(f"""
                CREATE TABLE chunks (id INTEGER PRIMARY KEY, media_path TEXT,
                                     content TEXT, vector BLOB);
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    content, content='chunks', content_rowid='id');
                CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content)
                        VALUES ('delete', old.id, old.content);
                    INSERT INTO chunks_fts(rowid, content)
                        VALUES (new.id, new.content); END;
                INSERT INTO chunks VALUES (1, '{OLD}/a.mp4', 'hello world', x'00');
                INSERT INTO chunks_fts(rowid, content) VALUES (1, 'hello world');
            """)
        rep = pr.remap_database(path, _remapper("/srv/nas"), apply=True)
        assert [(h.table, h.column) for h in rep.hits] == [("chunks", "media_path")]
        with sqlite3.connect(path) as con:
            con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")
            assert con.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'hello'"
            ).fetchall() == [(1,)]

    def test_stored_and_stale_roots(self, app_db, tmp_path):
        assert pr.stored_roots(app_db) == {OLD: 1}
        assert pr.stale_roots(app_db) == {OLD: (1, "")}
        pr.remap_database(app_db, _remapper(tmp_path), apply=True)
        assert pr.stale_roots(app_db) == {}

    def test_nested_roots_count_towards_the_outer_one(self, app_db):
        db = Database(app_db)
        db.initialize()
        db.upsert_transcript("b.mp4", SAMPLE, media_dir=f"{OLD}/sub", abs_path=f"{OLD}/sub/b.mp4")
        db.close_all()
        assert pr.stored_roots(app_db) == {OLD: 2}

    def test_a_symlinked_root_is_stale(self, app_db, tmp_path):
        """The server resolves its media directory through symlinks, so rows
        stored under the link name are never looked up."""
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        pr.remap_database(app_db, _remapper(link), apply=True)
        assert pr.stale_roots(app_db) == {str(link): (1, str(real))}


# ── config.yaml ──────────────────────────────────────────────────────────

CONFIG = f"""\
# my settings
security:
  allowed_roots:
  - {OLD}
directories:
  media: {OLD}          # the NAS
  uploads: '{OLD}/uploads'
  work: /home/user/.cache/work
database:
  backup_path: {OLD}/Archive
"""


class TestConfigText:
    def test_keeps_comments_and_quoting(self):
        text, changed = pr.remap_config_text(CONFIG, _remapper("/srv/nas"))
        assert changed == 4
        assert "  media: /srv/nas          # the NAS\n" in text
        assert "  uploads: '/srv/nas/uploads'\n" in text
        assert text.startswith("# my settings\n")
        assert "/home/user/.cache/work" in text

    def test_refuses_a_path_that_would_change_the_yaml(self):
        with pytest.raises(ValueError, match="quotes"):
            pr.remap_config_text(CONFIG, _remapper("/srv/nas #2"))


# ── End to end ───────────────────────────────────────────────────────────


class TestRun:
    @pytest.fixture
    def install(self, tmp_path, app_db):
        new = tmp_path / "nas"
        new.mkdir()
        cfg = tmp_path / "config.yaml"
        text = CONFIG.replace("/home/user/.cache/work", str(tmp_path / "work"))
        cfg.write_text(text + f"  path: {app_db}\n", encoding="utf-8")
        return cfg, app_db, new

    def _run(self, cfg, new, **kw):
        lines = []
        code = pr.run([(OLD, str(new))], config_path=cfg, out=lines.append, **kw)
        return code, "\n".join(lines)

    def test_dry_run_reports_and_changes_nothing(self, install):
        cfg, db, new = install
        before = cfg.read_text()
        code, text = self._run(cfg, new)
        assert code == 0
        assert "Dry run" in text and "transcripts.abs_path" in text
        assert cfg.read_text() == before
        assert _values(db, "SELECT abs_path FROM transcripts") == [f"{OLD}/a.mp4"]

    def test_apply_translates_backs_up_and_is_repeatable(self, install):
        cfg, db, new = install
        code, text = self._run(cfg, new, apply=True)
        assert code == 0, text
        assert _values(db, "SELECT abs_path FROM transcripts") == [f"{new}/a.mp4"]
        data = yaml.safe_load(cfg.read_text())
        assert data["directories"]["media"] == str(new)
        assert data["security"]["allowed_roots"] == [str(new)]
        backups = sorted(p.name for p in db.parent.glob("*.before-path-remap-*"))
        assert any(n.startswith("library.before-path-remap-") for n in backups)
        assert any(n.startswith("config.before-path-remap-") for n in backups)

        code, text = self._run(cfg, new, apply=True)
        assert code == 0 and "Nothing to translate" in text

    def test_the_images_and_sounds_index_moves_too(self, install):
        """media.db keeps every recording's full path; it was left behind."""
        import sqlite3

        cfg, db, new = install
        media = db.parent / "media.db"
        with sqlite3.connect(media) as c:
            c.execute("CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE)")
            c.execute("INSERT INTO files (path) VALUES (?)", (f"{OLD}/a.mp4",))
        code, text = self._run(cfg, new, apply=True)
        assert code == 0, text
        assert _values(media, "SELECT path FROM files") == [f"{new}/a.mp4"]

    def test_the_audio_analysis_moves_too(self, install):
        """audio.db keeps every recording's full path (its fingerprint
        segments only ids)."""
        import sqlite3

        cfg, db, new = install
        audio = db.parent / "audio.db"
        with sqlite3.connect(audio) as c:
            c.execute("CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE)")
            c.execute("INSERT INTO files (path) VALUES (?)", (f"{OLD}/a.mp4",))
        code, text = self._run(cfg, new, apply=True)
        assert code == 0, text
        assert _values(audio, "SELECT path FROM files") == [f"{new}/a.mp4"]

    def test_a_symlink_target_is_resolved(self, install, tmp_path):
        cfg, db, new = install
        link = tmp_path / "link"
        link.symlink_to(new)
        code, text = self._run(cfg, link, apply=True)
        assert code == 0, text
        assert "is a symlink" in text
        assert _values(db, "SELECT abs_path FROM transcripts") == [f"{new}/a.mp4"]

    def test_bad_mapping_is_an_error(self, install):
        cfg, _db, _new = install
        lines = []
        assert pr.run([("relative", "/x")], config_path=cfg, out=lines.append) == 1
        assert "not absolute" in lines[0]


class TestDropConflicting:
    def test_the_new_servers_rows_make_way(self, app_db):
        """The new server indexed a.mp4 again under the new path and read the
        sidecar JSON in: its media row and transcript are dropped (the
        transcript with the media row, via the foreign key), the moved rows
        with their history take their place."""
        db = Database(app_db)
        db.initialize()
        db.upsert_media_file("a.mp4", abs_path="/srv/nas/a.mp4")
        db.upsert_transcript("a.mp4", SAMPLE, media_dir="/srv/nas", abs_path="/srv/nas/a.mp4")
        db.close_all()
        old_id = _values(app_db, f"SELECT id FROM transcripts WHERE abs_path = '{OLD}/a.mp4'")

        rep = pr.remap_database(app_db, _remapper("/srv/nas"), apply=True, drop_conflicting=True)
        # The media row at least; the transcript goes with it by the foreign
        # key, or is counted itself if its table comes first — the order
        # differs between SQLite versions, the result below doesn't
        assert rep.applied and rep.dropped >= 1
        assert _values(app_db, "SELECT abs_path FROM media_files") == ["/srv/nas/a.mp4"]
        assert _values(app_db, "SELECT id FROM transcripts") == old_id
        with sqlite3.connect(app_db) as con:
            assert con.execute("PRAGMA foreign_key_check").fetchall() == []
            con.execute("INSERT INTO segments_fts(segments_fts) VALUES('integrity-check')")

    def test_dry_run_still_writes_nothing(self, app_db):
        db = Database(app_db)
        db.initialize()
        db.upsert_media_file("a.mp4", abs_path="/srv/nas/a.mp4")
        db.close_all()
        rep = pr.remap_database(app_db, _remapper("/srv/nas"), drop_conflicting=True)
        assert not rep.applied and rep.dropped == 0
        assert len(_values(app_db, "SELECT id FROM media_files")) == 2


class TestChainedMappings:
    """A → B together with B → C: the rows at B are themselves moving on,
    so they are no collision — and must never be dropped as one."""

    @pytest.fixture
    def chain_db(self, tmp_path):
        path = tmp_path / "library.db"
        db = Database(path)
        db.initialize()
        for root in (OLD, "/srv/b"):
            db.upsert_media_file("a.mp4", abs_path=f"{root}/a.mp4")
            db.upsert_transcript("a.mp4", SAMPLE, media_dir=root, abs_path=f"{root}/a.mp4")
        db.close_all()
        return path

    def _chain(self):
        return pr.Remapper(pr.parse_mappings([(OLD, "/srv/b"), ("/srv/b", "/srv/c")]))

    def test_no_conflict_and_both_rows_move(self, chain_db):
        rep = pr.remap_database(chain_db, self._chain(), apply=True)
        assert not rep.conflicts and rep.applied
        assert sorted(_values(chain_db, "SELECT abs_path FROM transcripts")) == [
            "/srv/b/a.mp4",
            "/srv/c/a.mp4",
        ]
        assert sorted(_values(chain_db, "SELECT abs_path FROM media_files")) == [
            "/srv/b/a.mp4",
            "/srv/c/a.mp4",
        ]

    def test_drop_conflicting_drops_nothing(self, chain_db):
        rep = pr.remap_database(chain_db, self._chain(), apply=True, drop_conflicting=True)
        assert rep.dropped == 0
        assert len(_values(chain_db, "SELECT id FROM transcripts")) == 2

    def test_two_roots_onto_one_is_refused_even_with_drop(self, tmp_path):
        path = tmp_path / "library.db"
        db = Database(path)
        db.initialize()
        for root in (OLD, "/srv/b"):
            db.upsert_media_file("a.mp4", abs_path=f"{root}/a.mp4")
        db.close_all()
        r = pr.Remapper(pr.parse_mappings([(OLD, "/srv/c"), ("/srv/b", "/srv/c")]))
        rep = pr.remap_database(path, r, apply=True, drop_conflicting=True)
        assert not rep.applied and rep.conflicts and rep.dropped == 0
        assert sorted(_values(path, "SELECT abs_path FROM media_files")) == [
            f"{OLD}/a.mp4",
            "/srv/b/a.mp4",
        ]
