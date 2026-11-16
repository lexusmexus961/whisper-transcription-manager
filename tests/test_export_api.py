"""Export endpoints (whisper_tm/api/export.py) and export jobs
(whisper_tm/core/export_jobs.py): selections, counts, preview, direct
downloads, jobs with progress/cancel/download."""

from __future__ import annotations

import io
import threading
import time
import zipfile
from pathlib import Path

import pytest
from flask import Flask

from whisper_tm.database import Database

ROOT = "/wtm-test-library/media"  # (not /lib: a symlink on some systems)


def _tx(text):
    return {
        "complete": True,
        "language": "en",
        "total_duration_sec": 10,
        "speaker_names": {"SPEAKER_00": "Alice"},
        "segments": [
            {"start": 0, "end": 3, "speaker": "SPEAKER_00", "text": text},
            {"start": 4, "end": 6, "speaker": "SPEAKER_01", "text": "Fine."},
        ],
    }


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    now = time.time()
    with d.connection() as c:
        for rel, status, video, age in (
            ("talks/a.mp3", "done", 0, 0),
            ("talks/b.mp4", "done", 1, 3 * 86400),
            ("talks/sub/c.mp3", "done", 0, 0),
            ("talks/d.mp3", "pending", 0, 0),
            ("other/e.mp3", "done", 0, 0),
        ):
            c.execute(
                "INSERT INTO media_files (rel_path, abs_path, status, is_video, created_at) VALUES (?, ?, ?, ?, ?)",
                (rel, f"{ROOT}/{rel}", status, video, now - age),
            )
    for rel, text in (
        ("talks/a.mp3", "Hello from a."),
        ("talks/b.mp4", "Hello from b."),
        ("talks/sub/c.mp3", "Hello from c."),
        ("other/e.mp3", "Hello from e."),
    ):
        d.upsert_transcript(rel, _tx(text), media_dir=ROOT, abs_path=f"{ROOT}/{rel}")
    d.replace_translation(f"{ROOT}/talks/a.mp3", "es", [{"id": 0, "text": "Hola desde a."}])
    return d


@pytest.fixture
def client(db, monkeypatch, tmp_path):
    import whisper_tm.core.security as sec
    from whisper_tm import extensions as ext
    from whisper_tm.api import export
    from whisper_tm.core import export_jobs

    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(sec, "allowed_roots", lambda: [Path(ROOT)])
    monkeypatch.setattr(
        export_jobs,
        "folder",
        lambda: (tmp_path / "exports").mkdir(exist_ok=True) or tmp_path / "exports",
    )
    app = Flask(__name__)
    app.register_blueprint(export.export_bp)
    return app.test_client()


def test_a_folder_with_or_without_subfolders(client):
    d = client.post("/api/export/count", json={"folders": [{"path": ROOT + "/talks"}]}).get_json()
    assert (
        d["count"] == 4
        and d["transcribed"] == 3
        and d["videos"] == 1
        and d["base"] == ROOT + "/talks"
    )
    assert d["names"] == ["a.mp3", "b.mp4", "d.mp3", "sub/c.mp3"]
    d = client.post(
        "/api/export/count", json={"folders": [{"path": ROOT + "/talks", "subfolders": False}]}
    ).get_json()
    assert d["count"] == 3


def test_only_what_was_added_today(client):
    d = client.post(
        "/api/export/count", json={"folders": [{"path": ROOT + "/talks"}], "added": "today"}
    ).get_json()
    assert d["count"] == 3, "b.mp4 was added three days ago"
    d = client.post(
        "/api/export/count", json={"folders": [{"path": ROOT + "/talks"}], "added": "7"}
    ).get_json()
    assert d["count"] == 4
    assert client.post("/api/export/count", json={"added": "soon"}).status_code == 400


def test_too_many_are_counted_but_not_exported(client, monkeypatch):
    from whisper_tm.api import export

    monkeypatch.setattr(export, "MAX_FILES", 2)
    d = client.post("/api/export/count", json={"folders": [{"path": ROOT}]}).get_json()
    assert d["count"] == 5 and d["too_many"] is True
    assert (
        client.post("/api/export/files", json={"folders": [{"path": ROOT}]}).get_json()["total"]
        == 5
    )
    assert (
        client.post("/api/export", json={"folders": [{"path": ROOT}], "format": "txt"}).status_code
        == 400
    )


def test_outside_the_library_is_refused(client):
    assert client.post("/api/export/count", json={"files": ["/etc/passwd"]}).status_code == 400
    assert client.post("/api/export/count", json={"folders": [{"path": "/etc"}]}).status_code == 400
    assert (
        client.post("/api/export", json={"files": [ROOT + "/../../etc/passwd"]}).status_code == 400
    )


def test_one_file_comes_back_as_itself(client):
    r = client.post("/api/export", json={"files": [ROOT + "/talks/a.mp3"], "format": "srt"})
    assert r.status_code == 200 and "filename=a.srt" in r.headers["Content-Disposition"]
    assert b"Alice: Hello from a." in r.data
    r = client.post(
        "/api/export",
        json={
            "files": [ROOT + "/talks/a.mp3"],
            "format": "txt",
            "options": {"what": "translation", "language": "es"},
        },
    )
    assert b"Hola desde a." in r.data


@pytest.mark.parametrize(
    "stem", ["한국어 회의", "Привет", "Köln – Teil 1", 'say "hi"', "’quoted’ 🎉", "two\nlines"]
)
def test_a_name_in_any_script_downloads(client, db, stem):
    # The header was built by hand and had to fit Latin-1: a Korean name
    # raised while the headers were sent, and the request never answered
    from werkzeug.http import parse_options_header

    rel = f"talks/{stem}.mp3"
    with db.connection() as c:
        c.execute(
            "INSERT INTO media_files (rel_path, abs_path, status, is_video, created_at) VALUES (?, ?, 'done', 0, ?)",
            (rel, f"{ROOT}/{rel}", time.time()),
        )
    db.upsert_transcript(rel, _tx("Hello."), media_dir=ROOT, abs_path=f"{ROOT}/{rel}")
    r = client.post("/api/export", json={"files": [f"{ROOT}/{rel}"], "format": "txt"})
    assert r.status_code == 200 and b"Hello." in r.data
    r.headers["Content-Disposition"].encode("latin-1")  # a header a server can send
    kind, options = parse_options_header(r.headers["Content-Disposition"])
    assert kind == "attachment" and options["filename"].endswith(".txt")
    assert "\n" not in r.headers["Content-Disposition"]


@pytest.mark.parametrize(
    "body",
    [
        {"files": [ROOT + "/talks/a.mp3"], "options": ["names"]},
        {"files": [ROOT + "/talks/a.mp3"], "format": "video", "video": "mp4"},
        {"files": [ROOT + "/talks/a.mp3"], "format": "audio", "audio": 3},
        {"folders": ["/somewhere"]},
        {"folders": "talks"},
    ],
)
def test_a_body_of_the_wrong_shape_is_the_senders_mistake(client, body):
    r = client.post("/api/export", json=body)
    assert r.status_code == 400 and r.get_json()["error"]


def test_a_formula_in_a_transcript_stays_text_in_the_table(client, db):
    # A line that begins with = + - or @ opens as a live formula in a spreadsheet
    rel = "talks/formula.mp3"
    with db.connection() as c:
        c.execute(
            "INSERT INTO media_files (rel_path, abs_path, status, is_video, created_at) VALUES (?, ?, 'done', 0, ?)",
            (rel, f"{ROOT}/{rel}", time.time()),
        )
    tx = _tx('=HYPERLINK("http://example.invalid","click")')
    tx["segments"][1]["text"] = "-1+1 and @SUM(A1)"
    tx["speaker_names"] = {"SPEAKER_00": "=cmd"}
    db.upsert_transcript(rel, tx, media_dir=ROOT, abs_path=f"{ROOT}/{rel}")
    r = client.post(
        "/api/export",
        json={"files": [f"{ROOT}/{rel}"], "format": "csv", "options": {"names": True}},
    )
    import csv

    rows = list(csv.reader(io.StringIO(r.data.decode("utf-8"))))
    assert (
        r.status_code == 200
        and rows[1][3].startswith("'=HYPERLINK")
        and rows[2][3].startswith("'-1+1")
    )
    assert rows[1][2] == "'=cmd"
    for row in rows[1:]:
        assert not any(cell[:1] in "=+-@" for cell in row[2:]), row


def test_several_files_come_as_a_zip_with_their_folders(client):
    r = client.post(
        "/api/export",
        json={
            "folders": [{"path": ROOT + "/talks"}],
            "format": "md",
            "options": {"keep_folders": True},
        },
    )
    assert r.headers["Content-Type"] == "application/zip"
    assert sorted(zipfile.ZipFile(io.BytesIO(r.data)).namelist()) == ["a.md", "b.md", "sub/c.md"]
    assert r.headers["X-Export-Skipped"] == "1", "d.mp3 has no transcript"


def test_preview_and_bad_formats(client):
    d = client.post(
        "/api/export/preview", json={"files": [ROOT + "/talks/a.mp3"], "format": "pdf"}
    ).get_json()
    assert "Hello from a." in d["text"] and "PDF" in d["note"]
    assert (
        client.post(
            "/api/export/preview", json={"files": [ROOT + "/talks/a.mp3"], "format": "exe"}
        ).status_code
        == 400
    )
    assert client.post("/api/export", json={"files": [], "format": "txt"}).status_code == 400


def _wait(client, jid, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        j = next(j for j in client.get("/api/export/jobs").get_json()["jobs"] if j["id"] == jid)
        if j["status"] not in ("waiting", "running"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_big_text_exports_run_as_a_job(client, monkeypatch):
    from whisper_tm.api import export

    monkeypatch.setattr(export, "DIRECT_LIMIT", 2)
    r = client.post("/api/export", json={"folders": [{"path": ROOT}], "format": "txt"})
    assert r.status_code == 202
    j = _wait(client, r.get_json()["job"]["id"])
    assert j["status"] == "done" and j["done"] == 5 and j["skipped"] == [["d.mp3", "no transcript"]]
    dl = client.get(f"/api/export/jobs/{j['id']}/download")
    assert sorted(zipfile.ZipFile(io.BytesIO(dl.data)).namelist()) == [
        "a.txt",
        "b.txt",
        "c.txt",
        "e.txt",
    ]
    assert client.delete(f"/api/export/jobs/{j['id']}").status_code == 200
    assert client.get(f"/api/export/jobs/{j['id']}/download").status_code == 404


def test_a_job_can_be_cancelled(tmp_path, monkeypatch):
    from whisper_tm.core import export_jobs as jobs

    monkeypatch.setattr(jobs, "folder", lambda: tmp_path)
    gate = threading.Event()

    def work(item, out_dir, progress, should_stop):
        progress(0.5)
        gate.wait(5)
        p = out_dir / f"{item}.txt"
        p.write_text(item)
        return [p]

    job = jobs.start("Export · 3 files", ["one", "two", "three"], work)
    time.sleep(0.1)
    assert job.status == "running" and job.progress > 0
    assert jobs.cancel(job.id)
    gate.set()
    for _ in range(100):
        if job.status == "cancelled":
            break
        time.sleep(0.05)
    assert job.status == "cancelled" and not (tmp_path / job.id).exists()
    assert not jobs.cancel(job.id), "only waiting or running jobs"


def test_one_failing_file_is_skipped_not_fatal(tmp_path, monkeypatch):
    from whisper_tm.core import export_jobs as jobs

    monkeypatch.setattr(jobs, "folder", lambda: tmp_path)

    def work(item, out_dir, progress, should_stop):
        if item == "bad":
            raise ValueError("no video in this recording")
        p = out_dir / f"{item}.txt"
        p.write_text(item)
        return [p]

    job = jobs.start("x", ["good", "bad"], work)
    for _ in range(100):
        if job.status not in ("waiting", "running"):
            break
        time.sleep(0.05)
    assert job.status == "done" and job.result == "good.txt"
    assert job.skipped == [["bad", "no video in this recording"]]
    job = jobs.start("x", ["bad"], work)
    for _ in range(100):
        if job.status not in ("waiting", "running"):
            break
        time.sleep(0.05)
    assert job.status == "error" and "no video" in job.error


def test_the_pickers_list_and_library_relative_paths(client, monkeypatch):
    from whisper_tm.config import config

    monkeypatch.setattr(type(config), "media_dir", property(lambda self: Path(ROOT)), raising=False)
    d = client.post(
        "/api/export/files", json={"folders": [{"path": "talks"}], "q": "sub"}
    ).get_json()
    assert (
        d["total"] == 1
        and d["files"][0]["rel"] == "sub/c.mp3"
        and d["files"][0]["abs"] == ROOT + "/talks/sub/c.mp3"
    )
    r = client.post("/api/export", json={"files": ["talks/a.mp3"], "format": "txt"})
    assert r.status_code == 200 and b"Hello from a." in r.data
    assert client.post("/api/export/count", json={"files": ["../x.mp3"]}).status_code == 400


def test_a_videos_subtitles_follow_the_media_speakers(client):
    d = client.post(
        "/api/export/preview",
        json={
            "files": [ROOT + "/talks/b.mp4"],
            "format": "video",
            "options": {"speakers": ["Nobody"]},
            "video": {"speakers": []},
        },
    ).get_json()
    assert "Hello from b." in d["text"] and "Fine." in d["text"]
