"""Chunked upload: the last chunk assembles the file — safely.

Before, the name was chosen (``while final_path.exists()``) and the
file assembled OUTSIDE the lock. Two uploads of the same name finishing at
once both picked ``talk.mp3``, both opened it with "wb" — one upload was
lost, both answers said "complete". And the file was assembled in place:
a scan seeing it halfway took a truncated recording for a new one.
"""

from __future__ import annotations

import io
import threading

import pytest
from flask import Flask

from whisper_tm.api import media

#: The content check itself — the fixture below puts a stand-in in its place
REAL_CHECK = media._audio_or_video


@pytest.fixture
def app(tmp_path, monkeypatch):
    from whisper_tm.config import config

    up = tmp_path / "media" / "uploads"
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    monkeypatch.setattr(type(config), "upload_dir", property(lambda self: up))
    monkeypatch.setattr(type(config), "dirs", property(lambda self: {"chunks": chunks}))
    from whisper_tm import scanner

    monkeypatch.setattr(scanner, "register_single_file", lambda p: None)
    # These tests are about putting the parts together; the content check
    # has tests of its own below
    monkeypatch.setattr(media, "_audio_or_video", lambda p: True)
    a = Flask(__name__)
    a.chunks = chunks
    a.register_blueprint(media.media_bp)
    a.up = up
    return a


def _post(client, file_id, data, name="talk.mp3", ci=0, tc=1):
    return client.post(
        "/api/upload_chunk",
        data={
            "fileId": file_id,
            "chunkIndex": str(ci),
            "totalChunks": str(tc),
            "fileName": name,
            "chunk": (io.BytesIO(data), "blob"),
        },
        content_type="multipart/form-data",
    )


def test_two_reservations_of_one_name_differ(tmp_path):
    a = media._reserve_upload_name(tmp_path, "talk.mp3")
    b = media._reserve_upload_name(tmp_path, "talk.mp3")
    try:
        assert a != b and a.name == "talk.mp3" and b.name == "talk_1.mp3"
    finally:
        media._release_upload_name(a)
        media._release_upload_name(b)


def test_concurrent_uploads_of_one_name_keep_both(app, monkeypatch):
    # Hold every assembly at the point where both have chosen a name
    barrier = threading.Barrier(2, timeout=5)
    real = media._reserve_upload_name

    def reserve(*a, **k):
        p = real(*a, **k)
        barrier.wait()
        return p

    monkeypatch.setattr(media, "_reserve_upload_name", reserve)
    results = {}

    def upload(fid, payload):
        with app.test_client() as c:
            results[fid] = _post(c, fid, payload).get_json()

    t1 = threading.Thread(target=upload, args=("one", b"A" * 5000))
    t2 = threading.Thread(target=upload, args=("two", b"B" * 7000))
    t1.start()
    t2.start()
    t1.join(10)
    t2.join(10)
    files = {p.name: p.read_bytes() for p in app.up.iterdir()}
    assert sorted(files) == ["talk.mp3", "talk_1.mp3"]
    assert sorted(len(v) for v in files.values()) == [5000, 7000]


def test_the_file_appears_only_when_complete(app):
    c = app.test_client()
    assert _post(c, "f", b"x" * 100, ci=0, tc=2).get_json()["status"] == "uploading"
    r = _post(c, "f", b"y" * 100, ci=1, tc=2).get_json()
    assert r["status"] == "complete"
    assert [p.name for p in app.up.iterdir()] == ["talk.mp3"]
    assert (app.up / "talk.mp3").read_bytes() == b"x" * 100 + b"y" * 100


# ── Audio and video only ──────────────────────────────────────────────────
#
# The upload folder can be pointed at any folder the app may write (the
# settings page): a .pth put into site-packages, a .desktop into autostart or
# an authorized_keys into ~/.ssh ran code. Only the built-in media types
# are taken — the settings' extra extensions don't widen it.


@pytest.mark.parametrize(
    "name", ["boot.pth", "evil.desktop", "authorized_keys", "page.html", "x.svg", "run.sh"]
)
def test_only_audio_and_video_are_taken(app, name):
    r = _post(app.test_client(), "f1", b"payload", name=name)
    assert r.status_code == 400 and "audio and video" in r.get_json()["error"]
    assert not app.up.exists() or not any(app.up.iterdir())


def test_an_extra_extension_of_the_settings_does_not_open_it(app, monkeypatch):
    from whisper_tm.config import config

    real = config.get
    monkeypatch.setattr(
        config,
        "get",
        lambda s, k=None, default=None: (
            [".pth"]
            if (s, k) == ("scanner", "extra_audio_extensions")
            else real(s, k, default=default)
        ),
    )
    r = _post(app.test_client(), "f2", b"payload", name="boot.pth")
    assert r.status_code == 400


def test_a_recording_still_goes_through(app):
    r = _post(app.test_client(), "f3", b"RIFF....WAVE", name="interview.wav")
    assert r.status_code == 200 and (app.up / "interview.wav").exists()


# ── The content too, not just the name ────────────────────────────────────
#
# The extension check let a key or a config in as "key.mp4" — and the app
# serves what lies in the library. ffprobe must find audio or video in it.


def test_a_file_that_only_calls_itself_mp4_is_refused(app, monkeypatch):
    import shutil

    if not shutil.which("ffprobe"):
        pytest.skip("no ffprobe")
    monkeypatch.setattr(media, "_audio_or_video", REAL_CHECK)
    r = _post(
        app.test_client(), "k1", b"-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n", name="key.mp4"
    )
    assert r.status_code == 400 and "holds neither" in r.get_json()["error"]
    assert not app.up.exists() or not any(app.up.iterdir())
    assert not any(app.chunks.iterdir())  # its chunks are gone too


def test_real_sound_passes_the_check(app, monkeypatch, tmp_path):
    import shutil
    import subprocess

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(media, "_audio_or_video", REAL_CHECK)
    wav = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(wav)],
        check=True,
    )
    r = _post(app.test_client(), "t1", wav.read_bytes(), name="tone.wav")
    assert r.status_code == 200 and (app.up / "tone.wav").exists()


# A manifest or playlist has "streams" too — other files': a DASH manifest
# named talk.mp4 made every later read decode the file it names, anywhere
# the server may read. Only real containers are recordings.


def _tone(folder, name="tone.m4a"):
    import subprocess

    out = folder / name
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            str(out),
        ],
        check=True,
    )
    return out


def _manifests(folder):
    """Files that are not recordings but point at one."""
    other = folder / "elsewhere"
    other.mkdir()
    tone = _tone(other)
    return {
        "dash.mp4": (
            '<?xml version="1.0"?>\n<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static" '
            'mediaPresentationDuration="PT1S" profiles="urn:mpeg:dash:profile:isoff-on-demand:2011">'
            f'<Period><AdaptationSet mimeType="audio/mp4"><Representation id="a" bandwidth="64000" '
            f'codecs="mp4a.40.2"><BaseURL>{tone}</BaseURL></Representation></AdaptationSet></Period></MPD>\n'
        ),
        "hls.mp4": f"#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n#EXTINF:1.0,\n{tone}\n#EXT-X-ENDLIST\n",
        "list.mp3": f"ffconcat version 1.0\nfile '{tone}'\n",
        "session.mp4": "v=0\no=- 0 0 IN IP4 127.0.0.1\ns=x\nc=IN IP4 127.0.0.1\nt=0 0\nm=audio 5004 RTP/AVP 0\n",
    }


@pytest.mark.parametrize("name", ["dash.mp4", "hls.mp4", "list.mp3", "session.mp4"])
def test_a_manifest_named_as_a_recording_is_refused(app, monkeypatch, tmp_path, name):
    import shutil

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(media, "_audio_or_video", REAL_CHECK)
    r = _post(app.test_client(), "m-" + name[:3], _manifests(tmp_path)[name].encode(), name=name)
    assert r.status_code == 400 and "holds neither" in r.get_json()["error"]
    assert not app.up.exists() or not any(app.up.iterdir())


def test_a_picture_named_mp4_is_refused(app, monkeypatch, tmp_path):
    import shutil
    import subprocess

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("no ffmpeg")
    monkeypatch.setattr(media, "_audio_or_video", REAL_CHECK)
    png = tmp_path / "p.png"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=32x32:d=1",
            "-frames:v",
            "1",
            str(png),
        ],
        check=True,
    )
    r = _post(app.test_client(), "p1", png.read_bytes(), name="holiday.mp4")
    assert r.status_code == 400


#: Every extension an upload may have, with what ffmpeg needs to make one
#: (None: this ffmpeg may have no encoder for it — then it is skipped)
_MAKE = {
    ".mp4": [],
    ".mkv": [],
    ".avi": [],
    ".mov": [],
    ".webm": [],
    ".ts": [],
    ".mts": ["-f", "mpegts"],
    ".m2ts": ["-f", "mpegts"],
    ".mpg": [],
    ".mpeg": [],
    ".wmv": [],
    ".flv": ["-ar", "44100"],
    ".m4v": [],
    ".ogv": [],
    ".3gp": ["-ar", "8000", "-ac", "1"],
    ".mp3": [],
    ".wav": [],
    ".flac": [],
    ".ogg": [],
    ".m4a": [],
    ".wma": [],
    ".aac": [],
    ".opus": [],
    ".3gpp": ["-f", "3gp", "-ar", "8000", "-ac", "1"],
    ".amr": ["-ar", "8000", "-ac", "1"],
    ".awb": ["-f", "amr", "-ar", "16000", "-ac", "1"],
    ".spx": [],
    ".caf": [],
    ".au": [],
    ".mka": [],
    ".ra": ["-f", "rm", "-ar", "8000", "-ac", "1"],
    ".gsm": ["-ar", "8000", "-ac", "1"],
    ".oga": [],
    ".snd": ["-f", "au"],
    ".voc": [],
}


def test_every_upload_extension_has_a_recipe():
    from whisper_tm.config import MEDIA_EXTENSIONS

    assert set(_MAKE) == set(MEDIA_EXTENSIONS)


@pytest.mark.parametrize("ext", sorted(_MAKE))
def test_a_real_recording_of_every_extension_passes(tmp_path, ext):
    """The container list must not refuse what the extension list lets in."""
    import shutil
    import subprocess

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("no ffmpeg")
    out = tmp_path / f"clip{ext}"
    made = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            *_MAKE[ext],
            "-strict",
            "-2",
            str(out),
        ],
        capture_output=True,
    )
    if made.returncode != 0 or not out.exists() or not out.stat().st_size:
        pytest.skip(f"this ffmpeg can't write {ext}")
    assert REAL_CHECK(out)


# ── The name: any script, nothing that leaves the folder ──────────────────
#
# secure_filename kept ASCII only: a Cyrillic name shrank to "mp3" and was
# refused "without an extension", an accented one lost its letters.


@pytest.mark.parametrize(
    "name",
    [
        "Привет.mp3",
        "日本語の会議.mp4",
        "한국어 회의.mp3",
        "🎉 party.mp3",
        "Grüße aus Köln – Teil 1.mp3",
        "ناقش.wav",
        "Don't stop.mp4",
    ],
)
def test_a_name_in_any_script_is_kept(app, name):
    r = _post(app.test_client(), "n1", b"RIFF....WAVE", name=name)
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["display_name"] == name and (app.up / name).exists()


@pytest.mark.parametrize(
    "sent, saved",
    [
        ("../../etc/cron.d/job.mp3", "job.mp3"),  # no way out of the folder
        ("C:\\Users\\me\\talk.mp4", "talk.mp4"),  # a Windows path
        (".hidden.mp3", "hidden.mp3"),  # not a hidden file
        ("what?*:|.wav", "what____.wav"),  # a share refuses these
        ("a\x00b\nc.mp3", "a_b_c.mp3"),  # control characters
        ("con.mp3", "_con.mp3"),  # a Windows device name
        ("trailing. .mp3 ", "trailing.mp3"),
        ("e\u0301te\u0301.mp3", "été.mp3"),  # one form of a letter (NFC)
    ],
)
def test_what_a_file_system_cannot_take_is_removed(sent, saved):
    assert media.upload_name(sent) == saved


def test_a_name_without_letters_left_still_uploads():
    assert media.upload_name("???.mp3") == "upload.mp3"
    assert media.upload_name("") == "upload"


@pytest.mark.parametrize("stem", ["x" * 400, "ä" * 300, "語" * 200, "🎉" * 120])
def test_a_long_name_leaves_room_for_the_transcript_beside_it(stem):
    name = media.upload_name(stem + ".flac")
    assert name.endswith(".flac") and len(name.encode("utf-8")) <= 240
    name.encode("utf-8").decode("utf-8")  # cut between characters, not inside one


def test_an_unwritable_upload_folder_says_why(app):
    import os

    if os.geteuid() == 0:
        pytest.skip("root writes everywhere")
    app.up.parent.mkdir(parents=True, exist_ok=True)
    app.up.parent.chmod(0o500)
    try:
        r = _post(app.test_client(), "u1", b"x" * 10, name="talk.mp3")
    finally:
        app.up.parent.chmod(0o700)
    assert r.status_code == 500 and "can't be written" in r.get_json()["error"]
    assert not any(app.chunks.iterdir())  # nothing left behind


def test_old_chunks_are_removed_at_start(app):
    import os
    import time

    old, new = app.chunks / "abandoned", app.chunks / "in-progress"
    for d in (old, new):
        d.mkdir()
        (d / "chunk_000000").write_bytes(b"x")
    long_ago = time.time() - 2 * 86400
    os.utime(old, (long_ago, long_ago))
    assert media.remove_stale_chunks() == 1
    assert not old.exists() and new.exists()


def test_an_upload_larger_than_the_cap_is_refused(app, monkeypatch):
    """Each chunk was capped by the request size; the whole file was not."""
    from whisper_tm.config import config

    real = config.get
    monkeypatch.setattr(
        config,
        "get",
        lambda s, k=None, default=None: (
            1e-6 if (s, k) == ("server", "max_upload_gb") else real(s, k, default=default)
        ),
    )
    c = app.test_client()
    assert _post(c, "big", b"x" * 600, ci=0, tc=3).get_json()["status"] == "uploading"
    r = _post(c, "big", b"x" * 600, ci=1, tc=3)  # 1,200 bytes > 1,000
    assert r.status_code == 413 and "max_upload_gb" in r.get_json()["error"]
    assert not any(app.chunks.iterdir())  # its chunks are gone


def test_an_upload_stops_before_the_disk_is_full(app, monkeypatch):
    import collections

    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(media.shutil, "disk_usage", lambda p: usage(10**12, 10**12 - 10**8, 10**8))
    r = _post(app.test_client(), "full", b"x" * 100)
    assert r.status_code == 507 and "nearly full" in r.get_json()["error"]
