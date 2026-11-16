"""The voices workspace: history of decisions, people, voice extractor.

The map and the guided assignment keep their routes in ``api/speaker.py``.
What is here came with the workspace's tabs:

* **History** — every decision on names and the trash, with undo and
  redo (``database/speaker.py``, "History of decisions").
* **People** — the named people with what is known about them; rename or
  merge one (a decision like any other, undoable).
* **Extract** — one person's voice as clips with their sentences
  (``core/voice_extract.py``).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from whisper_tm.config import config
from whisper_tm.core import paths, voice_extract
from whisper_tm.database.speaker import speaker_db

logger = logging.getLogger(__name__)

voices_bp = Blueprint("voices", __name__)


def _root() -> str:
    return str(config.media_dir)


def _display(abs_path: str, root: str) -> str:
    return paths.shown(abs_path, root) if root else abs_path


def _iso(at: str | None) -> str | None:
    """``datetime('now')`` text (UTC) → ISO with a ``Z`` — the page shows
    it in local time."""
    return f"{at[:10]}T{at[11:19]}Z" if at else None


def _entry(r: dict, root: str) -> dict:
    s = r.get("summary") or {}
    return {
        "id": r["id"],
        "at": _iso(r["at"]),
        "kind": r["kind"],
        "origin": r.get("origin") or "",
        "name": r.get("name"),
        "voices": r["voices"],
        "spread": r["spread"],
        "speakers": s.get("speakers", 0),
        "recordings": s.get("recordings", 0),
        "speech_hours": round((s.get("speech_sec") or 0.0) / 3600, 2),
        "samples": [{**x, "file": _display(x["abs_path"], root)} for x in s.get("samples", [])],
        "undone": bool(r.get("undone_by")),
        "undone_at": _iso(r.get("undone_at")),
        "reconstructed": bool(r.get("reconstructed")),
    }


# ── History ──────────────────────────────────────────────────────────────


@voices_bp.route("/api/voices/history")
def history():
    """Decisions, newest first. ``kind``: all, name, trash, undone;
    ``q`` finds a name; ``before`` continues a page."""
    db = speaker_db()
    try:
        limit = min(max(int(request.args.get("limit", 40)), 1), 200)
    except (TypeError, ValueError):
        limit = 40
    before = request.args.get("before", type=int)
    kind = request.args.get("kind", "all")
    rows, more = db.decisions(
        limit=limit, before=before, kind=kind, query=str(request.args.get("q") or "")
    )
    root = _root()
    return jsonify(
        {"entries": [_entry(r, root) for r in rows], "more": more, "counts": db.history_counts()}
    )


@voices_bp.route("/api/voices/history/<int:did>")
def history_entry(did):
    """One entry and what an undo (or redo) of it would do — changes
    nothing."""
    try:
        p = speaker_db().undo_preview(did)
    except KeyError:
        return jsonify({"error": "no such decision"}), 404
    return jsonify(
        {
            "entry": _entry(p["entry"], _root()),
            "redo": p["redo"],
            "moves": p["moves"],
            "left": p["left"],
            "later_same_name": [{**x, "at": _iso(x["at"])} for x in p["later_same_name"]],
        }
    )


@voices_bp.route("/api/voices/history/<int:did>/voices")
def history_voices(did):
    """The ids of the voices an entry changed — to light them on the map."""
    ids = speaker_db().decision_voice_ids(did)
    if ids is None:
        return jsonify({"error": "no such decision"}), 404
    return jsonify({"ids": ids})


@voices_bp.route("/api/voices/history/<int:did>/undo", methods=["POST"])
def history_undo(did):
    """Undo one decision. The answer carries the voices it made open again
    as a question — to decide them anew right away."""
    return _revert(did, redo=False)


@voices_bp.route("/api/voices/history/<int:did>/redo", methods=["POST"])
def history_redo(did):
    """Put an undone decision back in force — for the voices no later
    decision changed since."""
    return _revert(did, redo=True)


def _revert(did: int, redo: bool):
    db = speaker_db()
    try:
        result = db.redo(did) if redo else db.undo(did)
    except KeyError:
        return jsonify({"error": "no such decision"}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 409
    if result.get("trash_changed"):
        # Voices back from the trash are missing from the held grouping of
        # the guided assignment until it is computed again.
        from whisper_tm.core import speaker_assignment

        speaker_assignment.INVENTORY.refresh_soon()
    root = _root()
    out = {
        "ok": True,
        "changed": result["changed"],
        "left": result["left"],
        "entry": _entry(result["entry"], root),
        # The Assign page's bar moves with an undo too (it used to stand
        # still). Only from a grouping already held:
        # an undo must not wait for one to be computed.
        "progress": _progress_if_held(),
    }
    if not redo:
        out["ask"] = _question(db, result["direct"], result["entry"], root)
    return jsonify(out)


def _progress_if_held() -> dict | None:
    from whisper_tm.core import speaker_assignment

    try:
        state_info = speaker_assignment.INVENTORY.held()
        return speaker_assignment.progress(state_info) if state_info else None
    except Exception:  # noqa: BLE001 — the undo itself is done
        logger.debug("Progress after an undo failed", exc_info=True)
        return None


def _question(db, ids, entry: dict, root: str) -> dict | None:
    """The voices an undo made open, in the shape of a suggestion of the
    guided assignment — ``vector_ids``, speakers, samples."""
    open_ids = db.open_voices(ids)
    if not open_ids:
        return None
    speakers = db.speakers_of(open_ids)
    return {
        "vector_ids": open_ids,
        "speaker": len(speakers),
        "recordings": len({s["abs_path"] for s in speakers}),
        "speech_hours": round(sum(s["speech_sec"] for s in speakers) / 3600, 2),
        "samples": [
            {
                "vector_id": s["vector_id"],
                "file": _display(s["abs_path"], root),
                "label": s["label"],
                "speech_sec": s["speech_sec"],
            }
            for s in speakers[:8]
        ],
        "decided": {
            "id": entry["id"],
            "kind": entry["kind"],
            "name": entry.get("name"),
            "at": _iso(entry["at"]),
        },
    }


# ── People ───────────────────────────────────────────────────────────────


@voices_bp.route("/api/voices/people")
def people():
    """The named people with their voices, speech, recordings and dates."""
    root = _root()
    rows = speaker_db().people_stats()
    for r in rows:
        r["first_named"] = _iso(r["first_named"])
        r["last_named"] = _iso(r["last_named"])
        r["samples"] = [{**x, "file": _display(x["abs_path"], root)} for x in r["samples"]]
    return jsonify({"people": rows})


@voices_bp.route("/api/voices/review")
def review():
    """Every voice of one person, to listen through and check: ``person``,
    filters (``source``, ``closer``, ``fit_max``/``fit_min``, ``date_from``/
    ``date_to``, ``folder``, ``file``, ``label``, ``min_speech``/
    ``max_speech``, ``spans``, ``per_recording``), ``sort`` (fit,
    date_desc, date_asc, recording, speech_desc), ``offset``, ``limit``."""
    from whisper_tm.core import voice_review

    person = str(request.args.get("person") or "").strip()
    if not person:
        return jsonify({"error": "no person"}), 400
    try:
        return jsonify(
            voice_review.page(
                person,
                request.args.to_dict(),
                request.args.get("offset", 0, type=int),
                request.args.get("limit", 100, type=int),
            )
        )
    except KeyError:
        return jsonify({"error": f"nobody is named {person}"}), 404


@voices_bp.route("/api/voices/review/decide", methods=["POST"])
def review_decide():
    """Act on voices of the review list — ``person``, ``ids``, ``action``
    (unname, trash, name) and for ``name`` the new name ``to``. One entry
    of the history; only voices still named as ``person`` are touched."""
    from whisper_tm.core import voice_review

    d = request.get_json(silent=True) or {}
    ids = d.get("ids") or []
    if not isinstance(ids, list) or not ids:
        return jsonify({"error": "no voices"}), 400
    try:
        r = voice_review.decide(
            str(d.get("person") or ""), ids, str(d.get("action") or ""), str(d.get("to") or "")
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, **r})


@voices_bp.route("/api/voices/people/rename", methods=["POST"])
def people_rename():
    """All voices of one person to another name — merges into an existing
    one. One decision in the history, so it can be undone."""
    d = request.get_json(silent=True) or {}
    old, new = str(d.get("from") or "").strip(), str(d.get("to") or "").strip()
    if not old or not new:
        return jsonify({"error": "both names are needed"}), 400
    db = speaker_db()
    try:
        r = db.rename_person(old, new)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, **r, "decision": db.last_decision_id()})


# ── Extract ──────────────────────────────────────────────────────────────


@voices_bp.route("/api/voices/extract/options")
def extract_options():
    """The filters with their defaults, and what an export can write."""
    return jsonify(
        {
            "defaults": voice_extract.DEFAULTS,
            "formats": list(voice_extract.FORMATS),
            "rates": list(voice_extract.RATES),
            "folder": str(voice_extract.exports_dir()),
        }
    )


@voices_bp.route("/api/voices/extract/preview", methods=["POST"])
def extract_preview():
    """Choose voices, sentences and clips — held for the export."""
    d = request.get_json(silent=True) or {}
    try:
        run = voice_extract.build_run(
            str(d.get("person") or ""), d.get("options") or {}, root=_root()
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa — reported on the page
        logger.exception("Voice extract preview failed")
        return jsonify({"error": str(e)[:300]}), 500
    return jsonify(voice_extract.run_view(run))


def _reachable_clip(run_id: str, i: int):
    """The run and the clip — or an answer saying why not."""
    from whisper_tm.core.security import root_allowed

    run = voice_extract.get_run(run_id)
    if run is None or not 0 <= i < len(run["clips"]):
        return None, (jsonify({"error": "the preview is gone — compute it again"}), 404)
    c = run["clips"][i]
    if not root_allowed(c["abs_path"]) or not Path(c["abs_path"]).exists():
        return None, (jsonify({"error": "recording unreachable"}), 404)
    return run, None


@voices_bp.route("/api/voices/extract/clip/<run_id>/<int:i>")
def extract_clip(run_id, i):
    """One clip of a preview as MP3 — cut on demand from the recording, as it
    stands with the hand edits. ``?format=wav`` downloads it as WAV."""
    run, err = _reachable_clip(run_id, i)
    if err:
        return err
    c = run["clips"][i]
    if request.args.get("format") == "wav":
        try:
            data = voice_extract.clip_wav(run, i)
        except Exception as e:
            return jsonify({"error": str(e)[:200]}), 500
        name = f"{voice_extract.slug(run['person'])}_{i + 1:05d}.wav"
        return send_file(
            io.BytesIO(data), mimetype="audio/wav", as_attachment=True, download_name=name
        )
    cache = Path(config.dirs.get("cache") or config.dirs["work"]) / "voice_samples"
    cache.mkdir(parents=True, exist_ok=True)
    # The bounds belong in the name: a trimmed clip is a different cut
    target = cache / f"x{run_id}_{i}_{int(c['start'] * 1000)}_{int(c['end'] * 1000)}.mp3"
    if not target.exists():
        try:
            voice_extract.cut_clip(run, i, target, cache)
        except Exception as e:
            return jsonify({"error": str(e)[:200]}), 500
    return send_file(str(target), mimetype="audio/mpeg", conditional=True)


@voices_bp.route("/api/voices/extract/context/<run_id>/<int:i>")
def extract_context(run_id, i):
    """The clip with a few seconds around it as WAV, for the clip editor —
    it holds this in memory to draw, play and trim. ``X-Clip-From`` says
    where in the recording it starts."""
    run, err = _reachable_clip(run_id, i)
    if err:
        return err
    try:
        pad = float(request.args.get("pad", 2.5))
    except (TypeError, ValueError):
        pad = 2.5
    try:
        data, start = voice_extract.context_wav(run, i, pad=pad)
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500
    r = send_file(io.BytesIO(data), mimetype="audio/wav")
    r.headers["X-Clip-From"] = f"{start:.3f}"
    r.headers["Cache-Control"] = "no-store"
    return r


@voices_bp.route("/api/voices/extract/clips/<run_id>")
def extract_clips(run_id):
    """A page of a preview's clips. ``state``: all, open, edited, kept,
    dropped; ``offset`` and ``limit`` page through them."""
    run = voice_extract.get_run(run_id)
    if run is None:
        return jsonify({"error": "the preview is gone — compute it again"}), 404
    try:
        offset = int(request.args.get("offset", 0))
        limit = int(request.args.get("limit", 60))
    except (TypeError, ValueError):
        offset, limit = 0, 60
    return jsonify(
        voice_extract.clips_page(run, offset, limit, str(request.args.get("state") or "all"))
    )


@voices_bp.route("/api/voices/extract/edit/<run_id>/<int:i>", methods=["POST"])
def extract_edit(run_id, i):
    """Change a clip by hand: ``start``/``end`` (seconds in the recording),
    ``text``, ``state`` (kept, dropped or empty) — or ``reset``. Kept in
    memory for the person, so the next preview has it too."""
    run = voice_extract.get_run(run_id)
    if run is None:
        return jsonify({"error": "the preview is gone — compute it again"}), 404
    try:
        done = voice_extract.edit_clip(run, i, request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, **done})


@voices_bp.route("/api/voices/extract/forget", methods=["POST"])
def extract_forget():
    """Forget the hand edits of one person (body: ``person``) — every clip
    back to its sentence, nothing kept or dropped."""
    person = str((request.get_json(silent=True) or {}).get("person") or "").strip()
    if not person:
        return jsonify({"error": "choose a person"}), 400
    return jsonify({"ok": True, "forgotten": voice_extract.forget_edits(person)})


@voices_bp.route("/api/voices/extract/dest")
def extract_dest():
    """Check a folder an export should be saved into (``?path=``): whether
    it can be used — it exists, lies in the allowed folders and can be
    written — and how much space is free there. Empty: the exports folder."""
    return jsonify(voice_extract.dest_info(request.args.get("path", "")))


@voices_bp.route("/api/voices/extract/export", methods=["POST"])
def extract_export():
    """Start the export of a preview in the background. Body: ``run_id``,
    the export options and, to save it into a folder of your own instead
    of the exports folder, ``dest`` — with ``unpacked`` for the dataset as
    files instead of a ZIP."""
    d = request.get_json(silent=True) or {}
    try:
        r = voice_extract.start_export(str(d.get("run_id") or ""), d)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    return jsonify(r)


@voices_bp.route("/api/voices/extract/status")
def extract_status():
    """The running (or last) export — progress, result, error — and the
    exports in the folder."""
    return jsonify(voice_extract.export_status())


@voices_bp.route("/api/voices/extract/stop", methods=["POST"])
def extract_stop():
    """Stop the running export. What it wrote so far is removed."""
    return jsonify({"ok": True, "was_running": voice_extract.stop_export()})


@voices_bp.route("/api/voices/extract/download/<path:name>")
def extract_download(name):
    """Download one finished export (a file in the exports folder)."""
    from whisper_tm.core.security import safe_path

    p = safe_path(voice_extract.exports_dir(), name)
    if not p.is_file() or ".part" in p.name:
        return jsonify({"error": "no such export"}), 404
    return send_file(str(p), as_attachment=True, download_name=p.name)


@voices_bp.route("/api/voices/extract/delete", methods=["POST"])
def extract_delete():
    """Delete one export from the exports folder. Body: ``name``."""
    d = request.get_json(silent=True) or {}
    if not voice_extract.delete_export(str(d.get("name") or "")):
        return jsonify({"error": "no such export"}), 404
    return jsonify({"ok": True, "files": voice_extract.list_exports()})
