"""The embedding service of "Images & sounds" — runs in its own environment.

core/media_server.py starts it with the Python of the media environment
(media/environment.py): EmbeddingGemma 2 needs transformers >= 5.19 and
sentence-transformers >= 6.1, newer than the app's own environment, which
must not move. So this file imports nothing from the app and talks HTTP on
the loopback address only.

**Two models, two jobs.** Queries run on the CPU in float32 — a text in
~0.1 s, an image in ~3.5 s, a minute of sound in ~1.3 s — so a search never
waits for the GPU and never takes memory from the queue. Indexing runs on
the GPU in bfloat16 (float16 returns NaN with this model), loaded when the
indexer has a turn at the GPU gate and released when it gives the card
back. The text tower alone serves text queries (~1 GB); the image and sound
towers load with the first file query.

**Indexing a chunk of moments** (``POST /index``) — the service decodes the
media itself (single-frame seeks, one ffmpeg thread each: decoding whole
videos, or re-encoding preview clips, drove the system load far too high
next to the queue):

1. a still per moment (also when stills aren't stored: it decides what is
   new);
2. a moment whose still looks like the last kept one (cosine ≥ ``dedupe``)
   is a repeat — a talking head over hours — and gets no clip;
3. the clip: ``clip_frames`` frames spread over ``clip_sec`` seconds (4
   frames: 5.5 clips/s on a 12 GB card, cosine 0.98 to 16 frames; 8
   frames: 0.99 at twice the time), and with ``videoaudio`` the same frames
   with their sound as one interleaved input;
4. the sound window of every moment, repeat or not;
5. what is said there, as text (``speech``).

Endpoints::

    GET  /health
    POST /embed       {"texts": [...], "prompt": "SearchQuery"}
    POST /embed-file  {"path", "kind": "image|audio|video", "text", "with_sound", "frames"}
    POST /index       see :func:`index_chunk`
    POST /release     {"device": "cuda"}

Vectors travel as base64 of little-endian float16, unit length, 768 numbers.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

logger = logging.getLogger("media-service")

SR = 16000
#: What a frame is scaled to before the model sees it (it resizes anyway)
FRAME_HEIGHT = 360
STILL_WIDTH = 640
#: Longest sound and video a file query reads
QUERY_SOUND_SEC = 60.0
QUERY_VIDEO_SEC = 60.0

_args = None
_models: dict[tuple[str, str], object] = {}
_model_locks = {"cpu": threading.Lock(), "cuda": threading.Lock()}
_load_lock = threading.Lock()
_started = time.time()


# ── Models ──────────────────────────────────────────────────────────────────


def _torch():
    import torch

    return torch


def _load(device: str, towers: str):
    """The model on ``device`` with ``towers`` ("text" or "all") — kept.

    "all" on the CPU replaces a text-only copy (one copy per device)."""
    key = (device, towers)
    with _load_lock:
        if key in _models:
            return _models[key]
        if device == "cpu" and towers == "text" and ("cpu", "all") in _models:
            return _models[("cpu", "all")]
        torch = _torch()
        from sentence_transformers import SentenceTransformer

        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            dtype = torch.float32
        kwargs = {} if towers == "all" else {"vision_config": None, "audio_config": None}
        t0 = time.time()
        model = SentenceTransformer(
            _args.model, device=device, model_kwargs={"torch_dtype": dtype}, config_kwargs=kwargs
        )
        if device == "cpu" and towers == "all":
            _models.pop(("cpu", "text"), None)
        _models[key] = model
        logger.info(
            "Loaded %s (%s towers) on %s in %.1f s", _args.model, towers, device, time.time() - t0
        )
        return model


def release(device: str) -> bool:
    with _load_lock:
        hit = [k for k in _models if k[0] == device]
        for k in hit:
            del _models[k]
    if hit and device == "cuda":
        torch = _torch()
        torch.cuda.empty_cache()
        logger.info("GPU model released")
    return bool(hit)


def _encode(model, inputs: list, batch: int, prompt: str | None = None) -> np.ndarray:
    kw = {"prompt_name": prompt} if prompt else {}
    out = model.encode(
        inputs,
        batch_size=max(1, batch),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        **kw,
    )
    out = np.asarray(out, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("the model returned NaN or inf")
    return out


def b64(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype="<f2").tobytes()).decode("ascii")


def unb64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f2").astype(np.float32)


# ── Decoding (ffmpeg, single seeks, one thread each) ────────────────────────


#: The containers a source may come in — the same list as the app's
#: core/media_formats.WHITELIST (this file runs in its own environment and
#: imports nothing of the app; a test keeps the two equal)
SOURCE_FORMATS = (
    "3g2,3gp,aac,ac3,aiff,amr,amrnb,amrwb,ape,asf,au,avi,caf,dsf,dts,dtshd,dv,eac3,"
    "flac,flv,gsm,gxf,h264,hevc,ivf,live_flv,loas,m4a,m4v,matroska,mj2,mlp,mov,mp3,"
    "mp4,mpc,mpc8,mpeg,mpegts,mpegvideo,mxf,nut,ogg,oma,rm,tak,truehd,tta,voc,w64,wav,"
    "webm,wtv,wv"
)


def _ffmpeg(args: list, timeout: float = 120) -> bytes:
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", *args],
        capture_output=True,
        timeout=timeout,
    )
    return r.stdout


def grab_frame(path: str, t: float, scale: str):
    from PIL import Image

    png = _ffmpeg(
        [
            "-ss",
            f"{max(0.0, t):.3f}",
            "-format_whitelist",
            SOURCE_FORMATS,
            "-i",
            path,
            "-frames:v",
            "1",
            "-vf",
            scale,
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
    )
    if not png:
        return None
    try:
        img = Image.open(io.BytesIO(png)).convert("RGB")
    except Exception:  # noqa: BLE001 — a broken frame is a missing frame
        return None
    return img if np.asarray(img).std() > 0 else None


def grab_frames(path: str, start: float, dur: float, n: int) -> list:
    """``n`` frames spread evenly over ``dur`` seconds from ``start``."""
    scale = f"scale=-2:{FRAME_HEIGHT}"
    frames = [grab_frame(path, start + dur * (k + 0.5) / n, scale) for k in range(max(1, n))]
    return [f for f in frames if f is not None]


def grab_sound(path: str, start: float, dur: float) -> np.ndarray | None:
    pcm = _ffmpeg(
        [
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{dur:.3f}",
            "-format_whitelist",
            SOURCE_FORMATS,
            "-i",
            path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-f",
            "s16le",
            "-",
        ]
    )
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return a if a.size >= SR // 2 else None


def frames_of_file(path: str, n: int, limit: float = QUERY_VIDEO_SEC) -> list:
    """``n`` frames spread over the first ``limit`` seconds of a video file."""
    import av

    with av.open(path) as c:
        stream = c.streams.video[0]
        dur = (
            float(stream.duration * stream.time_base)
            if stream.duration
            else float(c.duration or 0) / 1e6
        )
    span = min(dur or limit, limit)
    return grab_frames(path, 0.0, span, n)


# ── Queries ─────────────────────────────────────────────────────────────────


def embed_texts(texts: list, prompt: str) -> np.ndarray:
    model = _load("cpu", "text")
    with _model_locks["cpu"]:
        return _encode(model, texts, 16, prompt)


def embed_file(
    path: str, kind: str, text: str = "", with_sound: bool = False, frames: int = 4
) -> np.ndarray:
    """A file as the query — optionally with a text, one interleaved input."""
    if kind == "image":
        from PIL import Image

        media, slot = {"image": Image.open(path).convert("RGB")}, "<|image|>"
    elif kind == "audio":
        sound = grab_sound(path, 0.0, QUERY_SOUND_SEC)
        if sound is None:
            raise ValueError("no sound in this file")
        media, slot = {"audio": {"array": sound, "sampling_rate": SR}}, "<|audio|>"
    elif kind == "video":
        pics = frames_of_file(path, frames)
        if not pics:
            raise ValueError("no pictures in this video")
        media, slot = {"video": pics}, "<|video|>"
        if with_sound:
            sound = grab_sound(path, 0.0, QUERY_VIDEO_SEC)
            if sound is not None:
                media["audio"] = {"array": sound, "sampling_rate": SR}
                slot += " <|audio|>"
    else:
        raise ValueError(f"unknown kind {kind!r}")
    if text or len(media) > 1:
        inp = {"text": f"{text} {slot}".strip(), **media}
    elif kind == "video":
        inp = media  # a bare list of frames would read as several images
    else:
        inp = next(iter(media.values()))
    model = _load("cpu", "all")
    with _model_locks["cpu"]:
        return _encode(model, [inp], 1)[0]


# ── Indexing ────────────────────────────────────────────────────────────────


def index_chunk(req: dict) -> dict:
    """Embed a chunk of moments of one file.

    ``req``: ``path``; ``video`` (bool); ``moments`` — ``[{"t", "start",
    "dur", "text", "thumb"}]`` (``thumb``: where its still goes as JPEG, or
    none); ``kinds`` — of image, video, videoaudio, audio, speech;
    ``clip_sec``, ``clip_frames``, ``dedupe`` (0 = keep all), ``prev`` (the
    file's last kept still, base64, or none), ``device``, ``batch``,
    ``workers``, ``thumb_width``; ``thumbs`` (false: write no thumbnails —
    the folder is full, they are cut when shown); ``repeats`` — which
    moments repeat, when an earlier pass over the file already looked
    (kinds added later: no look at the stills again unless stills are
    wanted).

    Returns ``{"moments": [{"repeat": bool, "still": bool, "vectors":
    {kind: base64}}], "prev": base64|None, "thumb_bytes": int, "timing":
    {...}}`` — ``still``: the moment kept a still of its own.
    """
    path = req["path"]
    moments = req.get("moments") or []
    kinds = set(req.get("kinds") or [])
    video = bool(req.get("video"))
    device = req.get("device") or "cuda"
    batch = int(req.get("batch") or 4)
    workers = max(1, int(req.get("workers") or 4))
    clip_sec = float(req.get("clip_sec") or 10.0)
    frames_n = max(1, int(req.get("clip_frames") or 4))
    dedupe = float(req.get("dedupe") or 0.0)
    prev = unb64(req["prev"]) if req.get("prev") else None
    known = req.get("repeats")
    if known is not None and len(known) != len(moments):
        raise ValueError("repeats must have one entry per moment")
    write_thumbs = req.get("thumbs", True) is not False
    thumb_bytes = 0
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if device == "cuda" and not _torch().cuda.is_available():
        device = "cpu"
    timing: dict[str, float] = {}
    out = [
        {"repeat": bool(known[k]) if known is not None else False, "still": False, "vectors": {}}
        for k in range(len(moments))
    ]
    model = _load(device, "all" if (kinds - {"speech"}) else "text")
    pool = ThreadPoolExecutor(workers)

    def timed(name, fn):
        t0 = time.time()
        try:
            return fn()
        finally:
            timing[name] = round(timing.get(name, 0.0) + time.time() - t0, 3)

    try:
        # 1.+2. Stills, and which moments repeat the last kept one (when no
        # earlier pass knows it already)
        visual = {"image", "video", "videoaudio"} & kinds
        need_stills = video and (visual if known is None else "image" in kinds)
        if need_stills:
            scale = f"scale={STILL_WIDTH}:-2"
            look = [k for k in range(len(moments)) if not out[k]["repeat"]]
            grabbed = timed(
                "grab_stills",
                lambda: list(pool.map(lambda k: grab_frame(path, moments[k]["t"], scale), look)),
            )
            stills = dict(zip(look, grabbed))
            have = [k for k in look if stills[k] is not None]
            if have:
                with _model_locks[device]:
                    vecs = timed(
                        "embed_stills", lambda: _encode(model, [stills[k] for k in have], batch * 2)
                    )
                for k, v in zip(have, vecs):
                    if (
                        known is None
                        and dedupe > 0
                        and prev is not None
                        and float(v @ prev) >= dedupe
                    ):
                        out[k]["repeat"] = True
                        continue
                    prev = v
                    out[k]["still"] = True
                    if "image" in kinds:
                        out[k]["vectors"]["image"] = b64(v)
                    thumb = moments[k].get("thumb")
                    if thumb and write_thumbs and not os.path.isfile(thumb):
                        img = stills[k].copy()
                        img.thumbnail((int(req.get("thumb_width") or 480), 10_000))
                        os.makedirs(os.path.dirname(thumb), exist_ok=True)
                        img.save(thumb, quality=82)
                        thumb_bytes += os.path.getsize(thumb)
            if known is None:
                for k in look:
                    if stills[k] is None:
                        out[k]["repeat"] = True  # no picture there (a black or broken frame)
        elif video and known is not None:
            for k in range(len(moments)):
                out[k]["still"] = not out[k]["repeat"]
        # 3. Clips of the new moments (+ their sound)
        if video and ({"video", "videoaudio"} & kinds):
            fresh = [k for k in range(len(moments)) if not out[k]["repeat"]]

            def clip(k):
                m = moments[k]
                start = max(0.0, m["t"] - clip_sec * 0.3)
                frames = grab_frames(path, start, clip_sec, frames_n)
                sound = grab_sound(path, start, clip_sec) if "videoaudio" in kinds else None
                return frames, sound

            got = timed("grab_clips", lambda: list(pool.map(clip, fresh)))
            vid = [(k, f) for k, (f, _) in zip(fresh, got) if len(f) >= 2]
            if "video" in kinds and vid:
                with _model_locks[device]:
                    vecs = timed(
                        "embed_clips", lambda: _encode(model, [{"video": f} for _, f in vid], batch)
                    )
                for (k, _), v in zip(vid, vecs):
                    out[k]["vectors"]["video"] = b64(v)
            both = [(k, f, a) for k, (f, a) in zip(fresh, got) if len(f) >= 2 and a is not None]
            if "videoaudio" in kinds and both:
                inputs = [
                    {
                        "text": "<|video|> <|audio|>",
                        "video": f,
                        "audio": {"array": a, "sampling_rate": SR},
                    }
                    for _, f, a in both
                ]
                with _model_locks[device]:
                    vecs = timed("embed_clips_sound", lambda: _encode(model, inputs, batch))
                for (k, _, _), v in zip(both, vecs):
                    out[k]["vectors"]["videoaudio"] = b64(v)
        # 4. The sound of every moment
        if "audio" in kinds:
            sounds = timed(
                "grab_sounds",
                lambda: list(pool.map(lambda m: grab_sound(path, m["start"], m["dur"]), moments)),
            )
            have = [k for k, a in enumerate(sounds) if a is not None]
            if have:
                with _model_locks[device]:
                    vecs = timed(
                        "embed_sounds",
                        lambda: _encode(
                            model,
                            [{"array": sounds[k], "sampling_rate": SR} for k in have],
                            batch * 4,
                        ),
                    )
                for k, v in zip(have, vecs):
                    out[k]["vectors"]["audio"] = b64(v)
        # 5. What is said there
        if "speech" in kinds:
            said = [(k, (m.get("text") or "").strip()) for k, m in enumerate(moments)]
            said = [(k, t) for k, t in said if len(t.split()) >= 3]
            if said:
                with _model_locks[device]:
                    vecs = timed(
                        "embed_speech",
                        lambda: _encode(model, [t for _, t in said], batch * 8, "Document"),
                    )
                for (k, _), v in zip(said, vecs):
                    out[k]["vectors"]["speech"] = b64(v)
    finally:
        pool.shutdown(wait=False)
    return {
        "moments": out,
        "prev": b64(prev) if prev is not None else None,
        "thumb_bytes": thumb_bytes,
        "timing": timing,
        "device": device,
    }


# ── HTTP ────────────────────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        torch = _torch()
        self._send(
            200,
            {
                "ok": True,
                "model": _args.model,
                "pid": os.getpid(),
                "uptime": round(time.time() - _started),
                "loaded": sorted(f"{d}:{t}" for d, t in _models),
                "cuda": bool(torch.cuda.is_available()),
                "versions": _versions(),
            },
        )

    def _refused(self) -> str:
        """Why this request is not the app's own ("" = it is): the key of
        this start, a Host of this machine (DNS rebinding), and JSON (a
        page's simple text/plain POST needs no preflight)."""
        token = os.environ.get("WHISPER_TM_MEDIA_TOKEN", "")
        if not token or not hmac.compare_digest(
            self.headers.get("X-Media-Token", "").encode(), token.encode()
        ):
            return "missing or wrong X-Media-Token"
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if host not in ("127.0.0.1", "localhost", "::1"):
            return "foreign Host"
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return "not JSON"
        return ""

    def do_POST(self):  # noqa: N802
        t0 = time.time()
        why = self._refused()
        if why:
            self._send(403, {"error": f"refused: {why}"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/embed":
                texts = [str(x)[:4000] for x in req.get("texts") or []][:256]
                if not texts:
                    raise ValueError("no texts")
                vecs = embed_texts(texts, req.get("prompt") or "SearchQuery")
                self._send(
                    200, {"vectors": [b64(v) for v in vecs], "ms": round((time.time() - t0) * 1000)}
                )
            elif self.path == "/embed-file":
                path = req.get("path") or ""
                if not os.path.isfile(path):
                    raise ValueError("no such file")
                v = embed_file(
                    path,
                    req.get("kind") or "",
                    str(req.get("text") or "")[:500].strip(),
                    bool(req.get("with_sound")),
                    int(req.get("frames") or 4),
                )
                self._send(200, {"vector": b64(v), "ms": round((time.time() - t0) * 1000)})
            elif self.path == "/index":
                res = index_chunk(req)
                res["ms"] = round((time.time() - t0) * 1000)
                self._send(200, res)
            elif self.path == "/release":
                self._send(200, {"released": release(req.get("device") or "cuda")})
            else:
                self._send(404, {"error": "not found"})
        except FileNotFoundError as e:
            self._send(404, {"error": f"file not found: {e}"})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — answer instead of dropping the connection
            oom = "out of memory" in str(e).lower()
            if oom:
                release("cuda")
            logger.exception("Request %s failed", self.path)
            self._send(507 if oom else 500, {"error": f"{type(e).__name__}: {e}", "oom": oom})

    def log_message(self, *args):
        pass


def _versions() -> dict:
    out = {}
    for name in ("torch", "transformers", "sentence_transformers"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def main(argv=None) -> None:
    global _args
    p = argparse.ArgumentParser(
        description="EmbeddingGemma 2 service of Whisper Transcription Manager's Images & sounds"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8083)
    p.add_argument("--model", default="google/embeddinggemma-2")
    p.add_argument("--threads", type=int, default=8, help="CPU threads for queries")
    _args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    _torch().set_num_threads(max(1, _args.threads))
    server = ThreadingHTTPServer((_args.host, _args.port), Handler)
    logger.info(
        "Media service on %s:%d (model %s, pid %d)",
        _args.host,
        _args.port,
        _args.model,
        os.getpid(),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
