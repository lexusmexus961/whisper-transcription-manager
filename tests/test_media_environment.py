"""The media environment (media/environment.py) and the service process
(core/media_server.py) — with a stand-in Python and a stand-in service, no
model and no network."""

from __future__ import annotations

import json
import socket
import stat
import sys
import textwrap
import time

import pytest

from whisper_tm.core import media_server
from whisper_tm.media import environment


def _script(path, body: str):
    path.write_text("#!" + sys.executable + "\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def envdir(tmp_path, monkeypatch):
    cfg = {"environment": str(tmp_path / "env"), "model": "google/embeddinggemma-2"}
    monkeypatch.setattr(environment, "_cfg", lambda name, default=None: cfg.get(name, default))
    environment._check_cache.update(at=0.0, key=None, value=None, refreshing=False)
    return {"cfg": cfg, "dir": tmp_path / "env"}


class TestTheCheck:
    def test_not_installed(self, envdir):
        st = environment.check()
        assert not st["ok"] and st["problems"] == ["not installed"] and not st["exists"]

    def _fake_python(self, envdir, packages: dict, model: bool):
        py = environment.python_path()
        py.parent.mkdir(parents=True)
        answer = json.dumps({**packages, "model": model})
        _script(
            py,
            f"""
            print({answer!r})
        """,
        )
        return py

    def test_ready(self, envdir):
        self._fake_python(
            envdir,
            {
                "torch": "2.8.0",
                "transformers": "5.19.0",
                "sentence_transformers": "6.1.0",
                "av": "18.0",
                "PIL": "12.0",
                "numpy": "2.4",
            },
            True,
        )
        st = environment.check()
        assert st["ok"] and st["model_cached"] and st["packages"]["transformers"] == "5.19.0"

    def test_what_is_missing_is_named(self, envdir):
        self._fake_python(
            envdir,
            {
                "torch": "2.8.0",
                "transformers": "4.57.6",
                "sentence_transformers": None,
                "av": "18.0",
                "PIL": "12.0",
                "numpy": "2.4",
            },
            False,
        )
        st = environment.check()
        assert not st["ok"]
        assert "sentence_transformers missing" in st["problems"]
        assert any("transformers 4.57.6 is too old" in p for p in st["problems"])
        assert "model not downloaded yet" in st["problems"]

    def test_kept_and_looked_at_again_in_the_background(self, envdir, monkeypatch):
        calls = []
        real = environment._probe
        monkeypatch.setattr(environment, "_probe", lambda key: calls.append(key) or real(key))
        environment.check()
        environment.check()
        assert len(calls) == 1  # kept
        environment._check_cache["at"] -= 120  # a minute old
        environment.check()  # answers at once …
        for _ in range(100):
            if len(calls) == 2:
                break
            time.sleep(0.01)
        assert len(calls) == 2  # … and looks again behind it
        environment.check(force=True)
        assert len(calls) == 3


class TestInstalling:
    def test_through_uv_then_the_model(self, envdir, monkeypatch, tmp_path):
        log = tmp_path / "uv.log"
        uv = _script(
            tmp_path / "uv",
            f"""
            import os, sys, json
            open({str(log)!r}, "a").write(json.dumps(sys.argv[1:]) + "\\n")
            if sys.argv[1] == "venv":
                env = sys.argv[2]
                os.makedirs(env + "/bin", exist_ok=True)
                py = env + "/bin/python"
                open(py, "w").write("#!{sys.executable}\\nimport sys, json\\n"
                    "if '-c' in sys.argv and 'snapshot_download' in sys.argv[-1]: print('/hf/model'); sys.exit(0)\\n"
                    "print(json.dumps({{'torch': '2.8.0', 'transformers': '5.19.0', 'sentence_transformers': '6.1.0',"
                    " 'av': '18', 'PIL': '12', 'numpy': '2', 'model': True}}))\\n")
                os.chmod(py, 0o755)
            print("done")
        """,
        )
        monkeypatch.setattr(environment, "_uv", lambda: str(uv))
        monkeypatch.setattr(environment, "_has_nvidia", lambda: False)
        assert environment.start_install()
        for _ in range(500):
            if not environment.install_state()["running"]:
                break
            time.sleep(0.01)
        st = environment.install_state()
        assert st["ok"] is True and st["error"] is None, st
        argv = [json.loads(line) for line in log.read_text().splitlines()]
        assert argv[0][0] == "venv" and argv[1][:2] == ["pip", "install"]
        assert "https://download.pytorch.org/whl/cpu" in argv[1]  # no NVIDIA card: CPU wheels
        assert "transformers==5.19.0" in argv[1] and "sentence-transformers==6.1.0" in argv[1]
        assert any("Ready." in line for line in st["lines"])

    def test_a_failing_step_is_reported(self, envdir, monkeypatch, tmp_path):
        uv = _script(tmp_path / "uv", "import sys\nprint('no network')\nsys.exit(2)\n")
        monkeypatch.setattr(environment, "_uv", lambda: str(uv))
        assert environment.start_install()
        for _ in range(500):
            if not environment.install_state()["running"]:
                break
            time.sleep(0.01)
        st = environment.install_state()
        assert st["ok"] is False and "exit code 2" in st["error"]


# ── The service process ────────────────────────────────────────────────────


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


SERVICE = """
import argparse, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
p = argparse.ArgumentParser()
p.add_argument("--host"); p.add_argument("--port", type=int); p.add_argument("--model"); p.add_argument("--threads")
a = p.parse_args()
class H(BaseHTTPRequestHandler):
    def _send(self, code, d):
        b = json.dumps(d).encode(); self.send_response(code); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send(200, {"ok": True, "model": a.model, "pid": os.getpid(), "loaded": [], "versions": {}})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/boom":
            self._send(507, {"error": "CUDA out of memory", "oom": True}); return
        self._send(200, {"echo": req, "path": self.path})
    def log_message(self, *x): pass
ThreadingHTTPServer((a.host, a.port), H).serve_forever()
"""


@pytest.fixture
def server(tmp_path, monkeypatch):
    port = _free_port()
    cfg = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": port,
        "start_timeout_sec": 20,
        "idle_stop_sec": 0,
        "cpu_threads": 2,
        "index_device": "cpu",
    }
    monkeypatch.setattr(media_server, "_cfg", lambda name, default=None: cfg.get(name, default))
    script = tmp_path / "media" / "service.py"
    script.parent.mkdir()
    script.write_text(SERVICE)
    monkeypatch.setattr(media_server, "service_script", lambda: script)
    monkeypatch.setattr(media_server, "base_directory", lambda: tmp_path)
    monkeypatch.setattr(
        environment, "python_path", lambda: __import__("pathlib").Path(sys.executable)
    )
    monkeypatch.setattr(environment, "model_id", lambda: "fake/model")
    monkeypatch.setattr(
        environment, "check", lambda force=False: {"ok": True, "model_cached": True}
    )
    srv = media_server.MediaServer()
    yield {"srv": srv, "cfg": cfg, "port": port}
    srv.stop("test end", force=True)


class TestTheServiceProcess:
    def test_started_on_the_first_call_and_answering(self, server):
        srv = server["srv"]
        assert not srv.state()["running"]
        out = srv.call("/embed", {"texts": ["x"]})
        assert out == {"echo": {"texts": ["x"]}, "path": "/embed"}
        st = srv.state()
        assert st["running"] and st["pid"]
        assert srv.stop("done", force=True) and not srv.state()["running"]

    def test_an_error_answer_keeps_its_status_and_oom(self, server):
        with pytest.raises(media_server.MediaUnavailable) as e:
            server["srv"].call("/boom", {})
        assert e.value.status == 507 and e.value.oom and "out of memory" in str(e.value)

    def test_switched_off_or_not_installed(self, server, monkeypatch):
        server["cfg"]["enabled"] = False
        with pytest.raises(media_server.MediaUnavailable, match="switched off"):
            server["srv"].ensure()
        server["cfg"]["enabled"] = True
        monkeypatch.setattr(
            environment, "python_path", lambda: __import__("pathlib").Path("/nope/python")
        )
        with pytest.raises(media_server.MediaUnavailable, match="not installed"):
            server["srv"].ensure()

    def test_a_leftover_of_an_earlier_run_is_replaced(self, server):
        first = media_server.MediaServer()
        first.ensure()
        old = first.state()["pid"]
        leftover = first._proc
        first._proc = None  # as after a crash of the app: the service lives on, unknown to us
        second = server["srv"]
        second.ensure()
        new = second.state()["pid"]
        assert new and new != old
        # The leftover is gone: it exits. Waited for, not probed with
        # kill(pid, 0) — an exited child answers that as a zombie until its
        # parent (this test) collects it, which made the check flaky
        assert leftover.wait(timeout=10) is not None

    def test_a_foreign_program_on_the_port_is_not_touched(self, server, tmp_path):
        import subprocess

        foreign = tmp_path / "other.py"
        foreign.write_text(SERVICE)  # the same answers, but not our script
        p = subprocess.Popen(
            [
                sys.executable,
                str(foreign),
                "--host",
                "127.0.0.1",
                "--port",
                str(server["port"]),
                "--model",
                "x",
                "--threads",
                "1",
            ]
        )
        try:
            for _ in range(100):
                try:
                    socket.create_connection(("127.0.0.1", server["port"]), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.05)
            with pytest.raises(media_server.MediaUnavailable, match="taken by another program"):
                server["srv"].ensure()
            assert p.poll() is None
        finally:
            p.kill()

    def test_idle_stop(self, server, monkeypatch):
        monkeypatch.setattr(media_server, "WATCH_TICK_SEC", 0.05)
        server["cfg"]["idle_stop_sec"] = 0.1
        srv = server["srv"]
        srv.ensure()
        for _ in range(200):
            if not srv.state()["running"]:
                break
            time.sleep(0.02)
        assert not srv.state()["running"]

    def test_no_idle_stop_during_an_index_run(self, server, monkeypatch):
        monkeypatch.setattr(media_server, "WATCH_TICK_SEC", 0.05)
        server["cfg"]["idle_stop_sec"] = 0.1
        srv = server["srv"]
        srv.indexing = True
        srv.ensure()
        time.sleep(0.4)
        assert srv.state()["running"]

    def test_a_refused_idle_stop_keeps_the_watcher(self, server, monkeypatch):
        # Previously the watcher returned even when stop() refused (a call came in
        # between) — the service then ran on without an idle stop
        monkeypatch.setattr(media_server, "WATCH_TICK_SEC", 0.05)
        server["cfg"]["idle_stop_sec"] = 0.1
        srv = server["srv"]
        real, refused = srv.stop, []

        def stop(reason="", *, force=False):
            if not force and not refused:
                refused.append(reason)
                return False
            return real(reason, force=force)

        monkeypatch.setattr(srv, "stop", stop)
        srv.ensure()
        for _ in range(200):
            if not srv.state()["running"]:
                break
            time.sleep(0.02)
        assert refused and not srv.state()["running"]

    def test_others_wait_for_a_cold_start_instead_of_posting_early(self, server, tmp_path):
        # Previously alive and an address were enough for the fast path — a
        # second caller posted before /health answered and failed
        (tmp_path / "media" / "service.py").write_text(
            SERVICE.replace(
                "a = p.parse_args()", "a = p.parse_args()\nimport time; time.sleep(1.5)"
            )
        )
        srv = server["srv"]
        out: dict = {}

        def call(name):
            try:
                out[name] = srv.call("/embed", {"n": name})
            except Exception as e:  # noqa: BLE001
                out[name] = e

        import threading

        first = threading.Thread(target=call, args=("first",))
        first.start()
        for _ in range(200):
            if srv.state()["running"]:
                break
            time.sleep(0.01)
        second = threading.Thread(target=call, args=("second",))
        second.start()
        first.join(20)
        second.join(20)
        assert out["first"]["path"] == out["second"]["path"] == "/embed", out
