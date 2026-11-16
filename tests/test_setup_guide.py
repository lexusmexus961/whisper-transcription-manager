"""Setup guide: first-run flag, system check verdicts, GPU test, docs viewer.

The check's job is to name the RIGHT problem for the hardware at hand, so
most tests feed it a hardware situation (which cards, which PyTorch build,
what CTranslate2 sees) and look at the verdict. The machine this runs on
does not matter — nothing here touches a real GPU.
"""

from __future__ import annotations

import json
import time

import pytest
from flask import Flask

from whisper_tm.core import diagnostics as dg
from whisper_tm.database import Database

NV = {"vendor": "nvidia", "slot": "0000:01:00.0", "name": "GeForce RTX 4070", "vram_mb": 12288}
AMD = {"vendor": "amd", "slot": "0000:02:00.0", "name": "Navi 33 [Radeon RX 7600]", "vram_mb": 8176}
TORCH_CUDA = {
    "installed": True,
    "version": "2.8.0+cu128",
    "backend": "cuda",
    "backend_version": "12.8",
    "available": True,
    "devices": ["RTX 4070"],
}
TORCH_CUDA_BLIND = {**TORCH_CUDA, "available": False, "devices": []}
TORCH_ROCM = {
    "installed": True,
    "version": "2.8.0+rocm7.2",
    "backend": "rocm",
    "backend_version": "7.2",
    "available": True,
    "devices": ["RX 7600"],
}
CT2_GPU = {"installed": True, "version": "4.8.1", "gpu_count": 1}
CT2_CPU = {"installed": True, "version": "4.8.1", "gpu_count": 0}


def _by_title(checks, title):
    return next(c for c in checks if c["title"] == title)


# ── GPU verdicts ─────────────────────────────────────────────────────────


class TestGpuChecks:
    def test_nvidia_all_good(self):
        c = dg._gpu_checks([NV], TORCH_CUDA, CT2_GPU)
        assert _by_title(c, "PyTorch")["status"] == dg.OK
        assert _by_title(c, "CTranslate2 (Whisper engine)")["status"] == dg.OK

    def test_amd_only_with_cuda_build_names_the_mismatch(self):
        """The typical mistake after swapping the card: NVIDIA packages still installed."""
        c = dg._gpu_checks([AMD], TORCH_CUDA_BLIND, CT2_CPU)
        torch_c = _by_title(c, "PyTorch")
        assert torch_c["status"] == dg.FAIL
        assert "ROCm" in torch_c["hint"]
        ct2 = _by_title(c, "CTranslate2 (Whisper engine)")
        assert ct2["status"] == dg.FAIL
        assert "WITH_HIP" in ct2["hint"]

    def test_no_gpu_is_a_warning_not_a_failure(self):
        c = dg._gpu_checks([], {**TORCH_CUDA_BLIND, "backend": "cpu"}, CT2_CPU)
        assert _by_title(c, "Graphics cards")["status"] == dg.WARN
        assert _by_title(c, "CTranslate2 (Whisper engine)")["status"] == dg.INFO

    @pytest.mark.parametrize(
        "torch_i",
        [
            {
                "installed": True,
                "version": "2.8.0+cpu",
                "backend": "cpu",
                "backend_version": "",
                "available": False,
                "devices": [],
            },  # setup.sh --gpu cpu
            TORCH_CUDA_BLIND,  # the Docker CPU variant
        ],
    )
    def test_a_machine_without_a_card_is_not_sent_looking_for_a_driver(self, torch_i):
        """A correct CPU installation looked half broken: three lines for one
        fact, two of them pointing to nvidia-smi and the CUDA libraries."""
        c = dg._gpu_checks([], torch_i, CT2_CPU)
        assert _by_title(c, "Graphics cards")["status"] == dg.WARN  # the one line that says it
        torch_c = _by_title(c, "PyTorch")
        assert (
            torch_c["status"] == dg.OK
            and "CPU" in torch_c["detail"]
            and "( " not in torch_c["detail"]
        )
        ct2 = _by_title(c, "CTranslate2 (Whisper engine)")
        assert ct2["status"] == dg.INFO
        said = " ".join(f"{x['detail']} {x.get('hint', '')}" for x in c).lower()
        assert "driver" not in said and "nvidia-smi" not in said and "cuda libraries" not in said

    def test_a_card_that_is_there_but_not_usable_still_gets_the_hint(self):
        c = dg._gpu_checks([NV], TORCH_CUDA_BLIND, CT2_CPU)
        assert (
            _by_title(c, "PyTorch")["status"] == dg.WARN
            and "nvidia-smi" in _by_title(c, "PyTorch")["hint"]
        )
        assert _by_title(c, "CTranslate2 (Whisper engine)")["status"] == dg.FAIL

    def test_missing_torch(self):
        c = dg._gpu_checks([NV], {"installed": False, "error": "No module named torch"}, CT2_GPU)
        assert _by_title(c, "PyTorch")["status"] == dg.FAIL

    @pytest.mark.parametrize(
        "card",
        [
            {"vendor": "intel", "slot": "0000:00:02.0", "name": "UHD Graphics 770", "vram_mb": 0},
            {
                "vendor": "other",
                "slot": "0000:00:01.0",
                "name": "QXL paravirtual graphic card",
                "vram_mb": 0,
            },
        ],
    )
    def test_an_igpu_or_a_vms_adapter_is_no_reason_to_reinstall(self, card):
        """A correct CPU installation was told to reinstall for NVIDIA because
        the machine's Intel iGPU (or a VM's display adapter) counted as a GPU."""
        cpu_torch = {**TORCH_CUDA_BLIND, "version": "2.8.0+cpu", "backend": "cpu"}
        c = dg._gpu_checks([card], cpu_torch, CT2_CPU)
        cards = _by_title(c, "Graphics cards")
        assert cards["status"] == dg.WARN and "No NVIDIA or AMD card" in cards["detail"]
        assert card["name"] in cards["detail"]
        assert "Reinstall" not in _by_title(c, "PyTorch")["hint"]
        assert _by_title(c, "CTranslate2 (Whisper engine)")["status"] == dg.INFO
        # with an NVIDIA card beside it, the CPU build is the mistake
        c = dg._gpu_checks([card, NV], cpu_torch, CT2_CPU)
        assert "Reinstall" in _by_title(c, "PyTorch")["hint"]


# ── AMD: experimental, and quiet when not in use ─────────────────────────


class TestAmdChecks:
    @pytest.fixture
    def no_rocm(self, monkeypatch):
        monkeypatch.setattr(
            dg,
            "rocm_info",
            lambda: {"installed": False, "version": "", "gfx": [], "hsa_override": ""},
        )

    def test_absent_without_amd(self, no_rocm):
        assert dg._amd_checks([NV], TORCH_CUDA) == []

    def test_every_amd_check_is_marked_experimental(self, no_rocm):
        c = dg._amd_checks([NV, AMD], TORCH_CUDA)
        assert c and all(x["experimental"] for x in c)
        assert "untested" in c[0]["detail"]

    def test_present_but_unused_card_raises_nothing(self, no_rocm):
        """Next to a working NVIDIA card, a missing ROCm is not a problem."""
        c = dg._amd_checks([NV, AMD], TORCH_CUDA)
        assert all(x["status"] == dg.INFO for x in c)

    def test_in_use_missing_rocm_fails(self, no_rocm):
        c = dg._amd_checks([AMD], TORCH_ROCM)
        assert _by_title(c, "ROCm")["status"] == dg.FAIL
        assert _by_title(c, "Status")["status"] == dg.WARN

    @pytest.mark.parametrize(
        "gfx,status", [("gfx1100", dg.OK), ("gfx1201", dg.OK), ("gfx1102", dg.WARN)]
    )
    def test_official_support_list(self, monkeypatch, gfx, status):
        monkeypatch.setattr(
            dg,
            "rocm_info",
            lambda: {"installed": True, "version": "7.2.0", "gfx": [gfx], "hsa_override": ""},
        )
        c = dg._amd_checks([AMD], TORCH_ROCM)
        assert _by_title(c, f"Target {gfx}")["status"] == status

    def test_unsupported_target_suggests_the_override(self, monkeypatch):
        monkeypatch.setattr(
            dg,
            "rocm_info",
            lambda: {"installed": True, "version": "7.2.0", "gfx": ["gfx1102"], "hsa_override": ""},
        )
        hint = _by_title(dg._amd_checks([AMD], TORCH_ROCM), "Target gfx1102")["hint"]
        assert "HSA_OVERRIDE_GFX_VERSION=11.0.0" in hint


# ── Overall verdict ──────────────────────────────────────────────────────


class TestOverall:
    def test_a_warning_never_hides_a_failure(self, monkeypatch):
        monkeypatch.setattr(dg, "list_gpus", lambda: [AMD])
        monkeypatch.setattr(dg, "_torch_info", lambda: TORCH_CUDA_BLIND)
        monkeypatch.setattr(dg, "_ct2_info", lambda: CT2_CPU)
        monkeypatch.setattr(
            dg,
            "rocm_info",
            lambda: {"installed": False, "version": "", "gfx": [], "hsa_override": ""},
        )
        assert dg.quick_checks()["overall"] == dg.FAIL


# ── GPU test subprocess ──────────────────────────────────────────────────


class TestGpuTestSubprocess:
    def _fake_run(self, monkeypatch, stdout="", stderr="", code=0, exc=None):
        class R:
            returncode = code

        R.stdout, R.stderr = stdout, stderr

        def run(*a, **k):
            if exc:
                raise exc
            return R

        monkeypatch.setattr(dg.subprocess, "run", run)

    def test_json_after_library_noise(self, monkeypatch):
        result = {"ok": True, "steps": [{"step": "x", "status": "ok", "detail": "", "secs": 0}]}
        self._fake_run(monkeypatch, stdout="some warning\n" + json.dumps(result) + "\n")
        assert dg.run_gpu_test_subprocess() == result

    def test_crash_is_reported_not_raised(self, monkeypatch):
        """A driver crash kills the test process — the server gets a readable error."""
        self._fake_run(monkeypatch, code=-11, stderr="Segmentation fault")
        r = dg.run_gpu_test_subprocess()
        assert r["ok"] is False
        assert "-11" in r["error"] and "Segmentation fault" in r["error"]

    def test_timeout(self, monkeypatch):
        self._fake_run(monkeypatch, exc=dg.subprocess.TimeoutExpired("x", 1))
        r = dg.run_gpu_test_subprocess(timeout=1)
        assert r["ok"] is False and "no result" in r["error"]


# ── HTTP endpoints ───────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = Database(tmp_path / "setup.db")
    d.initialize()
    monkeypatch.setattr("whisper_tm.api.setup.db", d)
    return d


@pytest.fixture
def docs(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "PATHS.md").write_text("# Paths — binding model\n\nText.\n", encoding="utf-8")
    (folder / "private.json").write_text('{"secret": 1}', encoding="utf-8")
    monkeypatch.setattr("whisper_tm.api.setup.DOCS_DIR", folder)
    return folder


@pytest.fixture
def client(db, docs):
    from whisper_tm.api.setup import setup_bp

    app = Flask(__name__)
    app.register_blueprint(setup_bp)
    return app.test_client()


class TestFirstRun:
    def test_fresh_installation_opens_the_guide(self, client):
        assert client.get("/api/setup/status").get_json()["first_run"] is True

    def test_done_closes_it_for_good(self, client):
        client.post("/api/setup/done")
        s = client.get("/api/setup/status").get_json()
        assert s["first_run"] is False and s["done"] is True

    def test_an_existing_library_is_not_a_first_run(self, client, db):
        """Installations from before the guide have no flag — but transcripts."""
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO transcripts (rel_path, abs_path, complete, created_at) "
                "VALUES ('a.mp4', '/m/a.mp4', 1, ?)",
                (time.time(),),
            )
        assert client.get("/api/setup/status").get_json()["first_run"] is False

    def test_it_names_the_commands(self, client, monkeypatch):
        """The guide's texts take the setup and start commands from here."""
        monkeypatch.delenv("WHISPER_TM_DOCKER", raising=False)
        c = client.get("/api/setup/status").get_json()["commands"]
        assert c == {
            "setup_llama": "./setup.sh --llama",
            "start": "./start.sh",
            "start_llm": "./start-llm.sh",
            "check": "./start.sh --check",
            "docker": False,
            "host_url": "http://host.docker.internal:<port>",
        }

    def test_in_docker_it_names_compose(self, client, monkeypatch):
        """The Docker image has no scripts: the guide names Compose."""
        monkeypatch.setenv("WHISPER_TM_DOCKER", "1")
        c = client.get("/api/setup/status").get_json()["commands"]
        assert c["docker"] is True
        assert c["start"] == "docker compose up -d"
        assert c["check"] == "docker compose run --rm whisper-tm --check"


class TestEndpoints:
    def test_check_adds_paths_and_ports(self, client, monkeypatch):
        monkeypatch.setattr(dg, "quick_checks", lambda: {"checks": [], "overall": "ok"})
        d = client.get("/api/setup/check").get_json()
        assert d["overall"] == "ok"
        assert {"config", "database", "media", "work"} <= set(d["paths"])
        assert "web" in d["ports"]

    def test_gpu_test_refused_while_a_job_runs(self, client, monkeypatch):
        monkeypatch.setattr("whisper_tm.core.model_idle._job_running", lambda: True)
        called = []
        monkeypatch.setattr(dg, "run_gpu_test_subprocess", lambda *a, **k: called.append(1))
        r = client.post("/api/setup/gpu_test")
        assert r.status_code == 409
        assert called == []

    def test_gpu_test_runs_when_idle(self, client, monkeypatch):
        monkeypatch.setattr("whisper_tm.core.model_idle._job_running", lambda: False)
        monkeypatch.setattr(
            dg, "run_gpu_test_subprocess", lambda idx=0, timeout=600: {"ok": True, "steps": []}
        )
        assert client.post("/api/setup/gpu_test").get_json()["ok"] is True


class TestDocs:
    def test_lists_markdown_only(self, client):
        docs = client.get("/api/setup/docs").get_json()["docs"]
        assert docs == [{"name": "PATHS.md", "title": "Paths — binding model"}]

    def test_reads_a_listed_document(self, client):
        d = client.get("/api/setup/docs/PATHS.md").get_json()
        assert d["text"].startswith("# Paths")

    @pytest.mark.parametrize("name", ["private.json", "..%2F..%2Fetc%2Fpasswd", "missing.md"])
    def test_anything_else_is_404(self, client, name):
        assert client.get(f"/api/setup/docs/{name}").status_code == 404

    def test_missing_docs_folder(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("whisper_tm.api.setup.DOCS_DIR", tmp_path / "nope")
        assert client.get("/api/setup/docs").get_json()["docs"] == []


def test_first_steps_offer_the_language():
    """'Detect automatically' is the default; a library in one language sets
    it once in the guide's first steps."""
    from pathlib import Path

    js = (Path(__file__).resolve().parent.parent / "whisper_tm/static/js/setupguide.js").read_text(
        encoding="utf-8"
    )
    assert "'3 · Language'" in js and 'id="sgLanguage"' in js
    assert "_sPutSettings({transcription: {default_language: value}})" in js
    assert "if (page === 'steps') sgLoadLanguage();" in js
