"""What the settings redesign fixed underneath the page.

The audit of every setting found values that did nothing, values that lost
their type on the way to config.yaml, a reset that switched the login off,
and a settings API that let anyone on the network put a command into every
yt-dlp call. Each finding has its test here.
"""

from __future__ import annotations

import pytest
import yaml

from whisper_tm.config import SECRET_PLACEHOLDER, ConfigManager, coerce_value, config


def _load(tmp_path, data: dict) -> ConfigManager:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    cm = ConfigManager()
    cm.load(p)
    return cm


# ── Values keep their type ────────────────────────────────────────────────


class TestTypes:
    def test_the_string_false_becomes_false(self, tmp_path):
        # The old page saved the embedding mode as "true"/"false" — and
        # bool("false") is True: the mode could never be switched back
        cm = _load(tmp_path, {"embedding": {"local": "false"}})
        assert cm.get("embedding", "local") is False
        cm = _load(tmp_path, {"embedding": {"local": "true"}})
        assert cm.get("embedding", "local") is True

    def test_numbers_in_quotes_become_numbers(self, tmp_path):
        cm = _load(
            tmp_path,
            {"hardware": {"oom_min_free_mb": "6000", "diarization_deadline_factor": "0.25"}},
        )
        assert cm.get("hardware", "oom_min_free_mb") == 6000
        assert cm.get("hardware", "diarization_deadline_factor") == 0.25

    def test_only_what_is_unambiguous_is_converted(self, tmp_path):
        cm = _load(
            tmp_path, {"hardware": {"oom_min_free_mb": "lots"}, "embedding": {"local": "maybe"}}
        )
        assert cm.get("hardware", "oom_min_free_mb") == "lots"
        assert cm.get("embedding", "local") == "maybe"

    def test_a_list_written_as_text(self):
        assert coerce_value(".rmvb, .vob", []) == [".rmvb", ".vob"]
        # one per line when there are lines — a phrase may hold a comma
        assert coerce_value("Thanks, see you\nMusic", []) == ["Thanks, see you", "Music"]

    def test_the_embedding_readers_parse_the_mode(self, monkeypatch):
        from whisper_tm.rag.service import embedding_config

        monkeypatch.setitem(config._data.setdefault("embedding", {}), "local", "false")
        assert embedding_config()["local"] is False


# ── Command-line flags stay out of config.yaml ────────────────────────────


class TestRunOnlyValues:
    def test_a_flag_is_not_saved(self, tmp_path):
        cm = _load(tmp_path, {"server": {"port": 5077}})
        cm.override("server.port", 5299)
        assert cm.get("server", "port") == 5299
        cm.save(tmp_path / "config.yaml")
        saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        assert saved["server"]["port"] == 5077

    def test_a_flag_for_a_key_the_file_lacked_is_left_out(self, tmp_path):
        cm = _load(tmp_path, {"queue": {"autostart": False}})
        cm.override("logging.trace_sql", True)
        cm.save(tmp_path / "config.yaml")
        saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        # not the flag: the default — which a file only holds when it differs
        assert (saved.get("logging") or {}).get("trace_sql") is not True

    def test_a_value_set_on_purpose_wins_over_the_flag(self, tmp_path):
        cm = _load(tmp_path, {"server": {"port": 5077}})
        cm.override("server.port", 5299)
        cm.set("server.port", 6000)
        cm.save(tmp_path / "config.yaml")
        saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        assert saved["server"]["port"] == 6000


# ── Smaller fixes ─────────────────────────────────────────────────────────


class TestSmallerFixes:
    def test_extra_extensions_are_compared_like_file_names(self, tmp_path):
        cm = _load(tmp_path, {"scanner": {"extra_video_extensions": ["RMVB", ".VOB"]}})
        assert {".rmvb", ".vob"} <= cm.video_extensions

    def test_a_wrong_type_is_a_warning_not_a_crash(self, tmp_path):
        cm = _load(
            tmp_path,
            {"vocal_isolation": {"mdx_overlap": "x"}, "hardware": {"asr_block_max_sec": 0}},
        )
        warnings = cm.validate()
        assert any("asr_block_max_sec" in w for w in warnings)

    def test_names_of_templates_are_not_secrets(self, tmp_path):
        # A template called "Token counts" came back as the placeholder, and
        # saving wrote the placeholder over its prompt
        cm = _load(
            tmp_path, {"llm": {"presets": {"Token counts": "Count them."}, "api_key": "sk-1"}}
        )
        masked = cm.all_masked()
        assert masked["llm"]["presets"]["Token counts"] == "Count them."
        assert masked["llm"]["api_key"] == SECRET_PLACEHOLDER


# ── Artifact patterns (regular expressions, not phrases) ─────────────────


class TestArtifactPatterns:
    def test_the_shipped_patterns_are_the_credits(self):
        from pathlib import Path

        d = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config.default.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert d["transcription"]["hallucination_patterns"] == [
            r"Untertitelung.*ZDF.*",
            r"Untertitelung.*BR \d{4}",
            r"Subtitles by the Amara\.org community",
            "Transcription by CastingWords",
        ]
        assert "custom_hallucinations" not in d["transcription"]
        assert "disabled_hallucinations" not in d["transcription"]

    def test_patterns_of_your_own_are_not_touched(self, tmp_path):
        cm = _load(
            tmp_path,
            {
                "transcription": {
                    "custom_hallucinations": ["End credits"],
                    "hallucination_patterns": ["Mine.*"],
                }
            },
        )
        assert cm.get("transcription", "hallucination_patterns") == ["Mine.*"]

    def test_no_own_phrases_leaves_the_shipped_ones(self, tmp_path):
        cm = _load(tmp_path, {"transcription": {"custom_hallucinations": []}})
        assert cm.get("transcription", "hallucination_patterns")[:2] == [
            r"Untertitelung.*ZDF.*",
            r"Untertitelung.*BR \d{4}",
        ]


# ── The settings API ──────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    own = tmp_path / "config.yaml"
    own.write_text("queue:\n  autostart: false\n  lanes: 1\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_TM_CONFIG", str(own))
    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path / "media")
    config._data["directories"]["work"] = str(tmp_path / "work")
    config._data.setdefault("database", {})["path"] = str(tmp_path / "own.db")
    (tmp_path / "media").mkdir()
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    yield create_app().test_client(), tmp_path
    config.load(None)


class TestSettingsApi:
    def test_the_page_gets_the_defaults(self, client):
        c, _ = client
        d = c.get("/api/settings").get_json()
        assert d["defaults"]["search"]["max_results"] == 1000
        assert "hallucinations_builtin" not in d  # no built-in list any more

    def test_a_broken_artifact_pattern_is_refused(self, client):
        c, _ = client
        before = config.get("transcription", "hallucination_patterns")
        d = c.put(
            "/api/settings",
            json={"transcription": {"hallucination_patterns": ["Fine.*", "(unclosed"]}},
        ).get_json()
        assert not d["ok"] and "not a valid regular expression" in " ".join(d["errors"])
        assert config.get("transcription", "hallucination_patterns") == before

    def test_good_patterns_are_saved(self, client):
        c, _ = client
        d = c.put(
            "/api/settings", json={"transcription": {"hallucination_patterns": ["Subtitles by.*"]}}
        ).get_json()
        assert d["ok"]
        assert config.get("transcription", "hallucination_patterns") == ["Subtitles by.*"]

    @pytest.mark.parametrize(
        "text, dropped, reason",
        [
            ("Untertitelung des ZDF, 2020", True, r"Untertitelung.*ZDF.*"),
            ("Untertitelung. BR 2018", True, r"Untertitelung.*BR \d{4}"),
            ("A real sentence.", False, None),
            ("…", True, "no letter or digit"),
        ],
    )
    def test_trying_lines_before_saving(self, client, text, dropped, reason):
        c, _ = client
        d = c.post(
            "/api/settings/artifact_test",
            json={
                "patterns": [r"Untertitelung.*ZDF.*", "", r"Untertitelung.*BR \d{4}"],
                "text": text,
            },
        ).get_json()
        assert d == {"dropped": dropped, "reason": reason, "errors": []}

    def test_trying_names_a_broken_line(self, client):
        c, _ = client
        d = c.post(
            "/api/settings/artifact_test",
            json={"patterns": ["ok.*", "(unclosed"], "text": "ok then"},
        ).get_json()
        assert d["dropped"] and d["errors"] and d["errors"][0].startswith("line 2")

    def test_a_select_string_is_saved_as_a_boolean(self, client):
        c, _ = client
        r = c.put("/api/settings", json={"embedding": {"local": "false"}})
        assert r.get_json()["ok"]
        assert config.get("embedding", "local") is False

    @pytest.mark.parametrize(
        "key, value",
        [
            ("security.allowed_roots", ["/"]),
            ("auth.public_paths", ["/"]),
            ("auth.enabled", False),
            ("server.port", 1),
            ("database.path", "/tmp/other.db"),
            ("logging.debug_dir", "/"),
            ("llama.extra_args", "; rm -rf ~"),
            ("embedding.local_extra_args", "--log-file /etc/x"),
        ],
    )
    def test_what_only_the_file_may_change(self, client, key, value):
        c, _ = client
        before = config.get(key)
        section, _, leaf = key.partition(".")
        r = c.put("/api/settings", json={section: {leaf.split(".")[0]: value}})
        d = r.get_json()
        assert not d["ok"] and any(key in e for e in d["errors"])
        assert config.get(key) == before

    def test_a_yt_dlp_option_that_runs_programs_is_refused(self, client):
        c, _ = client
        for bad in (
            '--exec "touch /tmp/x"',
            "--ffmpeg-location /tmp/evil",
            "-o /etc/x",
            "--exec-before-download=id",
            "--downloader curl",
        ):
            d = c.put("/api/settings", json={"youtube": {"extra_args": bad}}).get_json()
            assert not d["ok"], bad
            assert config.get("youtube", "extra_args") in ("", None), bad
        d = c.put(
            "/api/settings",
            json={"youtube": {"extra_args": "--limit-rate 5M --sub-langs de --write-subs"}},
        ).get_json()
        assert d["ok"]

    def test_a_size_a_job_divides_by_can_not_be_zero(self, client):
        c, _ = client
        d = c.put("/api/settings", json={"hardware": {"asr_block_max_sec": 0}}).get_json()
        assert not d["ok"]
        assert config.get("hardware", "asr_block_max_sec") == 54000

    def test_a_deleted_vocabulary_stays_deleted(self, client):
        c, _ = client
        c.put(
            "/api/settings", json={"transcription": {"vocab_presets": {"A": "x", "Podcast": "y"}}}
        )
        assert config.get("transcription", "vocab_presets") == {"A": "x", "Podcast": "y"}
        c.put("/api/settings", json={"transcription": {"vocab_presets": {"Podcast": "y"}}})
        assert config.get("transcription", "vocab_presets") == {"Podcast": "y"}

    def test_the_media_folder_goes_through_its_setter(self, client):
        c, tmp = client
        d = c.put("/api/settings", json={"directories": {"media": str(tmp / "missing")}}).get_json()
        assert not d["ok"]
        (tmp / "media" / "sub").mkdir()
        d = c.put(
            "/api/settings", json={"directories": {"media": str(tmp / "media" / "sub")}}
        ).get_json()
        assert d["ok"] and config.media_dir == (tmp / "media" / "sub").resolve()

    def test_a_reset_keeps_login_access_and_what_the_user_made(self, client):
        c, _ = client
        config.set("auth.enabled", True)
        config.set("auth.password", "pw")
        config.set("security.allowed_roots", ["/mnt/media"])
        config.set("database.backup_path", "/mnt/media/backups")
        config.set("transcription.hf_token", "hf_x")
        config.set("llm.presets", {"Mine": "my prompt"})
        config.set("transcription.vocab_presets", {"Series": "names"})
        config.set("search.max_results", 42)
        assert c.post("/api/settings/reset").get_json()["ok"]
        assert config.get("search", "max_results") == 1000  # reset
        assert config.get("auth", "enabled") is True
        assert config.get("auth", "password") == "pw"
        assert config.get("security", "allowed_roots") == ["/mnt/media"]
        assert config.get("database", "backup_path") == "/mnt/media/backups"
        assert config.get("transcription", "hf_token") == "hf_x"
        assert config.get("llm", "presets") == {"Mine": "my prompt"}
        assert config.get("transcription", "vocab_presets") == {"Series": "names"}


class TestDoneSounds:
    def test_an_uploaded_sound_survives_the_work_folder_cleanup(self, client):
        import io

        c, _ = client
        r = c.post(
            "/api/sounds/upload",
            data={"file": (io.BytesIO(b"ID3"), "ding.mp3")},
            content_type="multipart/form-data",
        )
        assert r.get_json()["ok"]
        config.clean_workdir()
        names = [s["name"] for s in c.get("/api/sounds").get_json()["sounds"]]
        assert "custom/ding.mp3" in names
        assert (config.data_dir / "sounds" / "ding.mp3").exists()

    def test_a_chosen_sound_that_is_gone_falls_back(self, client):
        c, _ = client
        c.post("/api/sounds/select", json={"name": "custom/gone.mp3"})
        assert c.get("/api/sounds").get_json()["current"] == "done.mp3"


class TestWholeLists:
    def test_a_deleted_built_in_template_stays_deleted(self, tmp_path):
        # Merged with the shipped list, every deleted built-in template came
        # back at the next start
        cm = _load(
            tmp_path,
            {"llm": {"presets": {"Mine": "my prompt"}}, "transcription": {"vocab_presets": {}}},
        )
        assert cm.get("llm", "presets") == {"Mine": "my prompt"}
        assert cm.get("transcription", "vocab_presets") == {}

    def test_without_a_list_of_its_own_the_shipped_one_applies(self, tmp_path):
        cm = _load(tmp_path, {"queue": {"autostart": False}})
        assert "Summary" in cm.get("llm", "presets")

    def test_other_sections_still_merge(self, tmp_path):
        cm = _load(tmp_path, {"hardware": {"low_vram": True}})
        assert cm.get("hardware", "low_vram") is True
        assert cm.get("hardware", "oom_wait_min") == 10
