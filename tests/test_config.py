"""
Tests for config.py — configuration manager.
"""

from pathlib import Path

import pytest
import yaml

from whisper_tm.config import ConfigManager, _deep_merge, _expand_path


class TestDeepMerge:
    def test_flat(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested(self):
        base = {"server": {"host": "0.0.0.0", "port": 9860}}
        override = {"server": {"port": 8080}}
        result = _deep_merge(base, override)
        assert result["server"]["host"] == "0.0.0.0"
        assert result["server"]["port"] == 8080

    def test_override_dict_with_value(self):
        base = {"a": {"b": 1}}
        override = {"a": "flat"}
        result = _deep_merge(base, override)
        assert result["a"] == "flat"

    def test_no_mutation(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        _deep_merge(base, override)
        assert "c" not in base["a"]

    def test_empty_override(self):
        base = {"a": 1}
        result = _deep_merge(base, {})
        assert result == {"a": 1}


class TestExpandPath:
    def test_home_expansion(self):
        result = _expand_path("~/test")
        assert str(Path.home()) in str(result)

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_DIR", "/tmp/testdir")
        result = _expand_path("${TEST_DIR}/sub")
        assert "/tmp/testdir/sub" in str(result)

    def test_relative_with_base(self, tmp_path):
        result = _expand_path("./data", base_dir=tmp_path)
        assert str(tmp_path) in str(result)

    def test_absolute(self):
        result = _expand_path("/tmp/absolute")
        assert str(result) == str(Path("/tmp/absolute").resolve())

    def test_empty(self):
        result = _expand_path("")
        assert result == Path("")


class TestConfigManager:
    @pytest.fixture
    def cfg(self):
        return ConfigManager()

    @pytest.fixture
    def config_file(self, tmp_path):
        """Create a temporary config.yaml."""
        cfg_data = {
            "server": {"port": 9999, "host": "127.0.0.1"},
            "transcription": {"default_model": "large-v3"},
            "directories": {"media": str(tmp_path / "media")},
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))
        (tmp_path / "media").mkdir()
        return cfg_path

    def test_load_defaults(self, cfg):
        """Without config file, defaults are loaded."""
        cfg.load(config_path="/nonexistent/path.yaml")
        assert cfg.get("server", "port", default=9860) == 9860
        assert cfg.is_loaded

    def test_load_custom(self, cfg, config_file):
        """Custom config overrides defaults."""
        cfg.load(config_file)
        assert cfg.get("server", "port") == 9999
        assert cfg.get("server", "host") == "127.0.0.1"

    def test_defaults_preserved(self, cfg, config_file):
        """Unoverridden defaults are preserved."""
        cfg.load(config_file)
        # max_upload_mb is not in custom config → default
        assert cfg.get("server", "max_upload_mb", default=200) == 200

    def test_get_nested(self, cfg, config_file):
        cfg.load(config_file)
        assert cfg.get("transcription.default_model") == "large-v3"

    def test_get_default(self, cfg):
        cfg.load()
        assert cfg.get("nonexistent", "key", default="fallback") == "fallback"

    def test_set_and_get(self, cfg, config_file):
        cfg.load(config_file)
        cfg.set("server", "port", 1234)
        assert cfg.get("server", "port") == 1234

    def test_set_nested_dotted(self, cfg, config_file):
        cfg.load(config_file)
        cfg.set("server.port", 5678)
        assert cfg.get("server", "port") == 5678

    def test_is_loaded_flag(self, cfg):
        assert not cfg.is_loaded
        cfg.load()
        assert cfg.is_loaded

    def test_validate_valid(self, cfg, config_file):
        cfg.load(config_file)
        warnings = cfg.validate()
        assert isinstance(warnings, list)

    def test_validate_bad_model(self, cfg):
        cfg.load()
        cfg.set("transcription", "default_model", "nonexistent-model-v99")
        warnings = cfg.validate()
        assert any("Unknown Whisper model" in w for w in warnings)

    def test_validate_bad_device(self, cfg):
        cfg.load()
        cfg.set("hardware", "device", "quantum")
        warnings = cfg.validate()
        assert any("Unknown device" in w for w in warnings)

    def test_validate_bad_port(self, cfg):
        cfg.load()
        cfg.set("server", "port", 99999)
        warnings = cfg.validate()
        assert any("Invalid port" in w for w in warnings)

    def test_save_and_reload(self, cfg, tmp_path):
        cfg.load()
        cfg.set("server", "port", 4242)
        save_path = cfg.save(tmp_path / "saved.yaml")
        assert save_path.exists()

        cfg2 = ConfigManager()
        cfg2.load(save_path)
        assert cfg2.get("server", "port") == 4242

    def test_media_extensions(self, cfg):
        cfg.load()
        exts = cfg.media_extensions
        assert ".mp4" in exts
        assert ".mp3" in exts
        assert ".wav" in exts

    def test_hf_token_from_env(self, cfg, monkeypatch, tmp_path):
        # Deliberately load an EMPTY config: cfg.load() without path finds the
        # developer's real config — if there's a token there, it wins (config before
        # ENV) and the test failed exactly on the machine where the application is used.
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"transcription": {}}))
        cfg.load(cfg_path)
        monkeypatch.setenv("HF_TOKEN", "test-token-123")
        assert cfg.hf_token == "test-token-123"

    def test_hf_token_from_config(self, cfg, tmp_path):
        cfg_data = {"transcription": {"hf_token": "config-token"}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))
        cfg.load(cfg_path)
        assert cfg.hf_token == "config-token"


class TestSaveTarget:
    """save() must write to where it was actually loaded from.

    Regression: a test server with WHISPER_TM_CONFIG on a
    scratch config overwrote the REAL home config of the installation on save
    — save() went unconditionally to home.
    """

    def test_forced_config_goes_back_to_file(self, monkeypatch, tmp_path):
        from whisper_tm.config import ConfigManager

        own_ones = tmp_path / "forced.yaml"
        own_ones.write_text(yaml.dump({"server": {"port": 5177}}))
        monkeypatch.setenv("WHISPER_TM_CONFIG", str(own_ones))
        cfg = ConfigManager()
        cfg.load()
        assert cfg.config_path == own_ones.resolve()
        target = cfg.save()
        assert target == own_ones.resolve()  # NOT the home config
        assert yaml.safe_load(own_ones.read_text())["server"]["port"] == 5177

    def test_explicit_path_goes_back(self, monkeypatch, tmp_path):
        from whisper_tm.config import ConfigManager

        monkeypatch.delenv("WHISPER_TM_CONFIG", raising=False)
        own_ones = tmp_path / "explicit.yaml"
        own_ones.write_text(yaml.dump({"server": {"port": 5198}}))
        cfg = ConfigManager()
        cfg.load(own_ones)
        assert cfg.save() == own_ones.resolve()

    def test_normal_config_stays_home(self, monkeypatch, tmp_path):
        """Without a file loaded, the home config is made."""
        import whisper_tm.config as cfg_mod
        from whisper_tm.config import ConfigManager

        monkeypatch.delenv("WHISPER_TM_CONFIG", raising=False)
        fake_home = tmp_path / "home-cfg"
        monkeypatch.setattr(cfg_mod, "user_config_dir", lambda name: str(fake_home))
        monkeypatch.chdir(tmp_path)  # no ./config.yaml
        cfg = ConfigManager()
        cfg.load()
        target = cfg.save()
        assert target == fake_home / "config.yaml"


class TestSavedIntoTheLoadedFile:
    """The first save of an installation loaded from ./config.yaml wrote the
    whole merged configuration into the home config: from then on that
    outranked ./config.yaml (edits there did nothing), the MCP token next to
    it was made anew, and every default was frozen at that version's value.
    Now a save goes back into the file that was loaded, with only what
    differs from the defaults."""

    def test_the_app_folders_file_stays_the_one(self, monkeypatch, tmp_path):
        import whisper_tm.config as cfg_mod
        from whisper_tm.config import ConfigManager

        monkeypatch.delenv("WHISPER_TM_CONFIG", raising=False)
        fake_home = tmp_path / "home-cfg"
        monkeypatch.setattr(cfg_mod, "user_config_dir", lambda name: str(fake_home))
        app = tmp_path / "app"
        app.mkdir()
        (app / "config.yaml").write_text(yaml.dump({"server": {"port": 5050}}))
        monkeypatch.chdir(app)
        cfg = ConfigManager()
        cfg.load()
        cfg.set("queue", "lanes", 2)
        target = cfg.save()
        assert target == (app / "config.yaml").resolve()
        assert not (fake_home / "config.yaml").exists()
        assert cfg.config_path == (app / "config.yaml").resolve()

    def test_only_what_differs_from_the_defaults(self, monkeypatch, tmp_path):
        from whisper_tm.config import ConfigManager

        monkeypatch.delenv("WHISPER_TM_CONFIG", raising=False)
        own = tmp_path / "config.yaml"
        own.write_text(yaml.dump({"server": {"port": 5050}, "queue": {"autostart": True}}))
        cfg = ConfigManager()
        cfg.load(own)
        cfg.set("queue", "lanes", 2)
        cfg.override("server.host", "0.0.0.0")  # this run only
        cfg.save()
        text = own.read_text()
        saved = yaml.safe_load(text)
        assert text.startswith("# Whisper Transcription Manager — your settings")
        # autostart was the default: no longer frozen in the file
        assert saved == {"server": {"port": 5050}, "queue": {"lanes": 2}}
        # And it reads back the same
        again = ConfigManager()
        again.load(own)
        assert again.get("queue", "lanes") == 2 and again.get("server", "port") == 5050
        assert again.get("queue", "autostart") is True

    def test_a_named_list_is_saved_whole(self, monkeypatch, tmp_path):
        """One template added in the summary dialog: the file held only that
        one, and loaded back it replaced the shipped list — every built-in
        template was gone after the next start. Deleting a built-in one
        didn't hold either: the file held nothing, the defaults came back."""
        from whisper_tm.config import ConfigManager

        monkeypatch.delenv("WHISPER_TM_CONFIG", raising=False)
        own = tmp_path / "config.yaml"
        own.write_text("")
        cfg = ConfigManager()
        cfg.load(own)
        shipped = dict(cfg.get("llm", "presets"))
        assert len(shipped) > 1
        cfg.set("llm", "presets", {**shipped, "Mine": "My own template."})
        cfg.save()
        again = ConfigManager()
        again.load(own)
        assert again.get("llm", "presets") == {**shipped, "Mine": "My own template."}

        fewer = dict(shipped)
        gone = next(iter(fewer))
        del fewer[gone]
        again.set("llm", "presets", fewer)
        again.save()
        third = ConfigManager()
        third.load(own)
        assert third.get("llm", "presets") == fewer

        # Unchanged, the list follows the defaults instead of being frozen
        third.set("llm", "presets", shipped)
        third.save()
        assert "presets" not in (yaml.safe_load(own.read_text()) or {}).get("llm", {})

    def test_all_defaults_is_an_empty_file(self, monkeypatch, tmp_path):
        from whisper_tm.config import ConfigManager

        monkeypatch.delenv("WHISPER_TM_CONFIG", raising=False)
        own = tmp_path / "config.yaml"
        own.write_text("")
        cfg = ConfigManager()
        cfg.load(own)
        cfg.save()
        assert yaml.safe_load(own.read_text()) in ({}, None)
