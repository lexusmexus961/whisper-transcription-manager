"""
LLM queue: enqueueing, order, cancellation, restart resumption,
partial results — and provider resolution along with migration of old config.
"""

import pytest

from whisper_tm.config import config


@pytest.fixture
def app_and_queue(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Own config file per test: since config.save() writes back to the loaded
    # file (instead of just to home), tests that call llm_provider.save() directly
    # would otherwise fill the SESSION config from conftest — and every later
    # config.load() of any module would read the test providers again.
    own_ones = tmp_path / "config.yaml"
    # The transcription queue stays paused and on one lane, as in the
    # session config: create_app() below starts its workers, and three
    # running lanes outlived this test into tests/test_queue.py.
    own_ones.write_text("queue:\n  autostart: false\n  lanes: 1\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_TM_CONFIG", str(own_ones))
    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path)
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    app = create_app()
    from whisper_tm.core.llm_queue import llm_queue

    # Do NOT run worker — tests control the steps themselves
    llm_queue.stop()
    llm_queue._running = False
    with llm_queue._lock:
        llm_queue._current = None
    from whisper_tm.extensions import db

    with db.connection() as conn:
        conn.execute("DELETE FROM llm_queue")
    import copy

    llm_before = copy.deepcopy(config._data.get("llm"))
    yield app.test_client(), llm_queue
    # Reset llm section: providers/assignments/thresholds from these
    # tests must not leak into other modules — since task
    # assignment in the chat path takes effect, foreign code really reads these fields.
    if llm_before is None:
        config._data.pop("llm", None)
    else:
        config._data["llm"] = llm_before


def test_enqueue_and_order(app_and_queue):
    _c, q = app_and_queue
    a = q.enqueue("summary", "a.mp4", {"preset": "Short version"})
    b = q.enqueue("translation", "b.mp4", {"language": "English"})
    c_ = q.enqueue("correction", "c.mp4", {}, front=True)
    assert a and b and c_
    snap = q.snapshot()
    assert snap["counts"]["pending"] == 3
    # 'front' is drawn first
    assert q._next_one()["filename"] == "c.mp4"
    assert q._next_one()["filename"] == "a.mp4"
    assert q._next_one()["filename"] == "b.mp4"
    assert q._next_one() is None  # nothing more open


def test_snapshot_counts_beyond_the_entry_limit(app_and_queue):
    """The entry list is capped; the header counts must not be."""
    _c, q = app_and_queue
    for i in range(7):
        q.enqueue("summary", f"f{i}.mp4", {})
    snap = q.snapshot(limit=3)
    assert len(snap["items"]) == 3
    assert snap["counts"]["pending"] == 7


def test_unknown_kind(app_and_queue):
    _c, q = app_and_queue
    with pytest.raises(ValueError):
        q.enqueue("nonsense", "a.mp4")


def test_cancelling_a_pending_entry(app_and_queue):
    _c, q = app_and_queue
    i = q.enqueue("correction", "a.mp4")
    assert q.cancel(i) is True
    assert q.snapshot()["counts"]["cancelled"] == 1
    assert q._next_one() is None  # cancelled ones are not drawn


def test_cancelling_a_running_one_sets_the_flag(app_and_queue):
    _c, q = app_and_queue
    i = q.enqueue("correction", "a.mp4")
    q._next_one()
    with q._lock:
        q._current = {"id": i}
    assert q.cancel(i) is True
    assert q.abort_requested(i) is True
    assert q.abort_requested(i + 999) is False


def test_partial_results_and_progress(app_and_queue):
    _c, q = app_and_queue
    i = q.enqueue("summary", "a.mp4")
    q.progress(i, 42, "Section 2/4")
    q.partial_result(i, {"number": 1, "total": 4, "text": "Part A"})
    q.partial_result(i, {"number": 2, "total": 4, "text": "Part B"})
    e = q.entry(i)
    assert e["progress"] == 42 and "2/4" in e["message"]
    assert [t["text"] for t in e["partial_results"]] == ["Part A", "Part B"]


def test_a_restart_picks_running_ones_up_again(app_and_queue):
    _c, q = app_and_queue
    i = q.enqueue("correction", "a.mp4")
    q._next_one()  # → running
    assert q.entry(i)["status"] == "running"
    q._cleanup_after_restart()  # simulated restart
    assert q.entry(i)["status"] == "pending"


def test_retry_and_clear(app_and_queue):
    _c, q = app_and_queue
    i = q.enqueue("correction", "a.mp4")
    q._update(i, status="error", error="broken")
    assert q.retry(i) is True
    assert q.entry(i)["status"] == "pending"
    q._update(i, status="done")
    assert q.clear("done") == 1
    assert q.entry(i) is None


def test_the_open_entry_finds_the_running_task(app_and_queue):
    """The UI hooks into a job that was already running before the
    dialog was opened (closing doesn't cancel anything)."""
    _c, q = app_and_queue
    i = q.enqueue("summary", "a.mp4")
    assert q.open_entry("summary", "a.mp4")["id"] == i
    assert q.open_entry("correction", "a.mp4") is None
    q._update(i, status="done")
    assert q.open_entry("summary", "a.mp4") is None


# ── API ────────────────────────────────────────────────────────────────────


def test_api_add_and_state(app_and_queue, monkeypatch):
    c, _q = app_and_queue
    config.set("llm", "endpoint", "http://x/v1")
    r = c.post("/api/llm/queue/add", json={"kind": "summary", "filenames": ["a.mp4", "b.mp4"]})
    assert r.status_code == 200 and r.get_json()["added"] == 2
    snap = c.get("/api/llm/queue").get_json()
    assert snap["counts"]["pending"] == 2


def test_api_add_without_an_llm_configuration(app_and_queue):
    c, _q = app_and_queue
    config.set("llm", "endpoint", "")
    r = c.post("/api/llm/queue/add", json={"kind": "correction", "filename": "a.mp4"})
    assert r.status_code == 400
    assert "not configured" in r.get_json()["error"]


def test_api_add_unknown_kind(app_and_queue):
    c, _q = app_and_queue
    r = c.post("/api/llm/queue/add", json={"kind": "xxx", "filename": "a.mp4"})
    assert r.status_code == 400


# ── Provider ──────────────────────────────────────────────────────────────


def test_provider_migration_from_the_old_config(app_and_queue):
    """A config with the old individual fields appears as provider
    "Standard" — existing users don't have to change anything."""
    _c, _q = app_and_queue
    from whisper_tm.core import llm_provider as lp

    config._data["llm"].pop("providers", None)
    config.set("llm", "endpoint", "http://localhost:9999/v1")
    config.set("llm", "model", "qwen3")
    ps = lp.all_items()
    assert ps[0].id == "standard" and ps[0].kind == "custom"
    assert ps[0].default_model == "qwen3"


def test_the_local_provider_comes_with_an_installed_llama_server(
    app_and_queue, monkeypatch, tmp_path
):
    """The bundled llama.cpp server is offered even with nothing configured
    — otherwise a fresh system would have no provider even though the setup
    brings one along. A matching endpoint is not listed twice."""
    _c, _q = app_and_queue
    from whisper_tm.core import llama_server
    from whisper_tm.core import llm_provider as lp

    program = tmp_path / "llama-server"
    program.write_text("")
    monkeypatch.setattr(llama_server, "binary", lambda: program)
    config._data["llm"].pop("providers", None)
    config.set("llm", "endpoint", "")
    ids = [p.id for p in lp.all_items()]
    assert ids == ["local"]
    # Configured provider on the same endpoint → no duplicate
    lp.save(
        [{"id": "own", "name": "Own", "kind": "llamacpp", "endpoint": llama_server.endpoint()}],
        in_file=False,
    )
    ids2 = [p.id for p in lp.all_items()]
    assert ids2 == ["own"]


def test_without_a_llama_server_a_deleted_local_provider_stays_deleted(app_and_queue, monkeypatch):
    """The automatic provider was added on EVERY read, program or not. On a
    machine without llama.cpp it pointed at a server that can never answer
    — and deleting it in the settings never stuck, it was back on the next
    load. A provider saved in the config stays exactly as saved."""
    _c, _q = app_and_queue
    from whisper_tm.core import llama_server
    from whisper_tm.core import llm_provider as lp

    monkeypatch.setattr(llama_server, "binary", lambda: None)
    config._data["llm"].pop("providers", None)
    config.set("llm", "endpoint", "")
    config.set("llm", "api_key", "")
    assert lp.all_items() == []
    assert lp._local_provider() is None

    # Saved (the local one included), the list stays untouched …
    saved = [
        {
            "id": "local",
            "name": "Local (llama.cpp)",
            "kind": "llamacpp",
            "endpoint": "http://127.0.0.1:8080/v1",
        },
        {"id": "oai", "name": "OpenAI", "kind": "openai", "endpoint": "https://api.openai.com/v1"},
    ]
    lp.save(saved, in_file=False)
    assert [p.id for p in lp.all_items()] == ["local", "oai"]
    # … and a deleted one does not come back
    lp.save(saved[1:], in_file=False)
    assert [p.id for p in lp.all_items()] == ["oai"]


def test_saving_a_provider_keeps_the_api_key(app_and_queue):
    """The UI never sends keys back — an empty field must not delete
    the stored key."""
    _c, _q = app_and_queue
    from whisper_tm.core import llm_provider as lp

    lp.save(
        [
            {
                "id": "a",
                "name": "A",
                "kind": "openai",
                "endpoint": "https://api.openai.com/v1",
                "api_key": "sk-secret",
                "default_model": "gpt-4o-mini",
            }
        ]
    )
    lp.save(
        [
            {
                "id": "a",
                "name": "A renamed",
                "kind": "openai",
                "endpoint": "https://api.openai.com/v1",
                "api_key": "",
                "default_model": "gpt-4o-mini",
            }
        ]
    )
    p = lp.all_items()[0]
    assert p.name == "A renamed" and p.api_key == "sk-secret"
    assert p.as_dict()["has_key"] is True
    assert "api_key" not in p.as_dict()  # never delivered to the UI


def test_provider_for_a_task(app_and_queue):
    _c, _q = app_and_queue
    from whisper_tm.core import llm_provider as lp

    lp.save(
        [
            {
                "id": "local",
                "name": "Local",
                "kind": "llamacpp",
                "endpoint": "http://localhost:8080/v1",
                "default_model": "qwen",
            },
            {
                "id": "oai",
                "name": "OpenAI",
                "kind": "openai",
                "endpoint": "https://api.openai.com/v1",
                "default_model": "gpt-4o",
            },
        ]
    )
    config.set("llm", "tasks", {"correction": {"provider": "oai", "model": "gpt-4o-mini"}})
    p, m = lp.for_task("correction")
    assert p.id == "oai" and m == "gpt-4o-mini"
    # Without an assignment → default (first active one) with its model
    p2, m2 = lp.for_task("translation")
    assert p2.id == "local" and m2 == "qwen"


def test_deleting_a_provider_clears_the_legacy_fields(app_and_queue):
    """Deleting all providers left llm.endpoint/
    api_key in place — the LLM kept sending (including the key) to the
    endpoint that was just deleted, with no UI way to clear it."""
    _c, _q = app_and_queue
    from whisper_tm.core import llm_provider as lp

    lp.save(
        [
            {
                "id": "a",
                "name": "A",
                "kind": "openai",
                "endpoint": "https://api.openai.com/v1",
                "api_key": "sk-secret",
                "default_model": "gpt-4o",
            }
        ]
    )
    assert config.get("llm", "endpoint", default="")
    lp.save([])  # last card deleted
    assert config.get("llm", "endpoint", default="") == ""
    assert config.get("llm", "api_key", default="") == ""
    assert config.get("llm", "model", default="") == ""
    # Deactivating (instead of deleting) clears it the same way
    lp.save(
        [
            {
                "id": "a",
                "name": "A",
                "kind": "openai",
                "endpoint": "https://api.openai.com/v1",
                "api_key": "sk-secret",
                "default_model": "gpt-4o",
            }
        ]
    )
    lp.save(
        [
            {
                "id": "a",
                "name": "A",
                "kind": "openai",
                "endpoint": "https://api.openai.com/v1",
                "active": False,
                "default_model": "gpt-4o",
            }
        ]
    )
    assert config.get("llm", "endpoint", default="") == ""


def test_an_empty_provider_model_protects_llm_model(app_and_queue):
    """A card without default_model (never
    clicked "Test") overwrote a working llm.model with ''."""
    _c, _q = app_and_queue
    from whisper_tm.core import llm_provider as lp

    config.set("llm", "model", "qwen3:32b")
    lp.save(
        [
            {
                "id": "a",
                "name": "A",
                "kind": "ollama",
                "endpoint": "http://localhost:11434",
                "default_model": "",
            }
        ]
    )
    assert config.get("llm", "model", default="") == "qwen3:32b"


def test_the_task_mapping_takes_effect_in_the_chat_payload(app_and_queue, monkeypatch):
    """llm.tasks was saved but not read by any
    execution path. Now build_chat_payload pulls the assignment via the
    task parameter."""
    _c, _q = app_and_queue
    from whisper_tm.core import llm_client
    from whisper_tm.core import llm_provider as lp

    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 32768)
    lp.save(
        [
            {
                "id": "local",
                "name": "Local",
                "kind": "llamacpp",
                "endpoint": "http://localhost:8080/v1",
                "default_model": "qwen",
            },
            {
                "id": "oai",
                "name": "OpenAI",
                "kind": "openai",
                "endpoint": "https://api.openai.com/v1",
                "api_key": "sk-x",
                "default_model": "gpt-4o-mini",
            },
        ]
    )
    config.set("llm", "tasks", {"translation": {"provider": "oai", "model": "gpt-4o"}})
    msgs = [{"role": "user", "content": "hi"}]
    # Assigned task → provider + model from the assignment
    url, headers, payload, _n = llm_client.build_chat_payload(msgs, task="translation")
    assert url.startswith("https://api.openai.com/v1")
    assert payload["model"] == "gpt-4o"
    assert headers.get("Authorization") == "Bearer sk-x"
    # Explicit model argument overrides the assignment
    _u2, _h2, p2, _n2 = llm_client.build_chat_payload(msgs, task="translation", model="gpt-4o-mini")
    assert p2["model"] == "gpt-4o-mini"
    # Task without an assignment → default provider (first active one)
    u3, _h3, p3, _n3 = llm_client.build_chat_payload(msgs, task="correction")
    assert u3.startswith("http://localhost:8080/v1")
    assert p3["model"] == "qwen"


def test_api_provider_roundtrip(app_and_queue):
    c, _q = app_and_queue
    r = c.post(
        "/api/llm/providers",
        json={
            "providers": [
                {
                    "id": "x",
                    "name": "X",
                    "kind": "ollama",
                    "endpoint": "http://localhost:11434",
                    "default_model": "llama3",
                }
            ]
        },
    )
    assert r.status_code == 200
    d = c.get("/api/llm/providers").get_json()
    assert d["providers"][0]["name"] == "X"
    assert {t["id"] for t in d["types"]} >= {"llamacpp", "ollama", "openai"}


def test_api_provider_duplicate_ids(app_and_queue):
    c, _q = app_and_queue
    r = c.post(
        "/api/llm/providers",
        json={
            "providers": [
                {"id": "x", "name": "A", "endpoint": "http://a"},
                {"id": "x", "name": "B", "endpoint": "http://b"},
            ]
        },
    )
    assert r.status_code == 400


# ── Folder batch ─────────────────────────────────────────────────────────


def test_the_folder_preview_counts_only_transcribed_ones(app_and_queue):
    c, _q = app_and_queue
    from whisper_tm.extensions import db

    basis = str(config.media_dir).rstrip("/")
    with db.connection() as conn:
        for rel, done, words in [
            ("a.mp4", 1, 5),
            ("below/b.mp4", 1, 7),
            ("c.mp4", 0, 3),
            ("silence.wav", 1, 0),
        ]:
            conn.execute(
                "INSERT INTO transcripts (rel_path, abs_path, complete, "
                "json_data, word_count) VALUES (?, ?, ?, '{}', ?)",
                (rel, f"{basis}/{rel}", done, words),
            )
    d = c.post(
        "/api/llm/queue/folder",
        json={"kind": "correction", "dry_run_only": True, "skip_existing": False},
    ).get_json()
    assert d["found"] == 2  # c.mp4 is not finished
    # Without a word every task fails: left out, and said so
    assert d["empty"] == 1 and "silence.wav" not in d["examples"]
    # Non-recursive: only the file at the root
    d2 = c.post(
        "/api/llm/queue/folder",
        json={
            "kind": "correction",
            "dry_run_only": True,
            "recursive": False,
            "skip_existing": False,
        },
    ).get_json()
    assert d2["found"] == 1


# ── Provider disappeared after saving ────────────────────────────────


def test_the_provider_survives_saving_the_settings(app_and_queue):
    """Providers saved via PUT /api/settings must not get lost — the
    big "Save" button sends them along in the llm section.
    """
    c, _q = app_and_queue
    r = c.put(
        "/api/settings",
        json={
            "llm": {
                "providers": [
                    {
                        "id": "local",
                        "name": "Local (llama.cpp)",
                        "kind": "llamacpp",
                        "endpoint": "http://127.0.0.1:8080/v1",
                    },
                    {
                        "id": "oai",
                        "name": "OpenAI",
                        "kind": "openai",
                        "endpoint": "https://api.openai.com/v1",
                        "api_key": "sk-new",
                        "default_model": "gpt-4o-mini",
                    },
                ]
            }
        },
    )
    assert r.status_code == 200, r.get_json()
    d = c.get("/api/llm/providers").get_json()
    names = [p["name"] for p in d["providers"]]
    assert "OpenAI" in names, f"OpenAI missing: {names}"
    oai = next(p for p in d["providers"] if p["id"] == "oai")
    assert oai["has_key"] is True


def test_the_api_key_survives_a_second_save_without_a_key(app_and_queue):
    """A second save without a key entry must not delete the key —
    not even via the masking placeholder."""
    from whisper_tm.config import SECRET_PLACEHOLDER

    c, _q = app_and_queue
    c.put(
        "/api/settings",
        json={
            "llm": {
                "providers": [
                    {
                        "id": "oai",
                        "name": "OpenAI",
                        "kind": "openai",
                        "endpoint": "https://api.openai.com/v1",
                        "api_key": "sk-secret",
                    }
                ]
            }
        },
    )
    # a) field empty
    c.put(
        "/api/settings",
        json={
            "llm": {
                "providers": [
                    {
                        "id": "oai",
                        "name": "OpenAI",
                        "kind": "openai",
                        "endpoint": "https://api.openai.com/v1",
                        "api_key": "",
                    }
                ]
            }
        },
    )
    from whisper_tm.core import llm_provider as lp

    assert next(p for p in lp.all_items() if p.id == "oai").api_key == "sk-secret"
    # b) field carries the masking placeholder from GET /api/settings
    c.put(
        "/api/settings",
        json={
            "llm": {
                "providers": [
                    {
                        "id": "oai",
                        "name": "OpenAI",
                        "kind": "openai",
                        "endpoint": "https://api.openai.com/v1",
                        "api_key": SECRET_PLACEHOLDER,
                    }
                ]
            }
        },
    )
    assert next(p for p in lp.all_items() if p.id == "oai").api_key == "sk-secret"


def test_saving_the_settings_allows_other_llm_values(app_and_queue):
    """Special-casing the provider must not swallow the other llm fields."""
    c, _q = app_and_queue
    r = c.put(
        "/api/settings",
        json={
            "llm": {
                "providers": [{"id": "x", "name": "X", "endpoint": "http://x/v1"}],
                "max_tokens": 2048,
                "correction": {"score_threshold": 0.55},
            }
        },
    )
    assert r.status_code == 200
    assert config.get("llm", "max_tokens") == 2048
    assert config.get("llm", "correction", "score_threshold") == 0.55


def test_endpoint_normalization_heals_doc_urls(app_and_queue):
    """The model-listing URL had been entered as
    the endpoint (https://api.openai.com/v1/models) — every chat call
    then ran against /v1/models/chat/completions and hit nothing."""
    from whisper_tm.core import llm_provider as lp

    lp.save(
        [
            {
                "id": "a",
                "name": "A",
                "kind": "openai",
                "endpoint": "https://api.openai.com/v1/models",
            },
            {
                "id": "b",
                "name": "B",
                "kind": "openai",
                "endpoint": "https://api.openai.com/v1/chat/completions",
            },
            {"id": "c", "name": "C", "kind": "llamacpp", "endpoint": "http://192.0.2.7:8080/v1/"},
        ],
        in_file=False,
    )
    eps = {p.id: p.endpoint for p in lp.all_items()}
    assert eps["a"] == "https://api.openai.com/v1"
    assert eps["b"] == "https://api.openai.com/v1"
    assert eps["c"] == "http://192.0.2.7:8080/v1"


def test_openai_reasoning_payload(app_and_queue):
    """gpt-5*/o*-models require max_completion_tokens and reject
    temperature; classic models still get max_tokens.
    And: without a recognizable context, max_tokens is hard-capped
    (256000 went to OpenAI unthrottled → 400)."""
    from whisper_tm.core import llm_client

    config.set("llm", "endpoint", "https://api.openai.com/v1")
    config.set("llm", "max_tokens", 256000)
    for model, field_name in (
        ("gpt-5-nano", "max_completion_tokens"),
        ("o3-mini", "max_completion_tokens"),
        ("gpt-4.1-nano", "max_tokens"),
        ("gpt-4o-mini", "max_tokens"),
    ):
        _url, _h, payload, _n = llm_client.build_chat_payload(
            [{"role": "user", "content": "x"}], model=model
        )
        assert field_name in payload, (model, payload.keys())
        assert payload[field_name] <= 16384  # cap without context
        if field_name == "max_completion_tokens":
            assert "max_tokens" not in payload
            assert "temperature" not in payload
        else:
            assert "temperature" in payload


def _summary_on_openai():
    """Two providers; summaries are assigned to the second one."""
    from whisper_tm.core import llm_provider as lp

    lp.save(
        [
            {
                "id": "local",
                "name": "Local",
                "kind": "llamacpp",
                "endpoint": "http://localhost:8080/v1",
                "default_model": "qwen",
            },
            {
                "id": "oai",
                "name": "OpenAI",
                "kind": "openai",
                "endpoint": "https://api.openai.com/v1",
                "api_key": "sk-x",
                "default_model": "gpt-4o-mini",
            },
        ],
        in_file=False,
    )
    config.set("llm", "tasks", {"summary": {"provider": "oai", "model": "gpt-4.1-nano"}})


def test_llm_status_names_what_the_default_runs_on(app_and_queue, monkeypatch):
    """The dialogs' default entry sends no model, so the task assignment
    decides. Its label comes from here and must name the model the
    assignment picks — not the legacy llm.model the entry showed (and, in
    the summarize dialogs, SENT) before, which the assignment need not use:
    a named model pulls its own provider along."""
    c, _q = app_and_queue
    from whisper_tm.core import llm_client

    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 32768)
    _summary_on_openai()
    config.set("llm", "correction", "score_threshold", 0.55)
    d = c.get("/api/llm/status").get_json()
    assert d["model"] == "qwen"  # the legacy field
    assert d["task_models"] == {
        "summary": "gpt-4.1-nano",
        "correction": "qwen",
        "translation": "qwen",
        "chat": "qwen",
    }
    # The correction dialog fills its threshold field with this
    assert d["correction_score_threshold"] == 0.55


def test_the_llm_input_view_shows_where_the_run_goes(app_and_queue, monkeypatch):
    """The summarize dialog's "LLM input" tab showed llm.endpoint and
    llm.model. With the default now leaving the model to the summary
    assignment, that would name a server the run doesn't use."""
    c, _q = app_and_queue
    from whisper_tm.core import llm_client

    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 32768)
    _summary_on_openai()
    body = {"segments": [{"start": 0, "end": 2, "text": "Hello there."}]}
    d = c.post("/api/llm/debug", json=body).get_json()
    assert d["api_url"].startswith("https://api.openai.com/v1")
    assert d["model"] == "gpt-4.1-nano"
    # A model named in the dialog pulls its own provider along
    d2 = c.post("/api/llm/debug", json={**body, "model": "qwen"}).get_json()
    assert d2["api_url"].startswith("http://localhost:8080/v1")
    assert d2["model"] == "qwen"


class _Reply:
    status_code = 200
    text = ""

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_a_summary_without_a_model_records_the_assigned_one(app_and_queue, monkeypatch):
    """No model chosen → the summary assignment picks one, and the saved
    summary must name THAT one. It recorded the legacy llm.model, which was
    only right while the dialogs sent that name."""
    _c, _q = app_and_queue
    from whisper_tm.core import llm_client
    from whisper_tm.core import summarize as s

    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 32768)
    _summary_on_openai()
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, model=json["model"])
        return _Reply({"choices": [{"message": {"content": "The summary."}}]})

    monkeypatch.setattr(s.requests, "post", fake_post)
    saved = {}
    monkeypatch.setattr(s, "load_summaries", lambda f, media_dir=None, **_: [])
    monkeypatch.setattr(
        s, "save_summaries", lambda f, lst, media_dir=None: saved.update(listing=lst) or True
    )

    assert s.make_summary("a.mp4", segments=[{"text": "Hello"}]) == "The summary."
    assert sent["url"].startswith("https://api.openai.com/v1")
    assert sent["model"] == "gpt-4.1-nano"
    assert saved["listing"][-1]["model"] == "gpt-4.1-nano"


def test_llm_status_returns_status_shape(app_and_queue):
    """Regression: a helper function that had slipped in between
    hijacked the /api/llm/status decorator — the route suddenly
    returned a LIST instead of the status object, and no test noticed."""
    c, _q = app_and_queue
    from whisper_tm.core import llm_provider as lp

    lp.save(
        [
            {
                "id": "oai",
                "name": "openai",
                "kind": "openai",
                "endpoint": "https://api.openai.com/v1",
                "default_model": "gpt-4.1-nano",
            }
        ],
        in_file=False,
    )
    d = c.get("/api/llm/status").get_json()
    assert isinstance(d, dict) and "configured" in d and "models" in d
    assert {"model": "gpt-4.1-nano", "providers": "openai"} in d["provider_models"]


class _NoteQueue:
    """The calls a task makes on the queue, answered without one."""

    def progress(self, *a):
        pass

    def partial_result(self, *a):
        pass

    def partial_text(self, *a):
        pass

    def abort_requested(self, *a):
        return False


def test_a_summary_names_the_speakers_as_the_file_view_does(app_and_queue, monkeypatch):
    """A speaker named in the file view reached the LLM as "SPEAKER_00", so
    the summary spoke of "Speaker_00" next to a transcript showing the name."""
    _c, _q = app_and_queue
    from whisper_tm.api import transcription
    from whisper_tm.core import llm_tasks
    from whisper_tm.core import summarize as s
    from whisper_tm.extensions import db

    # The route module keeps the database of the app it was imported with
    monkeypatch.setattr(transcription, "db", db)
    media = str(config.media_dir)
    db.upsert_transcript(
        "a.mp4",
        {
            "complete": True,
            "speaker_names": {"SPEAKER_00": "Alice"},
            "segments": [
                {"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hi."},
                {"start": 1, "end": 2, "speaker": "SPEAKER_01", "text": "Ho."},
            ],
        },
        media_dir=media,
        abs_path=media + "/a.mp4",
    )
    seen = []

    def fake_stream(segments, **kw):
        seen.append(s.format_transcript(segments, kw["include_speakers"], kw["include_timestamps"]))
        yield {"text": "The summary."}

    monkeypatch.setattr(s, "summary_stream", fake_stream)
    monkeypatch.setattr(s, "change_summaries", lambda *a, **kw: (True, None))

    entry = {
        "id": 1,
        "filename": "a.mp4",
        "params": {
            "include_speakers": True,
            "include_timestamps": False,
            "system_prompt": "Sum up.",
        },
    }
    assert llm_tasks._summary(entry, _NoteQueue())["text"] == "The summary."
    assert seen == ["[Alice] Hi.\n[SPEAKER_01] Ho."]
    # The same file from its own folder (the extension's uploads)
    entry["params"]["media_dir"] = media
    llm_tasks._summary(entry, _NoteQueue())
    assert seen[-1] == "[Alice] Hi.\n[SPEAKER_01] Ho."
    # A correction writes the segments back: it keeps the labels
    assert [x["speaker"] for x in llm_tasks._load_segments("a.mp4")] == ["SPEAKER_00", "SPEAKER_01"]
