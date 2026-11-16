"""The Images & sounds service answers only the app: a key of each start
(passed by environment), a Host of this machine and a JSON body. A web page
could post to 127.0.0.1:8083 — text/plain needs no preflight — and make it
read files and write stills past the allowed roots."""

import pytest


class _Fake:
    """Just enough of the handler for ``_refused``."""

    def __init__(self, headers):
        self.headers = headers


@pytest.fixture
def refused(monkeypatch):
    np = pytest.importorskip("numpy")
    del np
    from whisper_tm.media import service

    monkeypatch.setenv("WHISPER_TM_MEDIA_TOKEN", "k" * 48)
    return lambda h: service.Handler._refused(_Fake(h))


GOOD = {"X-Media-Token": "k" * 48, "Host": "127.0.0.1:8083", "Content-Type": "application/json"}


def test_the_apps_request_passes(refused):
    assert refused(GOOD) == ""
    assert refused({**GOOD, "Host": "localhost:8083"}) == ""
    assert refused({**GOOD, "Host": "[::1]:8083"}) == ""


@pytest.mark.parametrize(
    "change,why",
    [
        ({"X-Media-Token": ""}, "Token"),
        ({"X-Media-Token": "x"}, "Token"),
        ({"Host": "evil.example:8083"}, "Host"),
        ({"Host": "192.0.2.7:8083"}, "Host"),
        ({"Content-Type": "text/plain"}, "JSON"),
    ],
)
def test_anything_else_is_refused(refused, change, why):
    assert why in refused({**GOOD, **change})


def test_without_a_key_of_its_start_nothing_passes(monkeypatch):
    pytest.importorskip("numpy")
    from whisper_tm.media import service

    monkeypatch.delenv("WHISPER_TM_MEDIA_TOKEN", raising=False)
    assert service.Handler._refused(_Fake(GOOD))


def test_the_app_sends_the_key():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "whisper_tm/core/media_server.py").read_text(
        encoding="utf-8"
    )
    assert 'env["WHISPER_TM_MEDIA_TOKEN"] = self._token' in src
    assert '"X-Media-Token": getattr(self, "_token", "")' in src
