"""The voice model without code from the network (core/speaker.py).

ReDimNet was loaded with torch.hub.load("IDRnD/ReDimNet:master",
trust_repo=True): whatever was on the branch at first use ran. Its code is
kept in the package now (whisper_tm/vendor/redimnet), and the weights —
from a release named "latest" — are checked against their SHA-256 before
anything of them is unpickled.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from whisper_tm.core import speaker  # noqa: E402


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    (tmp_path / "checkpoints").mkdir()
    return tmp_path / "checkpoints" / speaker.REDIMNET_URL.rsplit("/", 1)[1]


def test_weights_that_are_not_the_expected_ones_are_refused(hub):
    torch.save({"model_config": {}, "state_dict": {}}, hub)
    with pytest.raises(RuntimeError, match="not the expected ones"):
        speaker._redimnet_weights()


def test_the_expected_weights_load_as_plain_tensors(hub, monkeypatch):
    torch.save({"model_config": {"F": 72}, "state_dict": {"w": torch.zeros(2)}}, hub)
    monkeypatch.setattr(speaker, "REDIMNET_SHA256", hashlib.sha256(hub.read_bytes()).hexdigest())
    state = speaker._redimnet_weights()
    assert state["model_config"] == {"F": 72} and state["state_dict"]["w"].shape == (2,)


def test_a_missing_file_is_downloaded_with_its_hash(hub, monkeypatch):
    seen = {}

    def download(url, dst, hash_prefix=None, progress=True):
        seen.update(url=url, hash=hash_prefix)
        torch.save({"model_config": {}, "state_dict": {}}, dst)

    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    with pytest.raises(RuntimeError):  # the stand-in isn't the real file
        speaker._redimnet_weights()
    assert seen == {"url": speaker.REDIMNET_URL, "hash": speaker.REDIMNET_SHA256}


def test_no_code_from_the_network():
    import inspect

    src = inspect.getsource(speaker.load_model)
    assert "torch.hub.load(" not in src and "whisper_tm.vendor.redimnet.model" in src


REAL = Path(torch.hub.get_dir()) / "checkpoints" / "M-vb2-ptn.pt"


@pytest.mark.skipif(not REAL.exists(), reason="the real weights are not downloaded here")
def test_the_vendored_model_takes_the_real_weights(monkeypatch):
    from whisper_tm.vendor.redimnet.model import ReDimNetWrap

    # The weights of this machine, where they were found when the file was
    # read. During the run the cache folder is the session's throwaway one:
    # without this the test downloaded the weights again on every run.
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(REAL.parent.parent))
    state = speaker._redimnet_weights()
    m = ReDimNetWrap(**state["model_config"])
    loaded = m.load_state_dict(state["state_dict"])
    assert not loaded.missing_keys and not loaded.unexpected_keys
    m.eval()
    with torch.no_grad():
        v = m(torch.zeros(1, 16000)).numpy().ravel()
    assert v.shape == (192,)
