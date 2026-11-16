# ReDimNet (vendored)

The model code of [IDRnD/ReDimNet](https://github.com/IDRnD/ReDimNet) at commit
`dd9f51948fc34f5f0479527db2d19fc7088ac9ab` (its code unchanged since September
2025), MIT-licensed (`LICENSE`). It used to be fetched with `torch.hub.load(...,
"master", trust_repo=True)` — whatever was on the branch at first use ran. Kept
here instead, unchanged but for the imports: `redimnet.layers…` became relative
(`.layers…`), and `layers/` got an `__init__.py`.

Two files hold code under the Apache License 2.0 (`LICENSE-APACHE-2.0`):
`layers/poolings.py` (from WeSpeaker, with its header) and the parts of
`layers/attention.py` copied from Hugging Face transformers v4.26.1 — the
notice at the top of that file was added here and names their copyright.

The weights are downloaded once from the project's release and checked against
their SHA-256 (`whisper_tm/core/speaker.py`).
