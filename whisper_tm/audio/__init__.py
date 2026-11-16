"""Audio analysis: what every recording sounds like, kept once.

One pass over a recording's sound — while it is transcribed, or later by a
background run for recordings transcribed before — keeps:

* **fingerprints** (fingerprint.py, index.py): pairs of spectral peaks,
  Shazam-style, so a snippet of a few seconds finds every recording that
  contains it, at the exact time, also re-encoded, quieter or with noise
  over it — and recordings that are the same, or contain one another, fall
  out of it (store.py, ``relations``);
* **the waveform** (levels.py): 20 values a second, the clip editor's
  timeline without decoding the file again;
* **loudness** (levels.py): integrated loudness and loudness range (EBU
  R128), the sample peak, how much is near silence, how much is clipped.

The parts:

* fingerprint.py — the peaks and their hashes, fed in pieces;
* levels.py — waveform and loudness, fed in pieces;
* index.py — the hash index on disk: immutable segments, merged in the
  background;
* store.py — ``audio.db`` (one row per recording) and the segments;
* analyze.py — decoding a recording or a WAV, and the search for a snippet;
* runner.py — the background run over the library.

Everything runs on the CPU: the GPU stays the queue's.
"""
