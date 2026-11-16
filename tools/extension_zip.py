#!/usr/bin/env python3
"""The browser extension as zip files, for a release (``make extension-zip``).

Two files in ``dist/``, named after the manifest's version:

* ``whisper-tm-extension-<version>.zip`` — to unpack and load unpacked
  (chrome://extensions → Developer mode → Load unpacked). It keeps the
  manifest's ``key``: the extension gets its fixed ID, the one the server
  trusts (core/security.py: OUR_EXTENSIONS).
* ``whisper-tm-extension-<version>-store.zip`` — for the Chrome Web Store,
  which refuses a manifest with a ``key``. The store gives the extension an ID
  of its own: list it in ``security.trusted_extensions`` of the server's
  config.

The README stays out of both.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "browser-extension"
LEFT_OUT = {"README.md"}


def build(out_dir: Path = ROOT / "dist") -> list[Path]:
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    version = manifest["version"]
    files = sorted(
        p
        for p in SOURCE.rglob("*")
        if p.is_file() and p.name not in LEFT_OUT and not p.name.startswith(".")
    )
    store = {k: v for k, v in manifest.items() if k != "key"}
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for suffix, the_manifest in (("", manifest), ("-store", store)):
        target = out_dir / f"whisper-tm-extension-{version}{suffix}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                name = p.relative_to(SOURCE).as_posix()
                if name == "manifest.json":
                    z.writestr(name, json.dumps(the_manifest, indent=2, ensure_ascii=False) + "\n")
                else:
                    z.write(p, name)
        made.append(target)
    return made


if __name__ == "__main__":
    for path in build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"):
        print(path)
