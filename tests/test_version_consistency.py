"""One version for the whole project: the package, pyproject and the
browser extension (an extension works with the server of its version)."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_package_pyproject_and_extension_agree():
    from whisper_tm import __version__

    pyproject = re.search(
        r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M
    )
    manifest = json.loads((ROOT / "browser-extension/manifest.json").read_text(encoding="utf-8"))
    assert pyproject and pyproject.group(1) == __version__ == manifest["version"]
