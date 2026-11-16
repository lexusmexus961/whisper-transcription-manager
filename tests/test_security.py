"""
Tests for core/security.py — path traversal protection.
"""

import pytest
from flask import Flask

from whisper_tm.core.security import (
    safe_path,
    safe_path_or_none,
)


@pytest.fixture
def app():
    """Minimal Flask app for abort() calls."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def base_dir(tmp_path):
    """Temporary base directory with test structure."""
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file.txt").write_text("test")
    (tmp_path / "file.txt").write_text("root")
    return tmp_path


class TestSafePath:
    """Path traversal protection tests."""

    def test_normal_file(self, app, base_dir):
        with app.test_request_context():
            result = safe_path(base_dir, "file.txt")
            assert result == base_dir / "file.txt"

    def test_subdirectory_file(self, app, base_dir):
        with app.test_request_context():
            result = safe_path(base_dir, "subdir/file.txt")
            assert result == base_dir / "subdir" / "file.txt"

    def test_traversal_dotdot(self, app, base_dir):
        """../../etc/passwd must be rejected."""
        with app.test_request_context(), pytest.raises(Exception):  # abort(403)
            safe_path(base_dir, "../../etc/passwd")

    def test_traversal_absolute(self, app, base_dir):
        """Absolute paths must be rejected if outside."""
        with app.test_request_context(), pytest.raises(Exception):
            safe_path(base_dir, "/etc/passwd")

    def test_traversal_similar_prefix(self, app, base_dir):
        """CRITICAL: /media/data-evil must not pass /media/data.

        This was the bug with str.startswith() — now fixed with relative_to().
        """
        # Create a sibling directory with a similar prefix
        evil_dir = base_dir.parent / (base_dir.name + "-evil")
        evil_dir.mkdir(exist_ok=True)
        evil_file = evil_dir / "secret.txt"
        evil_file.write_text("evil")

        with app.test_request_context():
            # Relative path that escapes to the sibling dir
            relative_evil = f"../{base_dir.name}-evil/secret.txt"
            with pytest.raises(Exception):
                safe_path(base_dir, relative_evil)

    def test_null_bytes(self, app, base_dir):
        """Null bytes must be removed."""
        with app.test_request_context():
            result = safe_path(base_dir, "file\x00.txt")
            assert "file" in str(result)

    def test_empty_path(self, app, base_dir):
        with app.test_request_context(), pytest.raises(Exception):
            safe_path(base_dir, "")

    def test_none_path(self, app, base_dir):
        with app.test_request_context(), pytest.raises(Exception):
            safe_path(base_dir, None)

    def test_symlink_escape(self, app, base_dir):
        """Symlinks pointing outside must be rejected."""
        link = base_dir / "escape_link"
        try:
            link.symlink_to("/tmp")
            with app.test_request_context(), pytest.raises(Exception):
                safe_path(base_dir, "escape_link/something")
        except OSError:
            pytest.skip("Symlinks not supported")


class TestSafePathOrNone:
    """safe_path_or_none returns None instead of aborting."""

    def test_normal(self, base_dir):
        result = safe_path_or_none(base_dir, "file.txt")
        assert result == base_dir / "file.txt"

    def test_traversal(self, base_dir):
        result = safe_path_or_none(base_dir, "../../etc/passwd")
        assert result is None

    def test_similar_prefix_attack(self, base_dir):
        """Same bug test as above, but for the _or_none variant."""
        evil_dir = base_dir.parent / (base_dir.name + "-evil")
        evil_dir.mkdir(exist_ok=True)
        (evil_dir / "secret.txt").write_text("evil")

        result = safe_path_or_none(base_dir, f"../{base_dir.name}-evil/secret.txt")
        assert result is None

    def test_empty(self, base_dir):
        assert safe_path_or_none(base_dir, "") is None

    def test_none(self, base_dir):
        assert safe_path_or_none(base_dir, None) is None
