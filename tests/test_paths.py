"""
Tests for core/paths.py — the canonical path normal form.

The model is in docs/PATHS.md: abs_path is the identity. These tests
hold the normal form constant so it doesn't diverge again.
"""

import pytest

from whisper_tm.core import paths


class TestCanon:
    def test_already_canonical(self):
        assert paths.canon("/nas/video/a.mp4") == "/nas/video/a.mp4"

    def test_doubled_separators(self):
        assert paths.canon("/nas//video///a.mp4") == "/nas/video/a.mp4"

    def test_dot_and_double_dot(self):
        assert paths.canon("/nas/./video/../video/a.mp4") == "/nas/video/a.mp4"

    def test_trailing_slash(self):
        assert paths.canon("/nas/video/") == "/nas/video"

    def test_the_root_stays(self):
        assert paths.canon("/") == "/"

    def test_a_backslash_is_a_name_character_on_posix(self):
        """On Linux, "\\" is a valid character in a filename.
        Treating it as a separator would invent a directory level and thus
        a wrong identity."""
        assert paths.canon("/nas/a\\b.mp4") == "/nas/a\\b.mp4"

    def test_empty(self):
        assert paths.canon("") == ""
        assert paths.canon(None) == ""
        assert paths.canon("   ") == ""

    def test_a_pathlib_input(self, tmp_path):
        assert paths.canon(tmp_path) == str(tmp_path).rstrip("/")

    def test_unicode_and_spaces(self):
        assert paths.canon("/nas/Ünïcode  dïr/föö bär.mp4") == "/nas/Ünïcode  dïr/föö bär.mp4"

    def test_percent_and_underscore_stay(self):
        """Special characters are normal name components, not wildcards."""
        assert paths.canon("/nas/50%_data/a.mp4") == "/nas/50%_data/a.mp4"

    def test_no_resolve(self, tmp_path):
        """Symlinks are NOT resolved — otherwise every normalization
        would be a filesystem access (NAS!) and depend on mount state."""
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target)
        assert paths.canon(link) == str(link)

    def test_a_path_that_does_not_exist(self):
        """Normalization is pure string arithmetic, no filesystem access."""
        assert paths.canon("/does/not/exist/a.mp4") == "/does/not/exist/a.mp4"


class TestCanonWindows:
    def test_a_drive_letter(self):
        assert paths.canon("C:\\video\\a.mp4") == "C:/video/a.mp4"

    def test_a_drive_root_keeps_its_slash(self):
        assert paths.canon("C:\\") == "C:/"

    def test_the_drive_is_normalized(self):
        assert paths.canon("C:\\video\\..\\video\\a.mp4") == "C:/video/a.mp4"

    def test_unc(self):
        assert paths.canon("\\\\server\\share\\a.mp4") == "//server/share/a.mp4"


class TestIsInside:
    def test_directly_below(self):
        assert paths.is_inside("/nas/video/a.mp4", "/nas/video")

    def test_deeper(self):
        assert paths.is_inside("/nas/video/series/f1.mp4", "/nas/video")

    def test_the_root_itself_is_inside(self):
        assert paths.is_inside("/nas/video", "/nas/video")

    def test_next_to_it_is_not_below_it(self):
        """The core: /nas/videos is NOT below /nas/video."""
        assert not paths.is_inside("/nas/videos/a.mp4", "/nas/video")

    def test_a_prefix_without_a_segment_boundary(self):
        assert not paths.is_inside("/a/bc/x.mp4", "/a/b")

    def test_above(self):
        assert not paths.is_inside("/nas/a.mp4", "/nas/video")

    def test_empty_inputs(self):
        assert not paths.is_inside("", "/nas")
        assert not paths.is_inside("/nas/a.mp4", "")

    def test_it_normalizes_first(self):
        assert paths.is_inside("/nas//video/./a.mp4", "/nas/video/")


class TestRelFrom:
    def test_the_simple_case(self):
        assert paths.rel_from("/nas/video/a.mp4", "/nas/video") == "a.mp4"

    def test_with_a_subfolder(self):
        assert paths.rel_from("/nas/video/series/f1.mp4", "/nas/video") == "series/f1.mp4"

    def test_outside_returns_none(self):
        """No ../../ result — that has no place as rel_path."""
        assert paths.rel_from("/elsewhere/a.mp4", "/nas/video") is None

    def test_next_to_it_returns_none(self):
        assert paths.rel_from("/nas/videos/a.mp4", "/nas/video") is None

    def test_the_root_itself_returns_an_empty_string(self):
        assert paths.rel_from("/nas/video", "/nas/video") == ""

    def test_the_windows_result_uses_forward_slashes(self):
        assert paths.rel_from("C:\\video\\series\\f1.mp4", "C:\\video") == "series/f1.mp4"

    def test_a_trailing_slash_on_the_root(self):
        assert paths.rel_from("/nas/video/a.mp4", "/nas/video/") == "a.mp4"


class TestShown:
    """A stored path as a page shows it (voice map, voices, extractor)."""

    def test_below_the_root_it_is_relative(self):
        assert paths.shown("/nas/video/series/f1.mp4", "/nas/video") == "series/f1.mp4"
        assert paths.shown("/nas/video/series/f1.mp4", "/nas/video/") == "series/f1.mp4"

    def test_a_neighbour_with_the_same_start_stays_whole(self):
        # A bare startswith made "extras/a.mp4" of it — a path to nowhere
        p = "/nas/videos/show extras/a.mp4"
        assert paths.shown(p, "/nas/videos/show") == p

    def test_outside_the_root_it_stays_whole(self):
        p = "/nas/videos/Misc/a.mp4"
        assert paths.shown(p, "/nas/datasets/games") == p
        assert paths.shown("", "/nas") == ""


class TestAbsFrom:
    def test_the_simple_case(self):
        assert paths.abs_from("a.mp4", "/nas/video") == "/nas/video/a.mp4"

    def test_with_a_subfolder(self):
        assert paths.abs_from("series/f1.mp4", "/nas/video") == "/nas/video/series/f1.mp4"

    def test_duplicate_separators_are_smoothed_out(self):
        """Exactly the bug of str(media_dir) + '/' + rel_path."""
        assert paths.abs_from("/series/f1.mp4", "/nas/video/") == "/nas/video/series/f1.mp4"

    def test_a_backslash_only_separates_under_a_windows_root(self):
        """The root decides: under C:/ the backslash separates, under /nas
        it belongs to the name."""
        assert paths.abs_from("series\\f1.mp4", "C:/video") == "C:/video/series/f1.mp4"
        assert paths.abs_from("a\\b.mp4", "/nas/video") == "/nas/video/a\\b.mp4"

    def test_an_empty_rel_returns_the_root(self):
        assert paths.abs_from("", "/nas/video") == "/nas/video"

    def test_without_a_root_it_is_empty(self):
        assert paths.abs_from("a.mp4", "") == ""


class TestRoundTrip:
    @pytest.mark.parametrize(
        "rel",
        [
            "a.mp4",
            "series/f1.mp4",
            "low/er/deeper/f1.mp4",
            "Ünïcode dïr/föö bär.mp4",
            "50%_data/a.mp4",
            "with.dots/in.names.mp4",
        ],
    )
    def test_rel_abs_rel(self, rel):
        root = "/nas/video"
        assert paths.rel_from(paths.abs_from(rel, root), root) == rel


class TestLikePrefix:
    def test_it_ends_with_a_slash(self):
        assert paths.like_prefix("/nas/video") == "/nas/video/"

    def test_percent_is_defused(self):
        assert paths.like_prefix("/mnt/50%data") == "/mnt/50\\%data/"

    def test_underscore_is_defused(self):
        assert paths.like_prefix("/mnt/my_dir") == "/mnt/my\\_dir/"

    def test_backslash_is_defused(self):
        # canon() already turns \ into /, a real backslash only remains with UNC
        assert (
            "\\\\" in paths.like_prefix("\\\\server\\share")
            or paths.like_prefix("\\\\server\\share") == "//server/share/"
        )

    def test_normalizes_the_root(self):
        assert paths.like_prefix("/nas/video/") == "/nas/video/"


class TestContractKept:
    """canon() promises an ABSOLUTE normal form. What cannot fulfill this
    must return "" instead of a plausible-looking value — otherwise
    it lands as supposed identity in the database."""

    @pytest.mark.parametrize(
        "input_value",
        [
            "foo/bar.mp4",
            "a.mp4",
            "./a.mp4",
            "../gone.mp4",
            ".",
            "..",
            "C:foo/bar",  # drive-relative, not absolute
        ],
    )
    def test_non_absolute_input_returns_empty(self, input_value):
        assert paths.canon(input_value) == ""

    def test_unc_spellings_yield_one_identity(self):
        """Both spellings of the same UNC path → ONE abs_path.

        This is the actual promise. It was broken: the Windows branch
        left the leading double slash (ntpath turns
        ``\\\\server\\share`` into exactly ``//server/share``), the POSIX branch
        cut it off. The same file thus got two identities and under
        the partial UNIQUE index two rows.

        Two tests previously held each half of the contradiction and
        were both green, because neither compared the two spellings of the SAME path.
        """
        assert paths.canon("\\\\server\\share\\a.mp4") == paths.canon("//server/share/a.mp4")

    def test_a_double_slash_is_preserved(self):
        """Exactly two leading slashes remain (POSIX allows them).

        Losslessly matched: a real UNC path without double slash is
        broken and not recoverable, an accidental double slash
        in a POSIX path is just an unusual spelling.
        """
        assert paths.canon("//mnt/x/a.mp4") == "//mnt/x/a.mp4"
        # Three or more, normpath combines into one anyway.
        assert paths.canon("///mnt/x/a.mp4") == "/mnt/x/a.mp4"
        assert paths.is_inside("//mnt/x/a.mp4", "//mnt/x")

    def test_whitespace_is_treated_the_same_on_both_sides(self):
        """POSIX and Windows branches must not strip differently.

        The Windows branch called .strip(), the POSIX branch did not — the same
        typo led to a different abs_path depending on platform.
        """
        assert paths.canon("/nas/video  ") == paths.canon("/nas/video")
        assert paths.canon("  /nas/video") == paths.canon("/nas/video")
        assert paths.canon("C:\\video  ") == paths.canon("C:\\video")


class TestIsAbsolute:
    @pytest.mark.parametrize(
        "p,expected",
        [
            ("/nas/a.mp4", True),
            ("C:/nas/a.mp4", True),
            ("C:\\nas\\a.mp4", True),
            ("\\\\server\\share", True),
            ("10:30 Meeting.mp4", False),  # colon in filename is legal on Linux
            ("a:b.mp4", False),
            ("C:foo", False),  # drive-relative
            ("series/a.mp4", False),
            ("", False),
            (None, False),
        ],
    )
    def test_detection(self, p, expected):
        assert paths.is_absolute(p) is expected


class TestNoEscapeFromTheRoot:
    @pytest.mark.parametrize(
        "rel",
        [
            "../../etc/passwd",
            "..",
            "../gone.mp4",
            "a/../../gone.mp4",
            "/../etc/passwd",
        ],
    )
    def test_abs_from_does_not_leave_the_root(self, rel):
        """Input comes partly from requests. A breakout would write an
        arbitrary absolute path into the identity column."""
        result = paths.abs_from(rel, "/nas/video")
        assert result == "" or paths.is_inside(result, "/nas/video")

    def test_a_harmless_dot_segment_stays_allowed(self):
        assert paths.abs_from("series/./f1.mp4", "/nas/video") == "/nas/video/series/f1.mp4"
