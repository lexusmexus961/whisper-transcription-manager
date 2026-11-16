"""The export core (whisper_tm/core/exporter.py): every text format and
option on an invented transcript."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile

import pytest

from whisper_tm.core import exporter as ex

DATA = {
    "speaker_names": {"SPEAKER_00": "Alice"},
    "segments": [
        {
            "start": 0.0,
            "end": 4.0,
            "speaker": "SPEAKER_00",
            "text": "We should move the release to Thursday.",
            "words": [
                {"word": "We", "score": 0.9},
                {"word": "should", "score": 0.3},
                {"word": "move", "score": 0.95},
                {"word": "the", "score": 0.9},
                {"word": "release", "score": 0.9},
                {"word": "to", "score": 0.9},
                {"word": "Thursday.", "score": 0.9},
            ],
        },
        {
            "start": 4.5,
            "end": 7.0,
            "speaker": "SPEAKER_01",
            "text": "The budget is almost used up.",
        },
        {
            "start": 61.0,
            "end": 70.0,
            "speaker": "SPEAKER_01",
            "text": "Let us keep the meeting short today because everyone has a lot of other work waiting for them.",
        },
        {"start": 71.0, "end": 73.0, "speaker": "SPEAKER_00", "text": "Agreed."},
    ],
}
TRANSLATION = [
    {"id": 0, "text": "Deberíamos mover el lanzamiento al jueves."},
    {"id": 1, "text": "El presupuesto está casi agotado."},
]


def doc(opts=None, **kw):
    o = ex.options(opts)
    d = ex.Doc(
        name="planning.m4a",
        rel="talks/planning.m4a",
        title="planning.m4a",
        length=73,
        language="en",
        lines=ex.build_lines(DATA, o, TRANSLATION, kw.get("cuts")),
        summary="A short planning talk.\n- Thursday",
    )
    return d, o


def test_defaults_and_checks():
    o = ex.options({})
    assert o["what"] == "transcript" and o["timestamps"] == "line" and o["names"] is True
    for bad in (
        {"what": "all"},
        {"part": "half"},
        {"timestamps": "hourly"},
        {"packing": "tar"},
        {"speakers": "Alice"},
        {"sub_cpl": "wide"},
    ):
        with pytest.raises(ex.ExportError):
            ex.options(bad)
    assert "secret" not in ex.options({"secret": 1}), "unknown options are dropped"
    assert ex.options({"sub_cpl": 500})["sub_cpl"] == 120


def test_text_with_names_and_times():
    d, o = doc()
    out = ex.render_txt(d, o)
    assert out.startswith("planning.m4a\n0:01:13 · EN")
    assert "[0:00:00] Alice: We should move the release to Thursday." in out
    assert "[0:00:04] Speaker 2: The budget is almost used up." in out


def test_only_some_speakers_and_lines_with_a_word():
    d, o = doc({"speakers": ["Alice"]})
    assert [ln["name"] for ln in d.lines] == ["Alice", "Alice"]
    d, o = doc({"speakers": ["SPEAKER_01"], "lines_with": "budget"})
    assert [ln["text"] for ln in d.lines] == ["The budget is almost used up."]


def test_timestamps_every_minute_or_none_and_no_names():
    d, o = doc({"timestamps": "minute", "names": False})
    out = ex.render_txt(d, o)
    assert "[0:00:00]" in out and "[0:01:00]" in out and "Alice:" not in out
    d, o = doc({"timestamps": "none"})
    assert "[0:" not in ex.render_txt(d, o)


def test_paragraphs_join_a_speakers_lines():
    d, o = doc({"paragraphs": True})
    assert len(d.lines) == 3  # Speaker 2's two lines in a row become one
    data = {
        "segments": [
            {"start": 0, "end": 1, "speaker": "A", "text": "One."},
            {"start": 1, "end": 2, "speaker": "A", "text": "Two."},
            {"start": 2, "end": 3, "speaker": "B", "text": "Three."},
        ]
    }
    lines = ex.build_lines(data, ex.options({"paragraphs": True}))
    assert [ln["text"] for ln in lines] == ["One. Two.", "Three."] and lines[0]["end"] == 2


def test_unsure_words_marked():
    d, o = doc({"unsure": True})
    assert d.lines[0]["text"].startswith("We [should?] move")


def test_translation_both_and_summary():
    d, o = doc({"what": "translation"})
    assert "Deberíamos mover el lanzamiento" in ex.render_txt(d, o)
    assert "Let us keep the meeting short" in ex.render_txt(d, o), (
        "a line without translation keeps its text"
    )
    d, o = doc({"what": "both"})
    out = ex.render_txt(d, o)
    assert "Thursday.\n    Deberíamos mover" in out
    d, o = doc({"what": "summary"})
    out = ex.render_txt(d, o)
    assert "A short planning talk." in out and "Thursday." not in out.split("Summary")[1].replace(
        "- Thursday", ""
    )
    d, o = doc({"summary_on_top": True})
    out = ex.render_txt(d, o)
    assert out.index("Summary") < out.index("We should move")


def test_the_clip_cuts_only():
    d, o = doc({"part": "cuts"}, cuts=[(60, 75)])
    assert [ln["start"] for ln in d.lines] == [61.0, 71.0]


def test_markdown_csv_json():
    d, o = doc({"summary_on_top": True})
    md = ex.render_md(d, o)
    assert md.startswith("# planning.m4a") and "## Summary" in md and "**Alice:** We should" in md
    d, o = doc({"what": "both"})
    rows = list(csv.reader(io.StringIO(ex.render_csv(d, o))))
    assert rows[0] == ["start", "end", "speaker", "text", "translation"] and rows[1][2] == "Alice"
    j = json.loads(ex.render_json(d, o))
    assert j["recording"] == "planning.m4a" and j["lines"][0]["translation"].startswith(
        "Deberíamos mover"
    )


def test_subtitles_follow_the_rules():
    d, o = doc({"sub_cpl": 20, "sub_lines": 1, "sub_max": 2.0, "names": False})
    cues = ex.cues(d, o)
    assert all(len(t) <= 20 or " " not in t for _, _, t in cues)
    assert all(b - a <= 2.0 + 1e-6 for a, b, _ in cues)
    long_line = [c for c in cues if 61 <= c[0] < 70]
    assert len(long_line) >= 5 and abs(long_line[-1][1] - 70.0) < 0.05
    srt = ex.render_srt(d, o)
    assert srt.startswith("1\n00:00:00,000 --> ") and re.search(r"\d\d:\d\d:\d\d,\d\d\d --> ", srt)
    vtt = ex.render_vtt(d, o)
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> ")


def test_word_document_is_a_valid_package():
    d, o = doc({"summary_on_top": True})
    data = ex.render_docx([d], o)
    z = zipfile.ZipFile(io.BytesIO(data))
    assert {"[Content_Types].xml", "_rels/.rels", "word/document.xml", "word/styles.xml"} <= set(
        z.namelist()
    )
    body = z.read("word/document.xml").decode()
    assert (
        "We should move the release to Thursday." in body
        and "Alice: " in body
        and "Summary" in body
    )
    import xml.dom.minidom

    xml_doc = xml.dom.minidom.parseString(body)  # well-formed
    assert xml_doc.documentElement.tagName == "w:document"


def test_pdf_is_well_formed():
    d, o = doc({"summary_on_top": True})
    many = [d] * 30  # several pages
    data = ex.render_pdf(many, o)
    assert data.startswith(b"%PDF-1.4") and data.rstrip().endswith(b"%%EOF")
    xref = int(data.rsplit(b"startxref\n", 1)[1].split(b"\n")[0])
    table = data[xref:].split(b"\n")
    assert table[0] == b"xref"
    count = int(table[1].split()[1])
    for i in range(1, count):
        off = int(table[2 + i].split()[0])
        assert data[off:].startswith(f"{i} 0 obj".encode()), f"object {i} not at its offset"
    assert b"/Count " in data and data.count(b"/Type /Page ") > 1
    assert b"([0:00:00] Alice: We should move the release to Thursday.) Tj" in data


def test_one_file_or_a_zip():
    d, o = doc()
    name, data, mime = ex.render([d], "txt", o)
    assert name == "planning.txt" and mime == "text/plain" and data.startswith(b"planning.m4a")
    d2 = ex.Doc(name="other.mp3", rel="talks/sub/other.mp3", title="other.mp3", lines=d.lines)
    name, data, mime = ex.render([d, d2], "srt", ex.options({"keep_folders": True}))
    assert mime == "application/zip"
    assert sorted(zipfile.ZipFile(io.BytesIO(data)).namelist()) == [
        "talks/planning.srt",
        "talks/sub/other.srt",
    ]
    name, data, mime = ex.render([d, d2], "md", ex.options({"packing": "one"}))
    assert mime == "text/markdown" and data.count(b"# ") >= 2
    with pytest.raises(ex.ExportError):
        ex.render([d, d2], "srt", ex.options({"packing": "one"}))
    with pytest.raises(ex.ExportError):
        ex.render([], "txt", o)


def test_a_pdf_of_another_script_is_refused_with_a_pointer_to_word():
    """Helvetica knows Latin-1 only: a Korean transcript was a page of '?'."""
    import pytest

    from whisper_tm.core import exporter as ex

    ko = ex.Doc(
        name="a.mp4",
        rel="a.mp4",
        title="a",
        lines=[
            {
                "start": 0,
                "end": 1,
                "speaker": "",
                "name": "",
                "text": "안녕하세요 여러분 반갑습니다",
            }
        ],
    )
    with pytest.raises(ex.ExportError, match="Word"):
        ex.render([ko], "pdf", {})
    de = ex.Doc(
        name="b.mp4",
        rel="b.mp4",
        title="b",
        lines=[
            {
                "start": 0,
                "end": 1,
                "speaker": "",
                "name": "",
                "text": "Grüße aus Köln — ein Gedankenstrich",
            }
        ],
    )
    name, data, mime = ex.render([de], "pdf", {})
    assert data.startswith(b"%PDF")


def test_japanese_subtitles_keep_their_line_length():
    """Japanese has no spaces: the wrap took a whole cue as one word and the
    'characters a line' rule did nothing."""
    from whisper_tm.core.exporter import _wrap

    text = "今日はとても良い天気ですね。明日も晴れるといいのですが、雨が降るかもしれません。"
    lines = _wrap(text, 16)
    assert len(lines) > 1 and "".join(lines) == text
    assert all(len(ln) <= 17 for ln in lines), lines  # a closing mark may stay at the end
    assert not any(ln[0] in "、。」" for ln in lines), lines
    assert _wrap("a short English line", 42) == ["a short English line"]
