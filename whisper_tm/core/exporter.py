"""
Exporting transcripts — one recording or many, in the text formats.

``build_lines`` turns a transcript (with its translation and summary when
asked) into the lines an export shows, after the options: what (the
transcript, a translation, both, only the summary), which speakers, which
part (the whole recording or its clip cuts), only lines with a word, and
paragraphs (a speaker's lines joined). The ``render_*`` functions write
those lines as text, subtitles (SRT, WebVTT, cut by the subtitle rules),
Markdown, a CSV table, JSON, a Word document or a PDF.

Word and PDF are written here without a library: a .docx is a ZIP of a few
XML files, and the PDF uses the standard Helvetica font — a few characters
it does not have (outside Latin-1) are replaced, which the export page says;
a text in another script is refused with a pointer to Word.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from xml.sax.saxutils import escape as _xml

TEXT_FORMATS = ("txt", "srt", "vtt", "md", "csv", "json", "docx", "pdf")
EXTENSION = {
    "txt": "txt",
    "srt": "srt",
    "vtt": "vtt",
    "md": "md",
    "csv": "csv",
    "json": "json",
    "docx": "docx",
    "pdf": "pdf",
}
MIME = {
    "txt": "text/plain",
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "md": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "zip": "application/zip",
}

DEFAULTS = {
    "what": "transcript",  # transcript | translation | both | summary
    "language": "",  # the translation's language ("" = the first one there is)
    "summary_on_top": False,
    "speakers": [],  # labels or names; empty = everyone
    "part": "whole",  # whole | cuts
    "lines_with": "",  # only lines that contain this
    "timestamps": "line",  # line | minute | none
    "names": True,  # speaker names in front of the lines
    "paragraphs": False,  # one speaker's consecutive lines as one paragraph
    "unsure": False,  # words the model was unsure of marked [like this?]
    "unsure_below": 0.5,
    "header": True,  # the recording's name, length, date on top
    "sub_cpl": 42,  # subtitles: characters per line
    "sub_lines": 2,  # lines per subtitle
    "sub_max": 7.0,  # longest on screen (seconds)
    "packing": "zip",  # zip (a file each) | one (one document)
    "keep_folders": False,  # in the ZIP, the folders below the export's root
}


class ExportError(ValueError):
    """An option that does not fit."""


def options(raw: dict | None) -> dict:
    """The options with their defaults, each checked."""
    o = dict(DEFAULTS)
    for k, v in (raw or {}).items():
        if k in DEFAULTS:
            o[k] = v
    if o["what"] not in ("transcript", "translation", "both", "summary"):
        raise ExportError("what: transcript, translation, both or summary")
    if o["part"] not in ("whole", "cuts"):
        raise ExportError("part: whole or cuts")
    if o["timestamps"] not in ("line", "minute", "none"):
        raise ExportError("timestamps: line, minute or none")
    if o["packing"] not in ("zip", "one"):
        raise ExportError("packing: zip or one")
    if not isinstance(o["speakers"], list) or not all(isinstance(x, str) for x in o["speakers"]):
        raise ExportError("speakers: a list of names or labels")
    try:
        o["sub_cpl"] = max(10, min(int(o["sub_cpl"]), 120))
        o["sub_lines"] = max(1, min(int(o["sub_lines"]), 4))
        o["sub_max"] = max(1.0, min(float(o["sub_max"]), 30.0))
        o["unsure_below"] = max(0.0, min(float(o["unsure_below"]), 1.0))
    except (TypeError, ValueError):
        raise ExportError("subtitle rules and the unsure threshold must be numbers") from None
    for k in ("summary_on_top", "names", "paragraphs", "unsure", "header", "keep_folders"):
        o[k] = bool(o[k])
    o["lines_with"] = str(o["lines_with"] or "")[:200]
    o["language"] = str(o["language"] or "")[:40]
    return o


@dataclass
class Doc:
    """One recording, ready to write."""

    name: str  # file name of the recording
    rel: str  # its path below the export's root (for the ZIP)
    title: str
    length: float = 0.0
    language: str = ""
    transcribed: float = 0.0
    lines: list = field(
        default_factory=list
    )  # {start, end, speaker, name, text, translation, words}
    summary: str = ""
    translation_language: str = ""


def _clock(sec: float) -> str:
    sec = max(0, int(sec or 0))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _stamp(sec: float, sep: str = ",") -> str:
    sec = max(0.0, float(sec or 0))
    ms = int(round((sec - int(sec)) * 1000))
    s = int(sec)
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}{sep}{ms:03d}"


def speaker_namer(names: dict):
    """A label's name: the transcript's own name, else "Speaker N"."""

    def name(label: str) -> str:
        if not label or label in ("—", "-"):
            return ""
        if names.get(label):
            return str(names[label])
        m = re.match(r"^SPEAKER_(\d+)$", label)
        return f"Speaker {int(m.group(1)) + 1}" if m else label

    return name


def build_lines(
    data: dict, opts: dict, translation: list | None = None, cuts: list | None = None
) -> list[dict]:
    """The transcript's segments as export lines, after the options."""
    name = speaker_namer(data.get("speaker_names") or {})
    trans = {}
    for t in translation or []:
        try:
            trans[int(t.get("id"))] = t.get("text") or ""
        except (TypeError, ValueError):
            continue
    want = {s.lower() for s in opts.get("speakers") or []}
    needle = (opts.get("lines_with") or "").lower()
    keep_cuts = (
        [(float(a), float(b)) for a, b in (cuts or [])] if opts.get("part") == "cuts" else None
    )
    out = []
    for i, s in enumerate(data.get("segments") or []):
        start, end = float(s.get("start") or 0), float(s.get("end") or 0)
        label = s.get("speaker") or ""
        who = name(label)
        if want and label.lower() not in want and who.lower() not in want:
            continue
        if keep_cuts is not None and not any(start < b and end > a for a, b in keep_cuts):
            continue
        text = (s.get("text") or "").strip()
        tr = trans.get(i, "")
        if needle and needle not in text.lower() and needle not in tr.lower():
            continue
        words = s.get("words") or []
        if opts.get("unsure") and words:
            below = opts.get("unsure_below", 0.5)
            text = (
                " ".join(
                    (
                        f"[{w.get('word', '').strip()}?]"
                        if (w.get("score") is not None and float(w.get("score")) < below)
                        else w.get("word", "").strip()
                    )
                    for w in words
                ).strip()
                or text
            )
        out.append(
            {
                "start": start,
                "end": max(start, end),
                "speaker": label,
                "name": who,
                "text": text,
                "translation": tr,
            }
        )
    if opts.get("paragraphs"):
        merged = []
        for ln in out:
            if merged and merged[-1]["speaker"] == ln["speaker"]:
                m = merged[-1]
                m["end"] = ln["end"]
                m["text"] = (m["text"] + " " + ln["text"]).strip()
                m["translation"] = (m["translation"] + " " + ln["translation"]).strip()
            else:
                merged.append(dict(ln))
        out = merged
    return out


def _body_text(ln: dict, opts: dict) -> str:
    what = opts.get("what")
    if what == "translation":
        return ln["translation"] or ln["text"]
    if what == "both":
        return ln["text"] + (f"\n    {ln['translation']}" if ln["translation"] else "")
    return ln["text"]


def _header_lines(doc: Doc) -> list[str]:
    bits = [
        b
        for b in (
            _clock(doc.length) if doc.length else "",
            doc.language.upper(),
            ("transcribed " + time.strftime("%Y-%m-%d", time.localtime(doc.transcribed)))
            if doc.transcribed
            else "",
        )
        if b
    ]
    return [doc.title, " · ".join(bits)] if bits else [doc.title]


# ── Text, Markdown, CSV, JSON ────────────────────────────────────────────


def render_txt(doc: Doc, opts: dict) -> str:
    out = []
    if opts.get("header"):
        out += _header_lines(doc) + [""]
    if doc.summary and (opts.get("summary_on_top") or opts.get("what") == "summary"):
        out += ["Summary", "", doc.summary.strip(), ""]
    if opts.get("what") == "summary":
        return "\n".join(out).rstrip() + "\n"
    last_minute = -1
    for ln in doc.lines:
        who = f"{ln['name']}: " if opts.get("names") and ln["name"] else ""
        if opts.get("timestamps") == "line":
            out.append(f"[{_clock(ln['start'])}] {who}{_body_text(ln, opts)}")
        else:
            if opts.get("timestamps") == "minute" and int(ln["start"] // 60) != last_minute:
                last_minute = int(ln["start"] // 60)
                out.append(f"\n[{_clock(last_minute * 60)}]")
            out.append(f"{who}{_body_text(ln, opts)}")
        if opts.get("paragraphs"):
            out.append("")
    return "\n".join(out).strip() + "\n"


def render_md(doc: Doc, opts: dict) -> str:
    out = []
    if opts.get("header"):
        h = _header_lines(doc)
        out += [f"# {h[0]}"] + ([f"*{h[1]}*"] if len(h) > 1 else []) + [""]
    if doc.summary and (opts.get("summary_on_top") or opts.get("what") == "summary"):
        out += ["## Summary", "", doc.summary.strip(), ""]
    if opts.get("what") == "summary":
        return "\n".join(out).rstrip() + "\n"
    if opts.get("summary_on_top") and doc.summary:
        out += ["## Transcript", ""]
    last_minute = -1
    for ln in doc.lines:
        who = f"**{ln['name']}:** " if opts.get("names") and ln["name"] else ""
        if opts.get("timestamps") == "minute" and int(ln["start"] // 60) != last_minute:
            last_minute = int(ln["start"] // 60)
            out += [f"### {_clock(last_minute * 60)}", ""]
        stamp = f"`{_clock(ln['start'])}` " if opts.get("timestamps") == "line" else ""
        body = _body_text(ln, opts).replace("\n    ", "  \n  *") + (
            "*" if opts.get("what") == "both" and ln["translation"] else ""
        )
        out += [f"{stamp}{who}{body}", ""]
    return "\n".join(out).rstrip() + "\n"


def _csv_text(value) -> str:
    """A text cell that a spreadsheet opens as text. A line that begins with
    =, +, - or @ (or a tab or line break before one) is a formula there —
    and a transcript of a downloaded video can hold one. An apostrophe in
    front makes it text; the programs don't show it."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def render_csv(doc: Doc, opts: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    cols = ["start", "end", "speaker", "text"] + (
        ["translation"] if opts.get("what") in ("both", "translation") else []
    )
    w.writerow(cols)
    for ln in doc.lines:
        row = [
            _clock(ln["start"]),
            _clock(ln["end"]),
            _csv_text(ln["name"] if opts.get("names") else ln["speaker"]),
            _csv_text(
                ln["translation"]
                if opts.get("what") == "translation" and ln["translation"]
                else ln["text"]
            ),
        ]
        if "translation" in cols:
            row.append(_csv_text(ln["translation"]))
        w.writerow(row)
    return buf.getvalue()


def render_json(doc: Doc, opts: dict) -> str:
    return json.dumps(
        {
            "recording": doc.name,
            "path": doc.rel,
            "length_sec": round(doc.length, 2),
            "language": doc.language,
            "summary": doc.summary
            if (opts.get("summary_on_top") or opts.get("what") == "summary")
            else None,
            "translation_language": doc.translation_language or None,
            "lines": []
            if opts.get("what") == "summary"
            else [
                {
                    "start": round(ln["start"], 3),
                    "end": round(ln["end"], 3),
                    "speaker": ln["speaker"],
                    "name": ln["name"],
                    "text": ln["text"],
                    **(
                        {"translation": ln["translation"]}
                        if opts.get("what") in ("both", "translation")
                        else {}
                    ),
                }
                for ln in doc.lines
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


# ── Subtitles ────────────────────────────────────────────────────────────


#: Characters a subtitle line must not start with (Japanese line-breaking
#: rule, kinsoku): they stay at the end of the line before
_NO_LINE_START = set(
    "、。，．・：；？！ー…‥」』）〕］｝〉》】〙〗〟’”ゝゞヽヾぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ,.:;?!)]}%"
)


def _break_unspaced(word: str, cpl: int) -> list[str]:
    """A run without spaces longer than a line — Japanese or Chinese, where
    words aren't spaced — cut every ``cpl`` characters, never so that a line
    starts with a closing mark or small kana."""
    out, cur = [], ""
    for ch in word:
        if len(cur) >= cpl and ch not in _NO_LINE_START:
            out.append(cur)
            cur = ""
        cur += ch
    if cur:
        out.append(cur)
    return out


def _wrap(text: str, cpl: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        pieces = _break_unspaced(word, cpl) if len(word) > cpl else [word]
        for piece in pieces:
            if cur and len(cur) + 1 + len(piece) > cpl:
                lines.append(cur)
                cur = piece
            else:
                cur = f"{cur} {piece}".strip()
    if cur:
        lines.append(cur)
    return lines


def cues(doc: Doc, opts: dict) -> list[tuple[float, float, str]]:
    """Subtitle cues after the rules: at most sub_lines lines of sub_cpl
    characters, at most sub_max seconds each — a long line is split, its
    time shared by the characters."""
    out = []
    cpl, nlines, longest = opts["sub_cpl"], opts["sub_lines"], opts["sub_max"]
    for ln in doc.lines:
        text = _body_text(ln, opts).replace("\n    ", " ")
        if opts.get("names") and ln["name"]:
            text = f"{ln['name']}: {text}"
        wrapped = _wrap(text, cpl)
        chunks = [wrapped[i : i + nlines] for i in range(0, len(wrapped), nlines)] or [[""]]
        total_chars = sum(len(" ".join(c)) for c in chunks) or 1
        t = ln["start"]
        span = max(0.01, ln["end"] - ln["start"])
        for c in chunks:
            d = span * len(" ".join(c)) / total_chars
            start = t
            while d > 0:
                piece = min(d, longest)
                out.append((start, start + piece, "\n".join(c)))
                start += piece
                d -= piece
            t += span * len(" ".join(c)) / total_chars
    return out


def render_srt(doc: Doc, opts: dict) -> str:
    return "\n".join(
        f"{i}\n{_stamp(a)} --> {_stamp(b)}\n{t}\n" for i, (a, b, t) in enumerate(cues(doc, opts), 1)
    )


def render_vtt(doc: Doc, opts: dict) -> str:
    return "WEBVTT\n\n" + "\n".join(
        f"{_stamp(a, '.')} --> {_stamp(b, '.')}\n{t}\n" for a, b, t in cues(doc, opts)
    )


# ── Word (.docx), written by hand ────────────────────────────────────────


def _docx_par(text: str, bold_prefix: str = "", style: str = "", italic: bool = False) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    runs = ""
    if bold_prefix:
        runs += (
            f'<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{_xml(bold_prefix)}</w:t></w:r>'
        )
    parts = text.split("\n")
    for i, p in enumerate(parts):
        rpr = "<w:rPr><w:i/></w:rPr>" if italic or i > 0 else ""
        br = "<w:br/>" if i else ""
        shown = _xml(p.strip() if i else p)
        runs += f'<w:r>{rpr}{br}<w:t xml:space="preserve">{shown}</w:t></w:r>'
    return f"<w:p>{ppr}{runs}</w:p>"


def render_docx(docs: list[Doc], opts: dict) -> bytes:
    body = []
    for n, doc in enumerate(docs):
        if n:
            body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        if opts.get("header"):
            h = _header_lines(doc)
            body.append(_docx_par(h[0], style="Title"))
            if len(h) > 1:
                body.append(_docx_par(h[1], style="Subtitle"))
        if doc.summary and (opts.get("summary_on_top") or opts.get("what") == "summary"):
            body.append(_docx_par("Summary", style="Heading1"))
            for para in doc.summary.strip().split("\n"):
                if para.strip():
                    body.append(_docx_par(para.strip().lstrip("#*- ").strip()))
        if opts.get("what") == "summary":
            continue
        if opts.get("summary_on_top") and doc.summary:
            body.append(_docx_par("Transcript", style="Heading1"))
        last_minute = -1
        for ln in doc.lines:
            if opts.get("timestamps") == "minute" and int(ln["start"] // 60) != last_minute:
                last_minute = int(ln["start"] // 60)
                body.append(_docx_par(_clock(last_minute * 60), style="Heading2"))
            stamp = f"[{_clock(ln['start'])}] " if opts.get("timestamps") == "line" else ""
            who = f"{ln['name']}: " if opts.get("names") and ln["name"] else ""
            body.append(_docx_par(_body_text(ln, opts), bold_prefix=stamp + who))
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{"".join(body)}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" '
        'w:footer="708" w:gutter="0"/>'
        "</w:sectPr></w:body></w:document>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" '
        'w:cs="Calibri"/>'
        '<w:sz w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing '
        'w:after="120"/></w:pPr></w:pPrDefault></w:docDefaults>'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>'
        "</w:style>"
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/>'
        '<w:sz w:val="40"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:rPr>'
        '<w:color w:val="666666"/><w:sz w:val="22"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr>'
        '<w:spacing w:before="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:rPr>'
        '<w:b/><w:color w:val="555555"/><w:sz w:val="24"/></w:rPr></w:style>'
        "</w:styles>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" '
            'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.documen'
            't.main+xml"/>'
            '<Override PartName="/word/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+'
            'xml"/>'
            "</Types>",
        )
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDoc'
            'ument" Target="word/document.xml"/>'
            "</Relationships>",
        )
        z.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            "</Relationships>",
        )
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", styles)
    return buf.getvalue()


# ── PDF, written by hand (Helvetica, A4) ─────────────────────────────────

# Helvetica's advance widths for ASCII 32..126 (per 1000 units)
_HELV = [
    278,
    278,
    355,
    556,
    556,
    889,
    667,
    191,
    333,
    333,
    389,
    584,
    278,
    333,
    278,
    278,
    556,
    556,
    556,
    556,
    556,
    556,
    556,
    556,
    556,
    556,
    278,
    278,
    584,
    584,
    584,
    556,
    1015,
    667,
    667,
    722,
    722,
    667,
    611,
    778,
    722,
    278,
    500,
    667,
    556,
    833,
    722,
    778,
    667,
    778,
    722,
    667,
    611,
    722,
    667,
    944,
    667,
    667,
    611,
    278,
    278,
    278,
    469,
    556,
    333,
    556,
    556,
    500,
    556,
    556,
    278,
    556,
    556,
    222,
    222,
    500,
    222,
    833,
    556,
    556,
    556,
    556,
    333,
    500,
    278,
    556,
    500,
    722,
    500,
    500,
    500,
    334,
    260,
    334,
    584,
]


def _pdf_width(text: str, size: float) -> float:
    return sum((_HELV[ord(c) - 32] if 32 <= ord(c) <= 126 else 556) for c in text) * size / 1000


def _pdf_str(text: str) -> str:
    raw = text.encode("latin-1", "replace").decode("latin-1")
    return raw.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


#: Characters the PDF may print as "?" before it is refused rather than written
PDF_LOSS_ALLOWED = 0.01


def pdf_losses(docs: list[Doc]) -> tuple[int, int]:
    """(characters the PDF's Latin-1 font can't print, characters in all) —
    of the text that would go into it."""
    lost = total = 0
    for d in docs:
        texts = [d.title, d.summary] + [
            str(ln.get("text") or "") + str(ln.get("translation") or "") + str(ln.get("name") or "")
            for ln in d.lines
        ]
        for t in texts:
            for ch in t or "":
                if ch.isspace():
                    continue
                total += 1
                if ord(ch) > 0xFF:
                    lost += 1
    return lost, total


def render_pdf(docs: list[Doc], opts: dict) -> bytes:
    lost, total = pdf_losses(docs)
    if lost > 3 and lost > total * PDF_LOSS_ALLOWED:
        # The built-in Helvetica knows Latin-1 only: a Korean, Russian or
        # Arabic transcript came out as a page of question marks
        raise ExportError(
            f"PDF can show Latin letters only — {lost:,} of {total:,} "
            f"characters here would print as '?'. Choose Word (.docx), "
            f"which keeps every script."
        )
    W, H, M = 595.0, 842.0, 56.0
    pages: list[list[str]] = []
    ops: list[str] = []
    y = H - M

    def new_page():
        nonlocal ops, y
        if ops:
            pages.append(ops)
        ops, y = [], H - M

    def line(
        text: str, size: float = 10.5, bold: bool = False, gap: float = 3.0, indent: float = 0.0
    ):
        nonlocal y
        width = W - 2 * M - indent
        words, cur, out = text.split(" "), "", []
        for wd in words:
            test = f"{cur} {wd}".strip()
            if cur and _pdf_width(test, size) > width:
                out.append(cur)
                cur = wd
            else:
                cur = test
        out.append(cur)
        for piece in out:
            if y < M + size:
                new_page()
            font = "F2" if bold else "F1"
            ops.append(
                f"BT /{font} {size} Tf {M + indent:.1f} {y:.1f} Td ({_pdf_str(piece)}) Tj ET"
            )
            y -= size * 1.32
        y -= gap

    for n, doc in enumerate(docs):
        if n:
            new_page()
        if opts.get("header"):
            h = _header_lines(doc)
            line(h[0], 17, True, 2)
            if len(h) > 1:
                line(h[1], 9.5, False, 10)
        if doc.summary and (opts.get("summary_on_top") or opts.get("what") == "summary"):
            line("Summary", 13, True, 4)
            for para in doc.summary.strip().split("\n"):
                if para.strip():
                    line(para.strip().lstrip("#*- ").strip(), 10.5, False, 3)
            y -= 6
        if opts.get("what") == "summary":
            continue
        if opts.get("summary_on_top") and doc.summary:
            line("Transcript", 13, True, 4)
        last_minute = -1
        for ln in doc.lines:
            if opts.get("timestamps") == "minute" and int(ln["start"] // 60) != last_minute:
                last_minute = int(ln["start"] // 60)
                line(_clock(last_minute * 60), 11, True, 2)
            stamp = f"[{_clock(ln['start'])}] " if opts.get("timestamps") == "line" else ""
            who = f"{ln['name']}: " if opts.get("names") and ln["name"] else ""
            body = _body_text(ln, opts)
            first, *rest = body.split("\n")
            line(stamp + who + first, 10.5, False, 1 if rest else 3)
            for r in rest:
                line(r.strip(), 9.5, False, 3, indent=18)
    new_page()

    objs: list[bytes] = []

    def obj(content: bytes) -> int:
        objs.append(content)
        return len(objs)

    font1 = obj(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )
    font2 = obj(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    )
    pages_id = obj(b"")  # the page tree, filled in once the pages are known
    page_ids = []
    for p in pages or [[]]:
        stream = "\n".join(p).encode("latin-1", "replace")
        sid = obj(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(
            obj(
                (
                    f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {W:.0f} {H:.0f}] "
                    f"/Resources << /Font << /F1 {font1} 0 R /F2 {font2} 0 R >> >> /Contents "
                    f"{sid} 0 R >>"
                ).encode()
            )
        )
    objs[pages_id - 1] = (
        "<< /Type /Pages /Kids ["
        + " ".join(f"{i} 0 R" for i in page_ids)
        + f"] /Count {len(page_ids)} >>"
    ).encode()
    real_pages = pages_id
    catalog = obj(f"<< /Type /Catalog /Pages {real_pages} 0 R >>".encode())
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + o + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        (
            f"trailer\n<< /Size {len(objs) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    return out.getvalue()


# ── Putting it together ──────────────────────────────────────────────────

RENDER_ONE = {
    "txt": render_txt,
    "md": render_md,
    "csv": render_csv,
    "json": render_json,
    "srt": render_srt,
    "vtt": render_vtt,
}


def render(docs: list[Doc], fmt: str, opts: dict) -> tuple[str, bytes, str]:
    """(file name, bytes, mime type): one file for one recording or with
    packing 'one'; a ZIP with a file each otherwise."""
    if fmt not in TEXT_FORMATS:
        raise ExportError(f"format: one of {', '.join(TEXT_FORMATS)}")
    if not docs:
        raise ExportError("nothing to export")
    ext = EXTENSION[fmt]

    def stem(d):
        return PurePosixPath(d.name).stem

    if len(docs) == 1 or opts.get("packing") == "one":
        if fmt == "docx":
            data = render_docx(docs, opts)
        elif fmt == "pdf":
            data = render_pdf(docs, opts)
        elif fmt == "json" and len(docs) > 1:
            data = ("[\n" + ",\n".join(render_json(d, opts) for d in docs) + "\n]").encode()
        elif fmt in ("srt", "vtt") and len(docs) > 1:
            raise ExportError("subtitles are one file per recording — pack them as a ZIP")
        else:
            sep = (
                "\n\n" + "─" * 40 + "\n\n"
                if fmt == "txt"
                else "\n\n---\n\n"
                if fmt == "md"
                else "\n"
            )
            parts = [RENDER_ONE[fmt](d, opts) for d in docs]
            if fmt == "csv" and len(parts) > 1:
                parts = [parts[0]] + [p.split("\n", 1)[1] if "\n" in p else "" for p in parts[1:]]
            data = sep.join(parts).encode("utf-8")
        name = (
            f"{stem(docs[0])}.{ext}"
            if len(docs) == 1
            else f"transcripts_{time.strftime('%Y%m%d_%H%M')}.{ext}"
        )
        return name, data, MIME[fmt]
    buf = io.BytesIO()
    taken: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for d in docs:
            base = PurePosixPath(d.rel) if opts.get("keep_folders") else PurePosixPath(d.name)
            arc = str(base.with_suffix("." + ext))
            k, n = arc, 2
            while k in taken:
                k = str(PurePosixPath(arc).with_name(f"{PurePosixPath(arc).stem} ({n}).{ext}"))
                n += 1
            taken.add(k)
            if fmt == "docx":
                data = render_docx([d], opts)
            elif fmt == "pdf":
                data = render_pdf([d], opts)
            else:
                data = RENDER_ONE[fmt](d, opts).encode("utf-8")
            z.writestr(k, data)
    return f"transcripts_{time.strftime('%Y%m%d_%H%M')}.zip", buf.getvalue(), MIME["zip"]
