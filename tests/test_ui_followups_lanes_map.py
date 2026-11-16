"""Follow-ups to the new window: a recording's lanes go by person and fold,
leaving a recording stops its sound, the voice map says it is loading, and
the folder bar's Auto button is as tall as its neighbours."""

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "whisper_tm" / "static"
REC = (STATIC / "js" / "recording.js").read_text(encoding="utf-8")
SHELL = (STATIC / "js" / "shell.js").read_text(encoding="utf-8")
GRAPH = (STATIC / "js" / "voicemap_graph.js").read_text(encoding="utf-8")
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "css" / "app.css").read_text(encoding="utf-8")


def _lanes():
    i = REC.index("    function lanes() {")
    return REC[i : REC.index("    function laneClick(", i)]


def test_lanes_share_one_row_per_name():
    body = _lanes()
    assert "byName" in body and "getSpeakerName(s)" in body
    # Labels that never speak get no lane; the longest speakers come first
    assert "filter(g => g.segs.length)" in body
    assert "sort((a, b) => b.sec - a.sec)" in body


def test_lanes_fold_with_many_speakers_and_remember_it():
    body = _lanes()
    assert "LANES_FEW" in body and "Rec.lanesToggle(event)" in body
    assert "rv_lanes_all" in REC
    assert "lanesToggle," in REC  # exported for the button
    assert ".rv-lanes-more" in CSS


def test_leaving_a_recording_pauses_its_player():
    assert "if (current === 'file' && place !== 'file') quiet();" in SHELL
    assert "#videoPlayer" in SHELL[SHELL.index("function quiet()") :]


def test_the_voice_map_shows_that_it_is_loading():
    assert 'id="mapLoading"' in HTML
    i = GRAPH.index("async function mapLoadGraph")
    body = GRAPH[i : GRAPH.index("\n/* Initial layout", i)]
    assert "mapLoading(true);" in body
    assert "mapLoading(false)" in body


def test_auto_button_is_as_tall_as_the_folder_bar_buttons():
    assert ".dir-bar .tb-autoscan { height:auto; }" not in CSS
    assert ".dir-bar .tb-autoscan { height:30px;" in CSS


ROOT = Path(__file__).resolve().parents[1]
FILEVIEW = (STATIC / "js" / "fileview.js").read_text(encoding="utf-8")
EXPORT = (STATIC / "js" / "export.js").read_text(encoding="utf-8")


def test_transcript_speaker_chips_one_per_person_and_fold():
    assert "spPeople" in FILEVIEW and "SP_FEW" in FILEVIEW
    assert "rv-spchip-x" in FILEVIEW and ".rv-spchips:not(.all) .rv-spchip-x" in CSS
    # the filter takes a person: voices with the same name count together
    assert "getSpeakerName(el.dataset.speaker) === getSpeakerName(activeSpeaker)" in FILEVIEW
    # the fold button is not a speaker chip (it must never light as "Everyone")
    assert "#rvSpChips .fl-opt:not(.rv-sp-more)" in REC


def test_export_speakers_one_per_name_numbered_like_the_export():
    i = EXPORT.index("const speakerItems = () =>")
    body = EXPORT[i : EXPORT.index("function speakerPicker", i)]
    assert "seen.has(item[0])" in body
    assert "Number(m[1]) + 1" in body


def test_export_preview_reports_the_whole_size(tmp_path):
    src = (ROOT / "whisper_tm" / "api" / "export.py").read_text(encoding="utf-8")
    assert '"size": len(text.encode("utf-8"))' in src
    assert "preview.size ||" in EXPORT


def test_analysis_plan_is_cached_and_cleared(monkeypatch):
    from whisper_tm.core import analysis

    calls = []
    monkeypatch.setattr(analysis, "resolve_source", lambda s: (["/a", "/b"], "set"))
    monkeypatch.setattr(
        analysis, "_plan", lambda p, lang, o, s: calls.append(1) or {"files": len(p)}
    )
    analysis.clear_plan_cache()
    assert analysis.plan({}) == {"files": 2}
    assert analysis.plan({}) == {"files": 2}
    assert len(calls) == 1
    from whisper_tm.api.settings import invalidate_stats_cache

    invalidate_stats_cache()
    analysis.plan({})
    assert len(calls) == 2
    analysis.clear_plan_cache()
