import time

import pytest

import audio_processor as ap
import editor
import ffmpeg_utils
import file_utils
import state_manager as sm
from models import EQ_BANDS, AudioSettings, Clip, ValidationError


# --- timestamp parsing ------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("75", 75.0), ("75.5", 75.5), ("1:15", 75.0), ("1:15.250", 75.25), ("01:01:15.5", 3675.5),
    ("0:00.001", 0.001), (12, 12.0), ("2,5", 2.5),
])
def test_parse_time(text, expected):
    assert editor.parse_time(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "abc", "1:75", "-3", "1::2", "1:2:3:4"])
def test_parse_time_invalid(text):
    with pytest.raises(ValidationError):
        editor.parse_time(text)


def test_format_time_roundtrip():
    for v in (0, 1.5, 75.25, 3675.5):
        assert editor.parse_time(editor.format_time(v)) == pytest.approx(v)


# --- filename sanitization ---------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("My Song.mp3", "My_Song.mp3"),
    ("../../etc/passwd", "passwd"),
    ("C:\\Users\\x\\track.WAV", "track.wav"),
    ("héllo wörld!!.flac", "hello_world_.flac"),
    ("", "file"),
    ("....", "file"),
])
def test_sanitize_filename(name, expected):
    assert file_utils.sanitize_filename(name) == expected


def test_sanitize_long_name():
    assert len(file_utils.sanitize_filename("a" * 500 + ".mp3")) <= 84


def test_extension_check():
    assert file_utils.check_extension("x.MP4") == "mp4"
    with pytest.raises(file_utils.UnsupportedFormatError):
        file_utils.check_extension("x.exe")


# --- clip editing --------------------------------------------------------------
@pytest.fixture
def clips():
    return editor.initial_clips("/src.wav", 60.0)


def test_trim(clips):
    cid = clips[0].id
    out = editor.trim(clips, cid, 5, 30, 60)
    assert (out[0].start, out[0].end) == (5, 30)
    assert clips[0].end == 60  # original untouched (non-destructive)


@pytest.mark.parametrize("s,e", [(30, 5), (-1, 10), (0, 61), (10, 10.01)])
def test_trim_invalid(clips, s, e):
    with pytest.raises(ValidationError):
        editor.trim(clips, clips[0].id, s, e, 60)


def test_split(clips):
    out = editor.split(clips, clips[0].id, 20)
    assert len(out) == 2
    assert (out[0].start, out[0].end, out[1].start, out[1].end) == (0, 20, 20, 60)
    assert out[0].id != out[1].id
    with pytest.raises(ValidationError):
        editor.split(clips, clips[0].id, 60)


def test_ordering_duplicate_delete(clips):
    c = editor.split(clips, clips[0].id, 20)
    a, b = c[0].id, c[1].id
    moved = editor.move(c, b, -1)
    assert [x.id for x in moved] == [b, a]
    assert [x.id for x in editor.move(moved, b, -1)] == [b, a]  # no-op at top
    dup = editor.duplicate(moved, a)
    assert len(dup) == 3 and dup[2].start == dup[1].start and dup[2].id != a
    deleted = editor.delete(dup, b)
    assert b not in [x.id for x in deleted]
    with pytest.raises(ValidationError):
        editor.delete([clips[0]], clips[0].id)


def test_add_silence(clips):
    out = editor.add_silence(clips, clips[0].id, 3)
    assert out[1].is_silence and out[1].duration == 3
    assert editor.timeline_duration(out) == pytest.approx(63)
    with pytest.raises(ValidationError):
        editor.add_silence(clips, None, 0)


def test_is_untouched(clips):
    assert editor.is_untouched(clips, 60)
    assert not editor.is_untouched(editor.trim(clips, clips[0].id, 1, 60, 60), 60)


# --- undo ------------------------------------------------------------------------
def test_undo():
    state = {}
    sm.init_state(state)
    state["clips"] = editor.initial_clips("/src.wav", 60.0)
    cid = state["clips"][0].id
    sm.apply_edit(state, editor.split, cid, 10)
    sm.apply_edit(state, editor.add_silence, None, 2)
    assert len(state["clips"]) == 3
    assert sm.undo(state) and len(state["clips"]) == 2
    assert sm.undo(state) and len(state["clips"]) == 1
    assert not sm.undo(state)


def test_failed_edit_does_not_push_undo():
    state = {}
    sm.init_state(state)
    state["clips"] = editor.initial_clips("/src.wav", 60.0)
    with pytest.raises(ValidationError):
        sm.apply_edit(state, editor.split, state["clips"][0].id, 999)
    assert state["undo_stack"] == []


# --- ffmpeg detection ----------------------------------------------------------------
def test_ffmpeg_detection():
    assert ffmpeg_utils.find_ffmpeg()
    assert ffmpeg_utils.find_ffprobe()
    assert ffmpeg_utils.has_filter("atempo")
    assert ffmpeg_utils.has_filter("equalizer")


def test_ffmpeg_missing(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(ffmpeg_utils.Path, "is_file", lambda self: False)
    assert ffmpeg_utils.find_ffmpeg() is None
    with pytest.raises(ffmpeg_utils.FFmpegNotFound):
        ffmpeg_utils.ffmpeg_bin()


# --- audio settings validation --------------------------------------------------------
def test_settings_validation():
    AudioSettings().validate()
    for bad in ({"speed": 3}, {"speed": 0.4}, {"pitch": 13}, {"bass": 16}, {"volume": 2.5},
                {"speed": "fast"}):
        with pytest.raises(ValidationError):
            AudioSettings(**bad).validate()
    s = AudioSettings()
    s.eq[60] = 13
    with pytest.raises(ValidationError):
        s.validate()


def test_settings_roundtrip_and_widgets():
    s = ap.preset_settings("Bass Boost")
    assert AudioSettings.from_dict(s.to_dict()) == s
    state = {}
    sm.init_widgets(state, s, force=True)
    assert sm.settings_from_widgets(state) == s


def test_presets_valid():
    for name in ap.PRESETS:
        ap.preset_settings(name)
    assert ap.preset_settings("Slow Practice").speed == 0.75


# --- filter generation ------------------------------------------------------------------
def test_filters_neutral():
    filters, librosa_needed = ap.global_filters(AudioSettings(), use_rubberband=False)
    assert filters == [] and not librosa_needed


def test_filters_full():
    s = AudioSettings(speed=1.25, pitch=3, bass=6, mid=-2, treble=4, volume=1.5)
    s.eq[1000] = 2
    filters, librosa_needed = ap.global_filters(s, use_rubberband=False)
    assert filters[0] == "atempo=1.25"
    assert any(f.startswith("bass=g=6") for f in filters)
    assert any(f.startswith("treble=g=4") for f in filters)
    assert "equalizer=f=1000:width_type=o:width=2:g=-2" in filters
    assert "equalizer=f=1000:width_type=o:width=1:g=2" in filters
    assert filters[-1] == "volume=1.5"
    assert librosa_needed
    rb, librosa_needed = ap.global_filters(s, use_rubberband=True)
    assert rb[0].startswith("rubberband=tempo=1.25:pitch=1.18921") and not librosa_needed


def test_fade_filters():
    assert ap.fade_filters(10, 2, 3) == ["afade=t=in:st=0:d=2", "afade=t=out:st=7:d=3"]
    clip = Clip(start=0, end=10, fade_in=1, volume=0.5)
    assert ap.clip_filters(clip, False) == ["afade=t=in:st=0:d=1", "volume=0.5"]


def test_tempo_chain_bounds():
    assert ap.tempo_filters(1.0) == []
    assert ap.tempo_filters(0.5) == ["atempo=0.5"]
    assert ap.tempo_filters(2.0) == ["atempo=2"]


# --- job folders / cleanup -----------------------------------------------------------------
def test_cleanup_respects_active_and_age():
    old_id, old_dir = file_utils.new_job()
    active_id, _ = file_utils.new_job()
    fresh_id, _ = file_utils.new_job()
    future = time.time() + 25 * 3600
    file_utils.touch_job(fresh_id)
    removed = file_utils.cleanup_old_jobs(active_ids=[active_id], now=future)
    assert old_id in removed and active_id not in removed
    assert not old_dir.exists()
    assert file_utils.cleanup_old_jobs(now=time.time()) == []


def test_job_id_validation():
    with pytest.raises(ValueError):
        file_utils.job_path("../../etc")

