"""Merging, multi-split, redo, overlays, instrument pieces, MP4 from audio, lyrics."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from streamlit.testing.v1 import AppTest

import audio_separator as sep
import editor
import ffmpeg_utils
import file_utils
import lyrics
import renderer
import state_manager as sm
import video_processor as vp
from conftest import dominant_freq, duration_of
from models import AudioSettings, Clip, Overlay, ValidationError

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _tone(path, freq, seconds=4.0, amp=0.3):
    sr = 44100
    t = np.arange(int(seconds * sr)) / sr
    x = amp * np.sin(2 * np.pi * freq * t)
    sf.write(path, np.stack([x, x], 1).astype("float32"), sr, subtype="PCM_16")
    return path


# ---------------------------------------------------------------- editor
def test_split_many_and_merge_back():
    clips = [Clip(0, 10, name="Full track")]
    cid = clips[0].id
    parts = editor.split_many(clips, cid, [7, 2, 5])
    assert [(c.start, c.end) for c in parts] == [(0, 2), (2, 5), (5, 7), (7, 10)]
    assert [c.name for c in parts] == [f"Full track ({n})" for n in (1, 2, 3, 4)]
    assert len({c.id for c in parts}) == 4 and parts[0].id == cid
    whole = editor.merge_all(parts)
    assert [(c.start, c.end, c.name) for c in whole] == [(0, 10, "Full track")]
    assert editor.is_untouched(whole, 10)


def test_split_equal_and_parse_times():
    clips = [Clip(0, 9)]
    assert [c.duration for c in editor.split_equal(clips, clips[0].id, 3)] == [3, 3, 3]
    assert editor.parse_times("1:05; 30  0:10.5") == [10.5, 30, 65]
    with pytest.raises(ValidationError):
        editor.split_equal(clips, clips[0].id, 1000)
    with pytest.raises(ValidationError):
        editor.split_many(clips, clips[0].id, [3, 3.01])


def test_merge_rules():
    a, b = Clip(0, 4), Clip(6, 9)
    with pytest.raises(ValidationError):
        editor.merge([a, b], a.id)  # not continuous
    with pytest.raises(ValidationError):
        editor.merge([a], a.id)  # nothing after it
    s = editor.add_silence(editor.add_silence([a], None, 2), None, 3)
    merged = editor.merge(s, s[1].id)
    assert len(merged) == 2 and merged[1].duration == 5 and merged[1].is_silence
    fast = Clip(4, 8, speed=1.5)
    assert not editor.can_merge(a, fast)


def test_redo():
    state = {}
    sm.init_state(state)
    state["clips"] = [Clip(0, 10)]
    cid = state["clips"][0].id
    sm.apply_edit(state, editor.split, cid, 4)
    assert sm.undo(state) and len(state["clips"]) == 1
    assert sm.redo(state) and len(state["clips"]) == 2
    assert not sm.redo(state)
    sm.undo(state)
    sm.apply_edit(state, editor.duplicate, cid)  # a new edit clears redo
    assert not sm.redo(state)


# ---------------------------------------------------------------- overlays
def test_overlay_is_placed_at_position(tmp_path):
    song = _tone(tmp_path / "song.wav", 440, 6)
    extra = _tone(tmp_path / "extra.wav", 2000, 4)
    out = renderer.render_layers(
        [Clip(0, 6)], [(str(song), AudioSettings())], tmp_path / "out.wav", "wav",
        overlays=[Overlay(str(extra), start=1, end=2, position=4, volume=1.0)])
    data, sr = sf.read(out, always_2d=True)
    assert len(data) / sr == pytest.approx(6, abs=0.05)

    def level(a, b, freq):
        x = data[int(a * sr):int(b * sr), 0]
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        return spec[np.argmin(np.abs(np.fft.rfftfreq(len(x), 1 / sr) - freq))]

    assert level(4.1, 4.9, 2000) > 100 * level(2.0, 2.8, 2000)  # only where it was placed
    assert level(5.2, 5.9, 2000) < level(4.1, 4.9, 2000) / 100  # and it stops after 1 s
    assert level(4.1, 4.9, 440) > 0.5 * level(2.0, 2.8, 440)  # the song keeps playing under it


def test_overlay_can_extend_the_song_and_is_validated(tmp_path):
    song = _tone(tmp_path / "song.wav", 440, 3)
    out = renderer.render_layers([Clip(0, 3)], [(str(song), AudioSettings())], tmp_path / "o.wav",
                                 "wav", overlays=[Overlay(str(song), 0, 3, position=2)])
    assert duration_of(out) == pytest.approx(5, abs=0.05)
    with pytest.raises(ValidationError):
        renderer.render_layers([Clip(0, 3)], [(str(song), AudioSettings())], tmp_path / "o.wav",
                               "wav", overlays=[Overlay(str(tmp_path / "missing.wav"), 0, 1)])


def test_fingerprint_changes_with_overlays():
    clips, s = [Clip(0, 3)], AudioSettings()
    assert sm.fingerprint(clips, s, "original") != sm.fingerprint(
        clips, s, "original", overlays=[Overlay("x.wav", 0, 1)])


# ---------------------------------------------------------------- instrument layers
def _project_with_stems(tmp_path, tone_wav, parts=("drums", "other")):
    job_id, job_dir = file_utils.new_job()
    with open(tone_wav, "rb") as fh:
        project = vp.import_media(job_id, job_dir, "tone.wav", fh)
    freqs = {"vocals": 300, "instrumental": 500, "drums": 800, "bass": 100, "other": 1500}
    for name in ("vocals", "instrumental", *parts):
        project.stems[name] = str(_tone(job_dir / "stems" / f"{name}.wav", freqs[name]))
    state = {}
    sm.init_state(state)
    sm.reset_for_new_project(state, project,
                             editor.initial_clips(project.working_audio, project.info.duration))
    return state, project


def test_mix_uses_instrument_pieces_and_muting(tmp_path, tone_wav):
    state, project = _project_with_stems(tmp_path, tone_wav)
    state["current_track"] = "mix"
    layers = sm.render_layers(state)
    assert [Path(p).stem for p, _ in layers] == ["vocals", "drums", "other"]
    state["stem_settings"]["drums"].volume = 0.0
    assert [Path(p).stem for p, _ in sm.render_layers(state)] == ["vocals", "other"]
    state["current_track"] = "instrumental"
    assert [Path(p).stem for p, _ in sm.render_layers(state)] == ["other"]
    state["current_track"] = "drums"
    out = renderer.render_layers(state["clips"], sm.render_layers(state), tmp_path / "d.wav")
    assert out.exists()  # a single muted piece still renders (as silence)
    state["stem_settings"]["drums"].volume = 1.0
    out = renderer.render_layers(state["clips"], sm.render_layers(state), tmp_path / "d.wav")
    assert dominant_freq(out) == pytest.approx(800, abs=5)


def test_app_with_pieces_overlay_and_lyrics(tmp_path, tone_wav):
    state, project = _project_with_stems(tmp_path, tone_wav)
    at = AppTest.from_file(APP, default_timeout=120)
    for k in ("project", "job_id", "clips"):
        at.session_state[k] = state[k]
    at.session_state.current_track = "mix"
    at.session_state.overlays = [Overlay(project.stems["drums"], 0, 1, position=2, name="Drums")]
    at.session_state.lyrics = lyrics.group_lines([[0.5, 0.9, "Hello"], [1.0, 1.4, "world"]])
    at.run()
    assert not at.exception
    assert "📝 Lyrics" in [t.label for t in at.tabs]
    assert any("Drums" in t.label for t in at.tabs)  # per-instrument sound controls
    [b for b in at.button if "Render Preview" in b.label][0].click().run()
    assert not at.exception and Path(at.session_state.preview["path"]).exists()
    cid = at.session_state.clips[0].id
    at.text_input(key=[t.key for t in at.text_input if str(t.key).startswith("sp_")][0]) \
        .set_value("1 2 3")
    [b for b in at.button if b.label == "Split"][0].click().run()
    assert not at.exception and len(at.session_state.clips) == 4
    at.button(key=f"mrg_{cid}").click().run()
    assert len(at.session_state.clips) == 3
    [b for b in at.button if "Merge all" in b.label][0].click().run()
    assert len(at.session_state.clips) == 1
    [b for b in at.button if b.label == "↶ Undo"][0].click().run()
    [b for b in at.button if b.label == "↷ Redo"][0].click().run()
    assert len(at.session_state.clips) == 1
    at.button(key=f"ovdel_{at.session_state.overlays[0].id}").click().run()
    assert not at.exception and at.session_state.overlays == []


# ---------------------------------------------------------------- merge files / MP4
def test_merge_audio_onto_audio(tmp_path, tone_wav):
    job_id, job_dir = file_utils.new_job()
    with open(tone_wav, "rb") as fh:
        project = vp.import_media(job_id, job_dir, "tone.wav", fh)
    second = _tone(tmp_path / "second.wav", 1000, 2)
    with open(second, "rb") as fh:
        merged, old_len = vp.merge_media(project, job_dir, "second.wav", fh)
    assert old_len == pytest.approx(4, abs=0.01) and merged.info.duration == pytest.approx(6, abs=0.02)
    assert not merged.info.has_video and merged.stems == {}
    clips = [Clip(0, old_len), Clip(old_len, merged.info.duration)]
    out = renderer.render_timeline(clips[1:], merged.working_audio, AudioSettings(), tmp_path / "b.wav")
    assert dominant_freq(out) == pytest.approx(1000, abs=5)
    assert editor.can_merge(*clips)


def test_merge_audio_onto_video_and_mp4_export(tmp_path, tone_wav, test_video):
    job_id, job_dir = file_utils.new_job()
    with open(test_video, "rb") as fh:
        project = vp.import_media(job_id, job_dir, "clip.mp4", fh)
    with open(tone_wav, "rb") as fh:
        merged, old_len = vp.merge_media(project, job_dir, "tone.wav", fh)
    info = ffmpeg_utils.probe(merged.source_path)
    assert info.has_video and info.has_audio and info.duration == pytest.approx(8, abs=0.3)
    assert merged.info.has_video and merged.info.duration == pytest.approx(8, abs=0.1)
    clips = [Clip(0, old_len), Clip(old_len, merged.info.duration)]
    audio = renderer.render_timeline(clips, merged.working_audio, AudioSettings(), tmp_path / "a.wav")
    out = vp.export_video(merged.info, clips, 1.0, audio, tmp_path / "out.mp4")
    assert ffmpeg_utils.probe(out).duration == pytest.approx(8, abs=0.3)


def test_mp4_from_audio_only(tmp_path, tone_wav):
    info = ffmpeg_utils.probe(tone_wav)
    out = vp.export_video(info, [Clip(0, 4)], 1.0, tone_wav, tmp_path / "song.mp4")
    got = ffmpeg_utils.probe(out)
    assert got.has_video and got.has_audio and got.duration == pytest.approx(4, abs=0.3)


# ---------------------------------------------------------------- lyrics
def test_lyrics_formats_and_timeline_mapping():
    words = [[0.5, 0.9, "Hello"], [1.0, 1.4, "world."], [6.0, 6.5, "Again"], [6.6, 7.0, "now"]]
    lines = lyrics.group_lines(words)
    assert [l["text"] for l in lines] == ["Hello world.", "Again now"]
    assert lyrics.to_lrc(lines, False).splitlines() == ["[00:00.50]Hello world.", "[00:06.00]Again now"]
    assert "<00:01.00>world." in lyrics.to_lrc(lines, True)
    assert "00:00:06,000 --> 00:00:07,000" in lyrics.to_srt(lines)
    assert "{\\kf40}Hello" in lyrics.to_ass(lines)
    # Second half of the song moved to the front and everything played at double speed.
    mapped = lyrics.map_to_timeline(lines, [Clip(5, 10), Clip(0, 5)], speed=2.0)
    assert [(l["text"], l["start"]) for l in mapped] == [("Again now", 0.5), ("Hello world.", 2.75)]
    # A trimmed-away line disappears.
    assert [l["text"] for l in lyrics.map_to_timeline(lines, [Clip(0, 3)])] == ["Hello world."]
    kept = lyrics.retime(1.0, 2.8, "Hello world.", lines[0])
    assert kept["words"][0][0] == 1.0 and kept["words"][-1][1] == 2.8
    assert [w[2] for w in lyrics.retime(0, 3, "Hi big world", lines[0])["words"]] == ["Hi", "big", "world"]


SLOW = pytest.mark.skipif(os.environ.get("MAS_SKIP_DEMUCS") == "1", reason="models skipped")


@SLOW
def test_lyrics_transcription_with_word_timings(tmp_path):
    if not lyrics.available() or not shutil.which("say"):
        pytest.skip("needs faster-whisper and the macOS speech tool")
    aiff, wav = tmp_path / "s.aiff", tmp_path / "s.wav"
    subprocess.run(["say", "-o", str(aiff), "The quick brown fox jumps over the lazy dog"], check=True)
    vp.extract_audio(aiff, wav)
    lines = lyrics.transcribe(wav, "en")
    words = [w for l in lines for w in l["words"]]
    text = " ".join(w[2] for w in words).lower()
    assert "quick brown fox" in text and "lazy dog" in text
    assert all(a <= b for a, b, _ in words) and words == sorted(words, key=lambda w: w[0])


@SLOW
def test_split_instruments_pieces_add_up(tmp_path):
    sr = 44100
    t = np.arange(sr * 8) / sr
    kick = np.sin(2 * np.pi * 60 * t) * np.exp(-((t % 0.5) * 18))  # a kick drum every half second
    bass = 0.25 * np.sign(np.sin(2 * np.pi * 82 * t)) * (np.sin(2 * np.pi * 82 * t) ** 2)
    x = (0.6 * kick + bass).astype("float32")
    src = tmp_path / "inst.wav"
    sf.write(src, np.stack([x, x], 1), sr, subtype="PCM_16")
    seen = []
    parts = sep.split_instruments(src, tmp_path / "out", progress=lambda v, s: seen.append(v))
    assert "other" in parts and set(parts) <= {"drums", "bass", "guitar", "piano", "other"}
    assert seen[-1] == 1.0
    total = sum(sf.read(p, always_2d=True)[0] for p in parts.values())
    ref, _ = sf.read(src, always_2d=True)
    assert np.sqrt(((total - ref) ** 2).mean()) < 0.01  # the pieces rebuild the instrumental
