"""End-to-end tests on generated media using the real FFmpeg."""

import os

import numpy as np
import pytest
import soundfile as sf

import audio_separator as sep
import editor
import ffmpeg_utils
import file_utils
import renderer
import state_manager as sm
import video_processor as vp
from conftest import dominant_freq, duration_of, rms
from models import AudioSettings


@pytest.fixture
def project(tone_wav):
    job_id, job_dir = file_utils.new_job()
    with open(tone_wav, "rb") as fh:
        p = vp.import_media(job_id, job_dir, "My Test Tone.wav", fh, tone_wav.stat().st_size)
    return p, job_dir


def test_upload_like_import(project):
    p, job_dir = project
    assert p.info.duration == pytest.approx(4.0, abs=0.01)
    assert not p.info.has_video
    assert (job_dir / "source" / "My_Test_Tone.wav").exists()
    assert sf.info(p.working_audio).samplerate == 44100


def test_upload_too_large(tone_wav, monkeypatch):
    monkeypatch.setattr(file_utils, "MAX_UPLOAD_MB", 0.01)
    job_id, job_dir = file_utils.new_job()
    with open(tone_wav, "rb") as fh, pytest.raises(file_utils.FileTooLargeError):
        vp.import_media(job_id, job_dir, "t.wav", fh)


def test_unreadable_file(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not really audio" * 100)
    job_id, job_dir = file_utils.new_job()
    with open(bad, "rb") as fh, pytest.raises((ffmpeg_utils.ProbeError, ffmpeg_utils.FFmpegError)):
        vp.import_media(job_id, job_dir, "bad.mp3", fh)


def render(p, clips, settings, fmt="wav", name="out"):
    out = p[1] / "renders" / f"{name}.{ 'mp3' if fmt in ('mp3', 'preview') else 'wav'}"
    return renderer.render_timeline(clips, p[0].working_audio, settings, out, fmt)


def test_trim_split_merge_render(project):
    p, _ = project
    clips = editor.initial_clips(p.working_audio, p.info.duration)
    clips = editor.trim(clips, clips[0].id, 0.5, 3.5, p.info.duration)  # 3 s
    clips = editor.split(clips, clips[0].id, 2.0)  # 1.5 + 1.5
    clips = editor.move(clips, clips[1].id, -1)
    clips = editor.add_silence(clips, clips[0].id, 1.0)  # +1 s
    clips = editor.duplicate(clips, clips[0].id)  # +1.5 s
    out = render(project, clips, AudioSettings())
    assert duration_of(out) == pytest.approx(5.5, abs=0.05)
    data, sr = sf.read(out, always_2d=True)
    row = next(r for r in editor.timeline_metadata(clips) if r["kind"] == "silence")
    t0 = row["timeline_start"]
    silence = data[int((t0 + 0.1) * sr):int((t0 + 0.9) * sr)]  # the inserted silence
    assert np.abs(data[int((t0 - 0.5) * sr):int((t0 - 0.1) * sr)]).max() > 0.3
    assert np.abs(silence).max() < 1e-3


def test_speed_preserves_pitch(project):
    p, _ = project
    clips = editor.initial_clips(p.working_audio, p.info.duration)
    out = render(project, clips, AudioSettings(speed=2.0))
    assert duration_of(out) == pytest.approx(2.0, abs=0.06)
    assert dominant_freq(out) == pytest.approx(440, abs=5)


def test_pitch_preserves_speed(project):
    p, _ = project
    clips = editor.initial_clips(p.working_audio, p.info.duration)
    out = render(project, clips, AudioSettings(pitch=12))
    assert duration_of(out) == pytest.approx(4.0, abs=0.06)
    assert dominant_freq(out) == pytest.approx(880, abs=10)


def test_speed_and_pitch_together(project):
    p, _ = project
    clips = editor.initial_clips(p.working_audio, p.info.duration)
    out = render(project, clips, AudioSettings(speed=0.5, pitch=-12))
    assert duration_of(out) == pytest.approx(8.0, abs=0.1)
    assert dominant_freq(out) == pytest.approx(220, abs=5)


def test_volume(project):
    p, _ = project
    clips = editor.initial_clips(p.working_audio, p.info.duration)
    base = rms(render(project, clips, AudioSettings(), name="a"))
    half = rms(render(project, clips, AudioSettings(volume=0.5), name="b"))
    assert half / base == pytest.approx(0.5, rel=0.05)


def test_eq_changes_level(project):
    p, _ = project
    clips = editor.initial_clips(p.working_audio, p.info.duration)
    s = AudioSettings()
    s.eq[600] = -12  # 440 Hz is within the 600 Hz band's octave
    base = rms(render(project, clips, AudioSettings(), name="a"))
    cut = rms(render(project, clips, s, name="b"))
    assert cut < base * 0.8
    boosted = rms(render(project, clips, AudioSettings(bass=-15, treble=6), name="c"))
    assert boosted != pytest.approx(base, rel=0.01)


def test_fades(project):
    p, _ = project
    clips = editor.initial_clips(p.working_audio, p.info.duration)
    clips = editor.set_fades(clips, clips[0].id, 1.0, 1.0)
    out = render(project, clips, AudioSettings())
    data, sr = sf.read(out, always_2d=True)
    assert np.abs(data[:200]).max() < 0.02 and np.abs(data[-200:]).max() < 0.02


def test_preview_mp3_and_exports(project):
    p, _ = project
    clips = editor.initial_clips(p.working_audio, p.info.duration)
    s = AudioSettings(speed=1.1, bass=4)
    prev = render(project, clips, s, "preview", "preview")
    assert ffmpeg_utils.probe(prev).format_name == "mp3"
    mp3 = render(project, clips, s, "mp3", "export")
    out = ffmpeg_utils.probe(mp3)
    assert out.format_name == "mp3" and out.duration == pytest.approx(4 / 1.1, abs=0.1)
    wav = render(project, clips, s, "wav", "export")
    assert sf.info(wav).subtype == "PCM_16"


def test_video_import_and_export(test_video):
    job_id, job_dir = file_utils.new_job()
    with open(test_video, "rb") as fh:
        p = vp.import_media(job_id, job_dir, "clip.mp4", fh)
    assert p.info.has_video and p.info.width == 320
    clips = editor.initial_clips(p.working_audio, p.info.duration)
    audio = job_dir / "renders" / "a.wav"

    # Sound-only change -> picture copied.
    s = AudioSettings(bass=5)
    renderer.render_timeline(clips, p.working_audio, s, audio, "wav")
    out = vp.export_video(p.info, clips, s.speed, audio, job_dir / "renders" / "same.mp4")
    info = ffmpeg_utils.probe(out)
    assert info.has_video and info.has_audio and info.duration == pytest.approx(4, abs=0.15)

    # Edited timeline + speed -> picture re-rendered to match.
    clips = editor.trim(clips, clips[0].id, 1.0, 3.0, p.info.duration)
    clips = editor.add_silence(clips, None, 1.0)
    s = AudioSettings(speed=1.5)
    renderer.render_timeline(clips, p.working_audio, s, audio, "wav")
    out = vp.export_video(p.info, clips, s.speed, audio, job_dir / "renders" / "edited.mp4")
    info = ffmpeg_utils.probe(out)
    assert info.has_video and info.duration == pytest.approx(3 / 1.5, abs=0.15)


def test_project_json_roundtrip(project):
    p, job_dir = project
    state = {}
    sm.init_state(state)
    sm.reset_for_new_project(state, p, editor.initial_clips(p.working_audio, p.info.duration))
    loaded, clips, settings, track = sm.load_project_file(job_dir / "project.json")
    assert loaded == p and clips == state["clips"] and track == "original"


@pytest.mark.skipif(not sep.separation_available() or os.environ.get("MAS_SKIP_DEMUCS") == "1",
                    reason="vocal separation not installed or skipped")
def test_vocal_separation(project, tmp_path):
    p, job_dir = project
    seen = []
    stems = sep.separate(__import__("pathlib").Path(p.working_audio), job_dir / "stems",
                         progress=lambda v, text: seen.append(v),
                         mode=os.environ.get("MAS_SEPARATION_MODE", "fast"))
    for name in ("instrumental", "vocals"):
        assert sf.info(stems[name]).duration == pytest.approx(4.0, abs=0.01)
    assert seen and seen[-1] <= 1.0
    # Stems are usable as edit sources.
    clips = editor.initial_clips(p.working_audio, p.info.duration)
    out = renderer.render_timeline(clips, str(stems["instrumental"]), AudioSettings(),
                                   job_dir / "renders" / "inst.mp3", "mp3")
    assert duration_of(out) == pytest.approx(4.0, abs=0.1)


def _tone(path, freq, seconds=4.0, amp=0.3):
    sr = 44100
    t = np.arange(int(seconds * sr)) / sr
    x = (amp * np.sin(2 * np.pi * freq * t)).astype("float32")
    sf.write(path, np.stack([x, x], 1), sr, subtype="PCM_16")
    return str(path)


def _peaks(path):
    data, sr = sf.read(str(path), always_2d=True)
    x = data[len(data) // 4: 3 * len(data) // 4, 0]
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return lambda f: spec[np.argmin(np.abs(freqs - f))] / spec.max()


def test_mix_with_separate_stem_settings(tmp_path):
    """Vocals and instrumental get their own pitch/volume, then are merged."""
    vocals = _tone(tmp_path / "v.wav", 440)
    inst = _tone(tmp_path / "i.wav", 660)
    clips = editor.initial_clips(vocals, 4.0)
    master = AudioSettings()
    out = renderer.render_layers(
        clips, [(vocals, [AudioSettings(pitch=12), master]), (inst, [AudioSettings(), master])],
        tmp_path / "mix.wav", "wav")
    level = _peaks(out)
    assert level(880) > 0.5 and level(660) > 0.5  # vocals moved up an octave, music unchanged
    assert level(440) < 0.05
    assert duration_of(out) == pytest.approx(4.0, abs=0.06)

    # Muting the vocals leaves only the music; master pitch shifts everything.
    out = renderer.render_layers(
        clips, [(vocals, [AudioSettings(volume=0)]), (inst, [AudioSettings(), AudioSettings(pitch=-12)])],
        tmp_path / "mix2.wav", "wav")
    level = _peaks(out)
    assert level(330) > 0.9 and level(440) < 0.05 and level(660) < 0.05


def test_mix_different_speeds_and_layers_from_state(project, tmp_path):
    p, job_dir = project
    p.stems = {"vocals": _tone(tmp_path / "v.wav", 440), "instrumental": _tone(tmp_path / "i.wav", 660)}
    state = {}
    sm.init_state(state)
    sm.reset_for_new_project(state, p, editor.initial_clips(p.working_audio, p.info.duration))
    state["current_track"] = "mix"
    state[sm.key("v", "speed")] = 0.5
    state[sm.key("i", "bass")] = 6.0
    sm.sync_settings_from_widgets(state)
    layers = sm.render_layers(state)
    assert [l[0] for l in layers] == [p.stems["vocals"], p.stems["instrumental"]]
    assert layers[0][1][0].speed == 0.5 and layers[1][1][0].bass == 6.0
    out = renderer.render_layers(state["clips"], layers, job_dir / "renders" / "m.mp3", "mp3")
    assert duration_of(out) == pytest.approx(8.0, abs=0.1)  # longest layer wins
    assert sm.effective_speed(state) == 1.0  # music drives the picture
    state["current_track"] = "vocals"
    assert sm.effective_speed(state) == 0.5
