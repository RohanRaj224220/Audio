"""Sound controls: speed, pitch, bass/mid/treble, equalizer, volume, fades.

This module only *builds* FFmpeg filter strings and performs the librosa
pitch-shift fallback. It never touches Streamlit.

Speed and pitch are independent:
  * speed uses FFmpeg `atempo` (time-stretch, pitch preserved), or
    `rubberband` when the installed FFmpeg has it;
  * pitch uses FFmpeg `rubberband` if available (keeps duration), otherwise
    librosa's phase-vocoder pitch_shift with high-quality resampling
    (also keeps duration).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

import ffmpeg_utils
from models import EQ_BANDS, AudioSettings, Clip

log = logging.getLogger("mas.audio")

SAMPLE_RATE = 44100
CHANNELS = 2

BASS_FREQ = 100
MID_FREQ = 1000
TREBLE_FREQ = 4000

PRESETS: dict[str, dict] = {
    "Original": {},
    "Bass Boost": {"bass": 8.0, "eq": {60: 4.0, 170: 2.0}},
    "Warm": {"bass": 3.0, "mid": 1.5, "treble": -3.0, "eq": {310: 1.5, 6000: -1.5, 12000: -2.0}},
    "Bright": {"bass": -1.0, "treble": 5.0, "eq": {3000: 2.0, 6000: 3.0, 12000: 3.0}},
    "Slow Practice": {"speed": 0.75},
    "Fast Practice": {"speed": 1.25},
}


def preset_settings(name: str) -> AudioSettings:
    """A fresh AudioSettings for the named preset (everything else neutral)."""
    data = PRESETS[name]
    s = AudioSettings()
    for key, value in data.items():
        if key == "eq":
            for band, gain in value.items():
                s.eq[band] = float(gain)
        else:
            setattr(s, key, float(value))
    return s.validate()


def pitch_ratio(semitones: float) -> float:
    return 2.0 ** (float(semitones) / 12.0)


def rubberband_available() -> bool:
    return ffmpeg_utils.has_filter("rubberband")


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def tempo_filters(speed: float) -> list[str]:
    """Pitch-preserving speed change. atempo accepts 0.5..2.0 everywhere."""
    if abs(speed - 1.0) < 1e-6:
        return []
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={_fmt(remaining)}")
    return filters


def tone_filters(bass: float = 0, mid: float = 0, treble: float = 0,
                 eq: dict | None = None) -> list[str]:
    filters = []
    if bass:
        filters.append(f"bass=g={_fmt(bass)}:f={BASS_FREQ}:width_type=o:width=1")
    if mid:
        filters.append(f"equalizer=f={MID_FREQ}:width_type=o:width=2:g={_fmt(mid)}")
    if treble:
        filters.append(f"treble=g={_fmt(treble)}:f={TREBLE_FREQ}:width_type=o:width=1")
    for band in EQ_BANDS:
        gain = (eq or {}).get(band, 0.0)
        if gain:
            filters.append(f"equalizer=f={band}:width_type=o:width=1:g={_fmt(gain)}")
    return filters


def volume_filters(volume: float) -> list[str]:
    if abs(volume - 1.0) < 1e-6:
        return []
    return [f"volume={_fmt(volume)}"]


def fade_filters(duration: float, fade_in: float, fade_out: float) -> list[str]:
    filters = []
    fade_in = min(max(fade_in, 0.0), duration)
    fade_out = min(max(fade_out, 0.0), duration)
    if fade_in > 0:
        filters.append(f"afade=t=in:st=0:d={_fmt(fade_in)}")
    if fade_out > 0:
        filters.append(f"afade=t=out:st={_fmt(max(0.0, duration - fade_out))}:d={_fmt(fade_out)}")
    return filters


def clip_filters(clip: Clip, use_rubberband: bool | None = None) -> list[str]:
    """Per-clip processing (fades, volume, tone, optional speed/pitch)."""
    if use_rubberband is None:
        use_rubberband = rubberband_available()
    filters = fade_filters(clip.duration, clip.fade_in, clip.fade_out)
    filters += tone_filters(clip.bass, clip.mid, clip.treble)
    filters += volume_filters(clip.volume)
    if clip.pitch and use_rubberband:
        filters.append(f"rubberband=tempo={_fmt(clip.speed)}:pitch={_fmt(pitch_ratio(clip.pitch))}"
                       ":pitchq=quality")
    else:
        if clip.pitch:
            log.warning("Per-clip pitch ignored for clip %s (no rubberband)", clip.id)
        filters += tempo_filters(clip.speed)
    return filters


def global_filters(settings: AudioSettings, use_rubberband: bool | None = None) -> tuple[list[str], bool]:
    """Filters for the whole timeline.

    Returns (filters, needs_librosa_pitch). When the second value is True the
    caller must run `pitch_shift_file` on the result.
    """
    settings.validate()
    if use_rubberband is None:
        use_rubberband = rubberband_available()
    filters: list[str] = []
    needs_librosa = False
    if settings.pitch and use_rubberband:
        filters.append(f"rubberband=tempo={_fmt(settings.speed)}"
                       f":pitch={_fmt(pitch_ratio(settings.pitch))}:pitchq=quality")
    else:
        filters += tempo_filters(settings.speed)
        needs_librosa = bool(settings.pitch)
    filters += tone_filters(settings.bass, settings.mid, settings.treble, settings.eq)
    filters += volume_filters(settings.volume)
    return filters, needs_librosa


def chain_filters(settings_list: list[AudioSettings],
                  use_rubberband: bool | None = None) -> tuple[list[str], float]:
    """Filters for several settings applied one after another (e.g. stem, then whole song).

    Returns (filters, librosa_semitones). A non-zero second value means the
    caller must run `pitch_shift_file` by that many semitones afterwards.
    """
    if use_rubberband is None:
        use_rubberband = rubberband_available()
    filters: list[str] = []
    librosa_semitones = 0.0
    for settings in settings_list:
        f, needs_librosa = global_filters(settings, use_rubberband)
        filters += f
        if needs_librosa:
            librosa_semitones += settings.pitch
    return filters, librosa_semitones


def combined_speed(settings_list: list[AudioSettings]) -> float:
    speed = 1.0
    for s in settings_list:
        speed *= s.speed
    return speed


def limiter_filter() -> str:
    # level=0 disables auto make-up gain so quiet material stays quiet.
    return "alimiter=limit=0.97:level=0"


def pitch_shift_file(src: Path, dest: Path, semitones: float) -> Path:
    """Shift pitch without changing duration using librosa (fallback path)."""
    import librosa  # heavy import, only when needed

    log.info("librosa pitch shift %s semitones on %s", semitones, src)
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    y = np.ascontiguousarray(data.T)
    try:
        shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(semitones), res_type="soxr_vhq")
    except Exception:  # older soxr builds lack vhq
        shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(semitones), res_type="soxr_hq")
    sf.write(str(dest), shifted.T, sr, subtype="FLOAT")
    return dest
