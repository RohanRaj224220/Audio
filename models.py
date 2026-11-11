"""Plain data models shared by every backend module.

Nothing in here imports Streamlit, so the same models can be reused by a
future desktop front end.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

# Supported ranges (also used for validation and UI limits).
SPEED_RANGE = (0.5, 2.0)
PITCH_RANGE = (-12, 12)
TONE_RANGE = (-15.0, 15.0)  # bass / mid / treble in dB
VOLUME_RANGE = (0.0, 2.0)  # 0% .. 200%
EQ_RANGE = (-12.0, 12.0)
EQ_BANDS = (60, 170, 310, 600, 1000, 3000, 6000, 12000)
FADE_MAX = 30.0

TRACK_ORIGINAL = "original"
TRACK_INSTRUMENTAL = "instrumental"
TRACK_VOCALS = "vocals"
TRACK_MIX = "mix"  # vocals + instrumental, each with its own settings, mixed together
STEMS = (TRACK_VOCALS, TRACK_INSTRUMENTAL)
# Pieces the instrumental can be split into (Demucs 6-stem model). Instruments without
# their own piece (flute, strings, synths, ...) end up in "other".
INSTRUMENT_STEMS = ("drums", "bass", "guitar", "piano", "other")
ALL_STEMS = STEMS + INSTRUMENT_STEMS
TRACKS = (TRACK_ORIGINAL, TRACK_INSTRUMENTAL, TRACK_VOCALS, TRACK_MIX) + INSTRUMENT_STEMS

KIND_AUDIO = "audio"
KIND_SILENCE = "silence"


class ValidationError(ValueError):
    """Raised when user supplied values are out of range or inconsistent."""


def _clamp_check(name: str, value: float, lo: float, hi: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number")
    if value != value:  # NaN
        raise ValidationError(f"{name} must be a number")
    if value < lo - 1e-9 or value > hi + 1e-9:
        raise ValidationError(f"{name} must be between {lo:g} and {hi:g}")
    return value


@dataclass
class Clip:
    """One piece of the timeline. Times are seconds in the source track."""

    start: float
    end: float
    source_path: Optional[str] = None
    kind: str = KIND_AUDIO
    name: str = "Clip"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    bass: float = 0.0
    mid: float = 0.0
    treble: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def is_silence(self) -> bool:
        return self.kind == KIND_SILENCE

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Clip":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Overlay:
    """An extra piece of sound laid on top of the song at `position` (seconds in the output)."""

    source_path: str
    start: float
    end: float
    position: float = 0.0
    volume: float = 1.0
    name: str = "Layer"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Overlay":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AudioSettings:
    """Global sound settings applied to the whole timeline at render time."""

    speed: float = 1.0
    pitch: float = 0.0
    bass: float = 0.0
    mid: float = 0.0
    treble: float = 0.0
    volume: float = 1.0
    eq: dict = field(default_factory=lambda: {b: 0.0 for b in EQ_BANDS})

    def validate(self) -> "AudioSettings":
        self.speed = _clamp_check("Speed", self.speed, *SPEED_RANGE)
        self.pitch = _clamp_check("Pitch", self.pitch, *PITCH_RANGE)
        self.bass = _clamp_check("Bass", self.bass, *TONE_RANGE)
        self.mid = _clamp_check("Mid", self.mid, *TONE_RANGE)
        self.treble = _clamp_check("Treble", self.treble, *TONE_RANGE)
        self.volume = _clamp_check("Volume", self.volume, *VOLUME_RANGE)
        clean = {}
        for band in EQ_BANDS:
            clean[band] = _clamp_check(f"EQ {band} Hz", self.eq.get(band, 0.0), *EQ_RANGE)
        unknown = set(self.eq) - set(EQ_BANDS)
        if unknown:
            raise ValidationError(f"Unknown equalizer bands: {sorted(unknown)}")
        self.eq = clean
        return self

    @property
    def changes_duration(self) -> bool:
        return abs(self.speed - 1.0) > 1e-6

    @property
    def is_neutral(self) -> bool:
        return (
            not self.changes_duration
            and self.pitch == 0
            and self.bass == 0
            and self.mid == 0
            and self.treble == 0
            and abs(self.volume - 1.0) < 1e-6
            and all(v == 0 for v in self.eq.values())
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["eq"] = {str(k): v for k, v in self.eq.items()}
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "AudioSettings":
        data = dict(data)
        eq = {int(k): float(v) for k, v in data.pop("eq", {}).items()}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        s = cls(**known)
        s.eq = {b: eq.get(b, 0.0) for b in EQ_BANDS}
        return s


@dataclass
class MediaInfo:
    path: str
    duration: float
    has_audio: bool
    has_video: bool
    format_name: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0

    @property
    def media_type(self) -> str:
        return "Video" if self.has_video else "Audio"


@dataclass
class Project:
    """Everything about one imported file. Clips live separately in state."""

    job_id: str
    source_name: str
    source_path: str
    working_audio: str  # 44.1 kHz stereo WAV extracted from the source
    info: MediaInfo
    stems: dict = field(default_factory=dict)  # track name -> wav path
    origin_url: str = ""  # set when imported from a YouTube/SoundCloud link

    def track_path(self, track: str) -> str:
        if track == TRACK_ORIGINAL:
            return self.working_audio
        path = self.stems.get(track)
        if not path:
            raise ValidationError("That track is not available yet. Remove vocals first.")
        return path

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        data = dict(data)
        info = MediaInfo(**data.pop("info"))
        return cls(info=info, **data)
