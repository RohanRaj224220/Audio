"""Non-destructive clip editing. Every function returns a NEW list of clips.

Clip times are seconds in the source track. Nothing here touches audio data.
"""

from __future__ import annotations

import copy
import re

from models import KIND_AUDIO, KIND_SILENCE, Clip, ValidationError

MIN_CLIP_SECONDS = 0.05
MAX_SILENCE_SECONDS = 600.0


def parse_time(text) -> float:
    """Parse '75', '75.5', '1:15', '1:15.250', '01:01:15.5' into seconds."""
    if isinstance(text, (int, float)):
        value = float(text)
    else:
        s = str(text or "").strip().replace(",", ".")
        if not s:
            raise ValidationError("Please enter a time")
        if not re.fullmatch(r"\d+(\.\d+)?|(\d+:){1,2}\d{1,2}(\.\d+)?", s):
            raise ValidationError(f"'{text}' is not a valid time. Use seconds or MM:SS.mmm")
        parts = s.split(":")
        value = 0.0
        for i, part in enumerate(parts):
            num = float(part)
            if i > 0 and num >= 60:
                raise ValidationError(f"'{text}' is not a valid time")
            value = value * 60 + num
    if value < 0 or value != value:
        raise ValidationError("Time cannot be negative")
    return round(value, 3)


def format_time(seconds: float) -> str:
    """Seconds -> 'M:SS.mmm' (or 'H:MM:SS.mmm')."""
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    if h:
        return f"{h}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m}:{s:02d}.{ms:03d}"


def _index(clips: list[Clip], clip_id: str) -> int:
    for i, c in enumerate(clips):
        if c.id == clip_id:
            return i
    raise ValidationError("That clip no longer exists")


def _copy(clips: list[Clip]) -> list[Clip]:
    return [copy.deepcopy(c) for c in clips]


def initial_clips(source_path: str, duration: float) -> list[Clip]:
    return [Clip(start=0.0, end=round(duration, 3), source_path=source_path, name="Full track")]


def timeline_duration(clips: list[Clip]) -> float:
    return sum(c.duration / (c.speed or 1.0) for c in clips)


def trim(clips: list[Clip], clip_id: str, start: float, end: float,
         source_duration: float) -> list[Clip]:
    out = _copy(clips)
    clip = out[_index(out, clip_id)]
    start, end = round(float(start), 3), round(float(end), 3)
    if clip.is_silence:
        length = end - start
        if length < MIN_CLIP_SECONDS or length > MAX_SILENCE_SECONDS:
            raise ValidationError("The selected time range is invalid.")
        clip.start, clip.end = 0.0, round(length, 3)
        return out
    if start < 0 or end > source_duration + 1e-3 or end - start < MIN_CLIP_SECONDS:
        raise ValidationError("The selected time range is invalid.")
    clip.start, clip.end = start, min(end, round(source_duration, 3))
    return out


def split(clips: list[Clip], clip_id: str, position: float) -> list[Clip]:
    """Split a clip at an absolute source time strictly inside it."""
    out = _copy(clips)
    i = _index(out, clip_id)
    clip = out[i]
    position = round(float(position), 3)
    if not (clip.start + MIN_CLIP_SECONDS <= position <= clip.end - MIN_CLIP_SECONDS):
        raise ValidationError("The split position must be inside the clip.")
    second = copy.deepcopy(clip)
    second.id = Clip(0, 0).id
    clip.end = position
    second.start = position
    clip.fade_out, second.fade_in = 0.0, 0.0
    base = re.sub(r" \((a|b)\)$", "", clip.name)
    clip.name, second.name = f"{base} (a)", f"{base} (b)"
    out.insert(i + 1, second)
    return out


def duplicate(clips: list[Clip], clip_id: str) -> list[Clip]:
    out = _copy(clips)
    i = _index(out, clip_id)
    dup = copy.deepcopy(out[i])
    dup.id = Clip(0, 0).id
    dup.name = f"{out[i].name} copy"
    out.insert(i + 1, dup)
    return out


def delete(clips: list[Clip], clip_id: str) -> list[Clip]:
    out = _copy(clips)
    if len(out) <= 1:
        raise ValidationError("You need at least one clip.")
    out.pop(_index(out, clip_id))
    return out


def move(clips: list[Clip], clip_id: str, offset: int) -> list[Clip]:
    out = _copy(clips)
    i = _index(out, clip_id)
    j = i + offset
    if j < 0 or j >= len(out):
        return out
    out[i], out[j] = out[j], out[i]
    return out


def add_silence(clips: list[Clip], after_clip_id: str | None, seconds: float) -> list[Clip]:
    seconds = round(float(seconds), 3)
    if seconds < MIN_CLIP_SECONDS or seconds > MAX_SILENCE_SECONDS:
        raise ValidationError("Silence must be between 0.05 and 600 seconds.")
    out = _copy(clips)
    silence = Clip(start=0.0, end=seconds, kind=KIND_SILENCE, name=f"Silence {seconds:g}s")
    if after_clip_id is None:
        out.append(silence)
    else:
        out.insert(_index(out, after_clip_id) + 1, silence)
    return out


def set_fades(clips: list[Clip], clip_id: str, fade_in: float, fade_out: float) -> list[Clip]:
    out = _copy(clips)
    clip = out[_index(out, clip_id)]
    if fade_in < 0 or fade_out < 0 or fade_in + fade_out > clip.duration + 1e-6:
        raise ValidationError("Fades must fit inside the clip.")
    clip.fade_in, clip.fade_out = round(fade_in, 3), round(fade_out, 3)
    return out


def set_clip_volume(clips: list[Clip], clip_id: str, volume: float) -> list[Clip]:
    if not 0 <= volume <= 2:
        raise ValidationError("Volume must be between 0% and 200%.")
    out = _copy(clips)
    out[_index(out, clip_id)].volume = round(float(volume), 3)
    return out


def is_untouched(clips: list[Clip], source_duration: float) -> bool:
    """True when the timeline is still exactly the original full track."""
    if len(clips) != 1:
        return False
    c = clips[0]
    return (c.kind == KIND_AUDIO and c.start <= 1e-3 and abs(c.end - source_duration) < 0.05
            and c.speed == 1.0)


def timeline_metadata(clips: list[Clip]) -> list[dict]:
    """Serializable summary used for project.json and the UI table."""
    rows, t = [], 0.0
    for n, c in enumerate(clips, 1):
        length = c.duration / (c.speed or 1.0)
        rows.append({"#": n, "name": c.name, "kind": c.kind, "start": c.start, "end": c.end,
                     "length": round(length, 3), "timeline_start": round(t, 3)})
        t += length
    return rows
