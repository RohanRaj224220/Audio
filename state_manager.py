"""Project/session state helpers.

Works on any dict-like mapping, so it is used with `st.session_state` in the
Streamlit app and with a plain dict in tests (or a future desktop app).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, MutableMapping

import editor
from file_utils import job_path
from models import (ALL_STEMS, EQ_BANDS, INSTRUMENT_STEMS, STEMS, TRACK_INSTRUMENTAL, TRACK_MIX,
                    TRACK_ORIGINAL, TRACK_VOCALS, AudioSettings, Clip, Overlay, Project,
                    ValidationError)

log = logging.getLogger("mas.state")

UNDO_LIMIT = 50

# Widget key prefixes for the Sound Controls tab: whole song, vocals, instrumental.
MASTER = "w"
PREFIXES = {MASTER: None, "v": "vocals", "i": "instrumental", "d": "drums", "b": "bass",
            "g": "guitar", "p": "piano", "o": "other"}
STEM_PREFIX = {stem: p for p, stem in PREFIXES.items() if stem}


def key(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


W_SPEED, W_PITCH, W_BASS, W_MID, W_TREBLE, W_VOLUME = (
    key(MASTER, n) for n in ("speed", "pitch", "bass", "mid", "treble", "volume"))


def eq_key(band: int, prefix: str = MASTER) -> str:
    return f"{prefix}_eq_{band}"


DEFAULTS = {
    "project": None,
    "clips": [],
    "current_track": TRACK_ORIGINAL,
    "undo_stack": [],
    "redo_stack": [],
    "overlays": [],  # extra layers placed on top of the song (models.Overlay)
    "lyrics": [],  # [{"start": s, "end": s, "text": str, "words": [[start, end, word], ...]}]
    "pieces": {},  # stem name -> lossless FLAC path (downloads)
    "extras": {},  # uploaded extra sounds: display name -> wav path
    "audio_settings": None,
    "stem_settings": None,  # {"vocals": AudioSettings, "instrumental": AudioSettings}
    "job_id": None,
    "preview": None,  # {"path": str, "fingerprint": str}
    "exports": {},  # kind -> path
    "stem_downloads": {},  # track -> mp3 path
    "selected_clip": None,
    "share_result": None,  # last "Compress for WhatsApp" output
    "upload_signature": None,
}


def init_state(state: MutableMapping) -> None:
    for key, value in DEFAULTS.items():
        if key not in state:
            state[key] = copy.deepcopy(value)
    if state["audio_settings"] is None:
        state["audio_settings"] = AudioSettings()
    if state["stem_settings"] is None:
        state["stem_settings"] = {}
    for stem in ALL_STEMS:
        state["stem_settings"].setdefault(stem, AudioSettings())
    for prefix in PREFIXES:
        init_widgets(state, prefix=prefix)


def init_widgets(state: MutableMapping, settings: AudioSettings | None = None,
                 force: bool = False, prefix: str = MASTER) -> None:
    s = settings or AudioSettings()
    values = {key(prefix, "speed"): s.speed, key(prefix, "pitch"): int(s.pitch),
              key(prefix, "bass"): s.bass, key(prefix, "mid"): s.mid,
              key(prefix, "treble"): s.treble, key(prefix, "volume"): int(round(s.volume * 100))}
    values.update({eq_key(b, prefix): s.eq[b] for b in EQ_BANDS})
    for k, value in values.items():
        if force or k not in state:
            state[k] = value


def settings_from_widgets(state: MutableMapping, prefix: str = MASTER) -> AudioSettings:
    s = AudioSettings(
        speed=float(state[key(prefix, "speed")]), pitch=float(state[key(prefix, "pitch")]),
        bass=float(state[key(prefix, "bass")]), mid=float(state[key(prefix, "mid")]),
        treble=float(state[key(prefix, "treble")]),
        volume=float(state[key(prefix, "volume")]) / 100.0,
        eq={b: float(state[eq_key(b, prefix)]) for b in EQ_BANDS},
    )
    return s.validate()


def sync_settings_from_widgets(state: MutableMapping) -> None:
    """Copy all slider values into audio_settings / stem_settings."""
    state["audio_settings"] = settings_from_widgets(state, MASTER)
    state["stem_settings"] = {stem: settings_from_widgets(state, p)
                              for stem, p in STEM_PREFIX.items()}


def render_layers(state: MutableMapping) -> list[tuple[str, list[AudioSettings]]]:
    """[(track_path, [stem settings, whole-song settings]), ...] for the current track."""
    project: Project = state["project"]
    track = state.get("current_track", TRACK_ORIGINAL)
    master = state["audio_settings"]
    stems = state["stem_settings"]
    if track == TRACK_ORIGINAL:
        return [(project.working_audio, [master])]
    bus = stems[TRACK_INSTRUMENTAL]
    parts = [p for p in INSTRUMENT_STEMS if p in project.stems]
    # Once the music is split into instruments, the instrumental is rebuilt from its pieces
    # (each with its own settings, then the Instrumental settings on top).
    music = ([(project.stems[p], [stems[p], bus, master]) for p in parts] if parts
             else [(project.track_path(TRACK_INSTRUMENTAL), [bus, master])]
             if TRACK_INSTRUMENTAL in project.stems else [])
    if track == TRACK_MIX:
        if not all(s in project.stems for s in STEMS):
            raise ValidationError("Remove vocals first to use the mix.")
        layers = [(project.stems[TRACK_VOCALS], [stems[TRACK_VOCALS], master])] + music
    elif track == TRACK_INSTRUMENTAL:
        project.track_path(track)  # raises when not separated yet
        layers = music
    elif track in INSTRUMENT_STEMS:
        layers = [(project.track_path(track), [stems[track], bus, master])]
    else:
        layers = [(project.track_path(track), [stems[track], master])]
    # A piece turned down to 0% is simply left out (saves rendering time).
    audible = [l for l in layers if all(s.volume > 1e-6 for s in l[1])]
    return audible or layers[-1:]


def effective_speed(state: MutableMapping) -> float:
    """Overall playback speed of the current track (used for the video picture)."""
    layers = render_layers(state)
    # In a mix with different stem speeds the music (instrumental) drives the picture.
    path, settings = layers[-1]
    speed = 1.0
    for s in settings:
        speed *= s.speed
    return speed


def reset_for_new_project(state: MutableMapping, project: Project, clips: list[Clip]) -> None:
    state["project"] = project
    state["job_id"] = project.job_id
    state["clips"] = clips
    state["undo_stack"] = []
    state["redo_stack"] = []
    state["overlays"] = []
    state["lyrics"] = []
    state["pieces"] = {}
    state["extras"] = {}
    state["current_track"] = TRACK_ORIGINAL
    state["preview"] = None
    state["exports"] = {}
    state["stem_downloads"] = {}
    state["share_result"] = None
    state["selected_clip"] = clips[0].id if clips else None
    save_project(state)


def push_undo(state: MutableMapping, clear_redo: bool = True) -> None:
    stack = state["undo_stack"]
    stack.append([c.to_dict() for c in state["clips"]])
    del stack[:-UNDO_LIMIT]
    if clear_redo:
        state["redo_stack"] = []


def undo(state: MutableMapping) -> bool:
    return _step(state, "undo_stack", "redo_stack")


def redo(state: MutableMapping) -> bool:
    return _step(state, "redo_stack", "undo_stack")


def _step(state: MutableMapping, source: str, other: str) -> bool:
    if not state.get(source):
        return False
    state.setdefault(other, []).append([c.to_dict() for c in state["clips"]])
    state["clips"] = [Clip.from_dict(d) for d in state[source].pop()]
    ids = {c.id for c in state["clips"]}
    if state.get("selected_clip") not in ids:
        state["selected_clip"] = state["clips"][0].id if state["clips"] else None
    save_project(state)
    return True


def apply_edit(state: MutableMapping, fn: Callable[..., list[Clip]], *args, **kwargs) -> None:
    """Run an editor function; only record undo if it succeeds."""
    new_clips = fn(state["clips"], *args, **kwargs)
    push_undo(state)
    state["clips"] = new_clips
    save_project(state)


def fingerprint(clips: list[Clip], settings: AudioSettings, track: str,
                stem_settings: dict | None = None, overlays: list[Overlay] | None = None) -> str:
    payload = json.dumps({"clips": [c.to_dict() for c in clips], "settings": settings.to_dict(),
                          "track": track, "overlays": [o.to_dict() for o in overlays or []],
                          "stems": {k: v.to_dict() for k, v in (stem_settings or {}).items()}},
                         sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def save_project(state: MutableMapping) -> Path | None:
    """Persist project metadata to workspace/jobs/<id>/project.json."""
    project: Project | None = state.get("project")
    if project is None:
        return None
    settings = state.get("audio_settings") or AudioSettings()
    data = {
        "project": project.to_dict(),
        "clips": [c.to_dict() for c in state["clips"]],
        "timeline": editor.timeline_metadata(state["clips"]),
        "current_track": state.get("current_track", TRACK_ORIGINAL),
        "audio_settings": settings.to_dict(),
        "stem_settings": {k: v.to_dict() for k, v in (state.get("stem_settings") or {}).items()},
        "overlays": [o.to_dict() for o in state.get("overlays") or []],
        "lyrics": state.get("lyrics") or [],
    }
    try:
        path = job_path(project.job_id) / "project.json"
        path.write_text(json.dumps(data, indent=2))
        return path
    except OSError:
        log.exception("Could not save project.json")
        return None


def load_project_file(path: Path) -> tuple[Project, list[Clip], AudioSettings, str]:
    data = json.loads(Path(path).read_text())
    return (Project.from_dict(data["project"]), [Clip.from_dict(c) for c in data["clips"]],
            AudioSettings.from_dict(data["audio_settings"]), data.get("current_track", TRACK_ORIGINAL))
