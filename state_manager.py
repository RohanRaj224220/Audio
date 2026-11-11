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
from models import (EQ_BANDS, STEMS, TRACK_MIX, TRACK_ORIGINAL, AudioSettings, Clip, Project,
                    ValidationError)

log = logging.getLogger("mas.state")

UNDO_LIMIT = 50

# Widget key prefixes for the Sound Controls tab: whole song, vocals, instrumental.
MASTER = "w"
PREFIXES = {MASTER: None, "v": "vocals", "i": "instrumental"}
STEM_PREFIX = {"vocals": "v", "instrumental": "i"}


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
        state["stem_settings"] = {stem: AudioSettings() for stem in STEMS}
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
    if track == TRACK_MIX:
        if not all(s in project.stems for s in STEMS):
            raise ValidationError("Remove vocals first to use the mix.")
        return [(project.stems[s], [stems[s], master]) for s in STEMS]
    return [(project.track_path(track), [stems[track], master])]


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
    state["current_track"] = TRACK_ORIGINAL
    state["preview"] = None
    state["exports"] = {}
    state["stem_downloads"] = {}
    state["share_result"] = None
    state["selected_clip"] = clips[0].id if clips else None
    save_project(state)


def push_undo(state: MutableMapping) -> None:
    stack = state["undo_stack"]
    stack.append([c.to_dict() for c in state["clips"]])
    del stack[:-UNDO_LIMIT]


def undo(state: MutableMapping) -> bool:
    if not state["undo_stack"]:
        return False
    state["clips"] = [Clip.from_dict(d) for d in state["undo_stack"].pop()]
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
                stem_settings: dict | None = None) -> str:
    payload = json.dumps({"clips": [c.to_dict() for c in clips], "settings": settings.to_dict(),
                          "track": track,
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
