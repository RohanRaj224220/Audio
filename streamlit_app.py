"""Media Audio Studio - Streamlit front end (UI only).

All processing lives in the backend modules; this file only wires widgets to
them and keeps per-user state in st.session_state.
"""

from __future__ import annotations

import io
import logging
import platform
import shutil
import sys
import time
from pathlib import Path

import streamlit as st

import audio_processor as ap
import audio_separator as sep
import compressor
import downloader
import editor
import ffmpeg_utils
import file_utils
import renderer
import state_manager as sm
import video_processor as vp
import waveform
from models import (EQ_BANDS, PITCH_RANGE, SPEED_RANGE, TONE_RANGE, TRACK_INSTRUMENTAL, TRACK_MIX,
                    TRACK_ORIGINAL, TRACK_VOCALS, ValidationError)

st.set_page_config(page_title="Media Audio Studio", page_icon="🎵", layout="wide")

log = logging.getLogger("mas.ui")

TRACK_LABELS = {TRACK_ORIGINAL: "Original", TRACK_INSTRUMENTAL: "Instrumental (no vocals)",
                TRACK_VOCALS: "Vocals only", TRACK_MIX: "Vocals + Instrumental (mix)"}
TRACK_SUFFIX = {TRACK_ORIGINAL: "edited", TRACK_INSTRUMENTAL: "instrumental",
                TRACK_VOCALS: "vocals", TRACK_MIX: "mix"}

MSG = {
    "ffmpeg": "FFmpeg is not available. The audio tools needed by this app are not installed.",
    "unreadable": "The uploaded file could not be read.",
    "unsupported": "This file format is unsupported.",
    "too_large": f"This file is too large. Please choose a file smaller than "
                 f"{file_utils.MAX_UPLOAD_MB:g} MB.",
    "sep_failed": "Vocal separation failed.",
    "memory": "There is not enough memory to process this file. Try a shorter song.",
    "range": "The selected time range is invalid.",
    "render": "Rendering failed.",
    "video": "Video export failed.",
}


# ----------------------------------------------------------------------------
# One-time process setup (safe, shared, read-only resources only)
# ----------------------------------------------------------------------------
@st.cache_resource
def boot() -> dict:
    file_utils.setup_logging()
    file_utils.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "streamlit": st.__version__,
        "ffmpeg": ffmpeg_utils.find_ffmpeg(),
        "ffmpeg_version": ffmpeg_utils.ffmpeg_version(),
        "rubberband": ap.rubberband_available(),
        "separation": sep.separation_available(),
    }
    if info["separation"]:
        try:
            import torch

            info["torch"] = torch.__version__
            info["device"] = sep.compute_device()
        except Exception as e:  # noqa: BLE001
            info["torch"] = f"import failed: {e}"
    log.info("Startup: %s", info)
    return {"info": info, "last_cleanup": 0.0}


def maybe_cleanup(runtime: dict) -> None:
    if time.time() - runtime["last_cleanup"] < 3600:
        return
    runtime["last_cleanup"] = time.time()
    try:
        file_utils.cleanup_old_jobs(active_ids=[st.session_state.job_id or ""])
    except Exception:  # noqa: BLE001
        log.exception("Cleanup failed")


@st.cache_data(max_entries=32, show_spinner=False)
def waveform_png(path: str, start: float, end: float, mtime: float) -> bytes:
    fig = waveform.waveform_figure(path, start, end)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    waveform.close(fig)
    return buf.getvalue()


@st.cache_data(max_entries=8, show_spinner=False)
def silence_png(duration: float) -> bytes:
    fig = waveform.silence_figure(duration)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    waveform.close(fig)
    return buf.getvalue()


def read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def flash(kind: str, text: str, where: str = "editor") -> None:
    st.session_state[f"flash_{where}"] = (kind, text)


def show_flash(where: str = "editor") -> None:
    msg = st.session_state.pop(f"flash_{where}", None)
    if msg:
        getattr(st, msg[0])(msg[1])


def current_fingerprint() -> str:
    return sm.fingerprint(st.session_state.clips, st.session_state.audio_settings,
                          st.session_state.current_track, st.session_state.stem_settings)


# ----------------------------------------------------------------------------
# Callbacks (run before the rerun, so the page updates immediately)
# ----------------------------------------------------------------------------
def do_edit(fn, *args) -> None:
    try:
        sm.apply_edit(st.session_state, fn, *args)
    except ValidationError as e:
        flash("warning", str(e))


def cb_undo() -> None:
    if not sm.undo(st.session_state):
        flash("info", "Nothing to undo.")


def cb_select(clip_id: str) -> None:
    st.session_state.selected_clip = clip_id


def cb_trim(clip_id: str, key_s: str, key_e: str) -> None:
    try:
        start = editor.parse_time(st.session_state[key_s])
        end = editor.parse_time(st.session_state[key_e])
        sm.apply_edit(st.session_state, editor.trim, clip_id, start, end,
                      st.session_state.project.info.duration)
        flash("success", "Clip trimmed.")
    except ValidationError as e:
        flash("warning", str(e))


def cb_split(clip_id: str, key: str) -> None:
    try:
        pos = editor.parse_time(st.session_state[key])
        sm.apply_edit(st.session_state, editor.split, clip_id, pos)
        flash("success", "Clip split in two.")
    except ValidationError as e:
        flash("warning", str(e))


def cb_silence(clip_id: str | None, key: str) -> None:
    do_edit(editor.add_silence, clip_id, float(st.session_state[key]))


def cb_fades(clip_id: str, key_in: str, key_out: str, key_vol: str) -> None:
    try:
        new = editor.set_fades(st.session_state.clips, clip_id, float(st.session_state[key_in]),
                               float(st.session_state[key_out]))
        new = editor.set_clip_volume(new, clip_id, float(st.session_state[key_vol]) / 100.0)
        sm.push_undo(st.session_state)
        st.session_state.clips = new
        sm.save_project(st.session_state)
        flash("success", "Clip settings saved.")
    except ValidationError as e:
        flash("warning", str(e))


def cb_preset(name: str, prefix: str = sm.MASTER) -> None:
    sm.init_widgets(st.session_state, ap.preset_settings(name), force=True, prefix=prefix)
    part = {"w": "whole song", "v": "vocals", "i": "instrumental"}[prefix]
    flash("success", f"Preset applied to {part}: {name}", where="sound")


def cb_new_project() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
def tab_import() -> None:
    st.subheader("Upload Song or Video")
    st.caption("Accepted: " + ", ".join(sorted(file_utils.ALLOWED_EXTENSIONS)).upper()
               + f" — up to {file_utils.MAX_UPLOAD_MB:g} MB")
    up = st.file_uploader("Upload Song or Video", type=sorted(file_utils.ALLOWED_EXTENSIONS),
                          label_visibility="collapsed")
    if up is not None:
        signature = f"{up.name}:{up.size}:{getattr(up, 'file_id', '')}"
        if signature != st.session_state.upload_signature:
            st.session_state.upload_signature = signature
            load_upload(up)

    st.markdown("**Or paste a YouTube or SoundCloud link**")
    st.radio("What to download", ["🎵 Audio (MP3)", "🎬 Video (MP4)"], key="import_kind",
             horizontal=True, label_visibility="collapsed")
    c1, c2 = st.columns([4, 1])
    c1.text_input("Link", key="import_url", label_visibility="collapsed",
                  placeholder="https://www.youtube.com/watch?v=...  or  https://soundcloud.com/...")
    if c2.button("⬇ Download & Import", type="primary", width="stretch"):
        kind = downloader.KIND_VIDEO if "MP4" in st.session_state.import_kind else downloader.KIND_AUDIO
        load_url(st.session_state.import_url, kind)
    st.caption("Only download songs and videos you own or have permission to use.")

    project = st.session_state.project
    if project is None:
        st.info("Start by uploading a song or a video.")
        return
    info = project.info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File", project.source_name if len(project.source_name) < 28
              else project.source_name[:25] + "…")
    c2.metric("Length", editor.format_time(info.duration))
    c3.metric("Type", file_utils.file_extension(project.source_name).upper())
    c4.metric("Contains", "Audio + Video" if info.has_video else "Audio")
    if project.origin_url:
        st.caption(f"Downloaded from {downloader.detect_site(project.origin_url)}: "
                   f"{project.origin_url}")
    if info.has_video:
        st.video(project.source_path)
    else:
        st.audio(project.working_audio)
    if project.origin_url and Path(project.source_path).exists():
        is_mp4 = Path(project.source_path).suffix.lower() == ".mp4"
        st.download_button("⬇ Download MP4" if is_mp4 else "⬇ Download MP3",
                           data=read_bytes(project.source_path),
                           file_name=Path(project.source_path).name,
                           mime="video/mp4" if is_mp4 else "audio/mpeg", key="dl_source_file")
    st.button("Start over with a different file", on_click=cb_new_project)


def load_upload(up) -> None:
    if up.size > file_utils.MAX_UPLOAD_MB * 1024 * 1024:
        st.error(MSG["too_large"])
        return
    try:
        file_utils.check_extension(up.name)
    except file_utils.UnsupportedFormatError:
        st.error(MSG["unsupported"])
        return
    job_id, job_dir = file_utils.new_job()
    try:
        with st.spinner("Opening your file..."):
            project = vp.import_media(job_id, job_dir, up.name, up, up.size)
    except file_utils.FileTooLargeError:
        st.error(MSG["too_large"])
        return
    except file_utils.UnsupportedFormatError:
        st.error(MSG["unsupported"])
        return
    except ffmpeg_utils.FFmpegNotFound:
        st.error(MSG["ffmpeg"])
        return
    except Exception:  # noqa: BLE001
        log.exception("Import failed for %s", up.name)
        st.error(MSG["unreadable"])
        return
    clips = editor.initial_clips(project.working_audio, project.info.duration)
    sm.reset_for_new_project(st.session_state, project, clips)
    st.success("File loaded. Use the tabs above to remove vocals, edit, or change the sound.")


def load_url(url: str, kind: str = downloader.KIND_AUDIO) -> None:
    try:
        site = downloader.detect_site(url)
    except downloader.DownloadError as e:
        st.warning(str(e))
        return
    job_id, job_dir = file_utils.new_job()
    bar = st.progress(0.0, text=f"Getting the song from {site}...")
    try:
        result = downloader.download_media(
            url, job_dir / "source", kind, progress=lambda p, msg: bar.progress(p, text=msg))
        bar.progress(1.0, text="Opening the file...")
        project = vp.import_file(job_id, job_dir, result.path, display_name=result.path.name,
                                 origin_url=result.url)
    except downloader.DownloadError as e:
        bar.empty()
        shutil.rmtree(job_dir, ignore_errors=True)
        st.error(str(e))
        return
    except Exception:  # noqa: BLE001
        log.exception("Import from link failed: %s", url)
        bar.empty()
        shutil.rmtree(job_dir, ignore_errors=True)
        st.error(MSG["unreadable"])
        return
    bar.empty()
    clips = editor.initial_clips(project.working_audio, project.info.duration)
    sm.reset_for_new_project(st.session_state, project, clips)
    st.success(f"Downloaded “{result.title}”. Use the tabs above to remove vocals, edit, "
               "or change the sound.")


def tab_vocals() -> None:
    project = st.session_state.project
    if project is None:
        st.info("Upload a song first (Import tab).")
        return
    st.subheader("Original Track")
    st.audio(project.working_audio)

    if not sep.separation_available():
        st.warning("Vocal removal is not available on this server. All other tools still work.")
    else:
        if sep.active_mode() == "best":
            st.write("Separates the singer's voice from the music using a high-quality AI model. "
                     "It takes about 1½ minutes per minute of song on a typical computer.")
        else:
            st.write("Separates the singer's voice from the music. This can take a few minutes.")
        st.checkbox("Extra bleed removal (recommended)", value=True, key="sep_clean",
                    help="Removes more of the music left in the vocals and of the voice left in "
                         "the music. Can make the sound slightly thinner; switch off and "
                         "separate again if you prefer.")
        label = "🎤 Separate Again" if project.stems else "🎤 Remove Vocals"
        if st.button(label, type="secondary" if project.stems else "primary"):
            run_separation(project)

    if project.stems:
        st.divider()
        downloads = st.session_state.stem_downloads
        base = Path(file_utils.sanitize_filename(project.source_name, "song")).stem
        c1, c2 = st.columns(2)
        for col, track, label in ((c1, TRACK_INSTRUMENTAL, "Instrumental"),
                                  (c2, TRACK_VOCALS, "Vocals")):
            with col:
                st.markdown(f"#### {label}")
                st.audio(project.stems[track])
                mp3 = downloads.get(track)
                if mp3 and Path(mp3).exists():
                    st.download_button(f"Download {label}", data=read_bytes(mp3),
                                       file_name=f"{base}_{track}.mp3", mime="audio/mpeg",
                                       key=f"dl_{track}", width="stretch")
        st.divider()
        choose_track(key="track_vocals_tab")


def choose_track(key: str) -> None:
    project = st.session_state.project
    options = [TRACK_ORIGINAL] + [t for t in (TRACK_INSTRUMENTAL, TRACK_VOCALS) if t in project.stems]
    if TRACK_INSTRUMENTAL in project.stems and TRACK_VOCALS in project.stems:
        options.append(TRACK_MIX)
    if len(options) == 1:
        return

    def _set():
        st.session_state.current_track = st.session_state[key]
        sm.save_project(st.session_state)

    st.radio("Which track do you want to edit and export?", options,
             index=options.index(st.session_state.current_track)
             if st.session_state.current_track in options else 0,
             format_func=TRACK_LABELS.get, key=key, on_change=_set, horizontal=True)


def run_separation(project) -> None:
    job_dir = file_utils.job_path(project.job_id)
    if not sep.model_is_cached():
        st.info("Downloading AI separation model on first use. This only happens once.")
    st.session_state.preview = None
    bar = st.progress(0.0, text="Separating vocals...")
    try:
        with st.spinner("Separating vocals..."):
            stems = sep.separate(Path(project.working_audio), job_dir / "stems",
                                 progress=lambda p, text: bar.progress(min(p, 1.0),
                                                                       text=text or "Separating vocals..."),
                                 clean=bool(st.session_state.get("sep_clean", True)))
            downloads = {}
            for track, path in stems.items():
                downloads[track] = str(renderer.convert_audio(
                    path, job_dir / "stems" / f"{track}.mp3", "mp3"))
    except sep.SeparationMemoryError:
        bar.empty()
        st.error(MSG["memory"])
        return
    except Exception:  # noqa: BLE001
        log.exception("Separation failed")
        bar.empty()
        st.error(MSG["sep_failed"])
        return
    bar.progress(1.0, text="Done")
    project.stems = {k: str(v) for k, v in stems.items()}
    st.session_state.stem_downloads = downloads
    st.session_state.current_track = TRACK_MIX
    sm.save_project(st.session_state)
    st.rerun()


def tab_editor() -> None:
    project = st.session_state.project
    if project is None:
        st.info("Upload a song first (Import tab).")
        return
    show_flash()
    clips = st.session_state.clips
    total = editor.timeline_duration(clips)
    top = st.columns([3, 1, 1])
    top[0].markdown(f"**{len(clips)} clip(s)** · total length **{editor.format_time(total)}** · "
                    f"editing: **{TRACK_LABELS[st.session_state.current_track]}**")
    top[1].button("↶ Undo", on_click=cb_undo, width="stretch",
                  disabled=not st.session_state.undo_stack)
    st.caption("Times can be typed as seconds (75.5) or minutes:seconds (1:15.5).")

    rows = editor.timeline_metadata(clips)
    st.dataframe(
        [{"#": r["#"], "Clip": r["name"], "From": editor.format_time(r["start"]) if r["kind"] == "audio"
          else "—", "To": editor.format_time(r["end"]) if r["kind"] == "audio" else "—",
          "Length": editor.format_time(r["length"]),
          "Plays at": editor.format_time(r["timeline_start"])} for r in rows],
        hide_index=True, width="stretch")

    ids = [c.id for c in clips]
    if st.session_state.selected_clip not in ids:
        st.session_state.selected_clip = ids[0]
    for n, clip in enumerate(clips, 1):
        selected = clip.id == st.session_state.selected_clip
        title = f"{n}. {clip.name} — {editor.format_time(clip.duration)}"
        with st.expander(("▶ " if selected else "") + title, expanded=selected):
            clip_panel(n, clip, len(clips))

    st.divider()
    c1, c2 = st.columns([1, 3])
    c1.number_input("Silence length (seconds)", 0.1, 600.0, 2.0, 0.5, key="sil_end")
    c2.write("")
    c2.button("Add Silence at the end", on_click=cb_silence, args=(None, "sil_end"))


def clip_panel(n: int, clip, count: int) -> None:
    project = st.session_state.project
    sig = f"{clip.id}_{clip.start}_{clip.end}"
    if clip.is_silence:
        st.write(f"Silence · length **{editor.format_time(clip.duration)}**")
    else:
        st.write(f"From **{editor.format_time(clip.start)}** to **{editor.format_time(clip.end)}** · "
                 f"length **{editor.format_time(clip.duration)}**")

    if clip.id == st.session_state.selected_clip:
        if clip.is_silence:
            st.image(silence_png(clip.duration), width="stretch")
        else:
            track = st.session_state.current_track
            path = project.working_audio if track == TRACK_MIX else project.track_path(track)
            st.image(waveform_png(path, clip.start, clip.end, Path(path).stat().st_mtime),
                     width="stretch")
    else:
        st.button("Show waveform", key=f"sel_{clip.id}", on_click=cb_select, args=(clip.id,))

    b = st.columns(4)
    b[0].button("⬆ Move Up", key=f"up_{clip.id}", disabled=n == 1, width="stretch",
                on_click=do_edit, args=(editor.move, clip.id, -1))
    b[1].button("⬇ Move Down", key=f"dn_{clip.id}", disabled=n == count, width="stretch",
                on_click=do_edit, args=(editor.move, clip.id, 1))
    b[2].button("Duplicate", key=f"dup_{clip.id}", width="stretch",
                on_click=do_edit, args=(editor.duplicate, clip.id))
    b[3].button("🗑 Delete", key=f"del_{clip.id}", disabled=count == 1, width="stretch",
                on_click=do_edit, args=(editor.delete, clip.id))

    t1, t2, t3 = st.columns(3)
    with t1.form(f"trim_{sig}", border=True):
        st.markdown("**Trim**")
        if clip.is_silence:
            st.text_input("Start", "0", key=f"ts_{sig}", disabled=True)
            st.text_input("Length", editor.format_time(clip.duration), key=f"te_{sig}")
        else:
            st.text_input("Start time", editor.format_time(clip.start), key=f"ts_{sig}")
            st.text_input("End time", editor.format_time(clip.end), key=f"te_{sig}")
        st.form_submit_button("Apply Trim", on_click=cb_trim, args=(clip.id, f"ts_{sig}", f"te_{sig}"))
    with t2.form(f"split_{sig}", border=True):
        st.markdown("**Split**")
        st.text_input("Split position", editor.format_time((clip.start + clip.end) / 2),
                      key=f"sp_{sig}")
        st.form_submit_button("Split Here", on_click=cb_split, args=(clip.id, f"sp_{sig}"))
    with t3.form(f"sil_{sig}", border=True):
        st.markdown("**Add Silence after this clip**")
        st.number_input("Seconds", 0.1, 600.0, 2.0, 0.5, key=f"sl_{sig}")
        st.form_submit_button("Add Silence", on_click=cb_silence, args=(clip.id, f"sl_{sig}"))

    if not clip.is_silence:
        fsig = f"{sig}_{clip.fade_in}_{clip.fade_out}_{clip.volume}"
        with st.form(f"fade_{fsig}", border=True):
            f1, f2, f3 = st.columns(3)
            max_fade = float(max(0.0, min(30.0, clip.duration)))
            f1.number_input("Fade in (seconds)", 0.0, max_fade, float(min(clip.fade_in, max_fade)),
                            0.5, key=f"fi_{fsig}")
            f2.number_input("Fade out (seconds)", 0.0, max_fade, float(min(clip.fade_out, max_fade)),
                            0.5, key=f"fo_{fsig}")
            f3.number_input("Clip volume (%)", 0, 200, int(round(clip.volume * 100)), 5,
                            key=f"cv_{fsig}")
            st.form_submit_button("Save Fades & Volume", on_click=cb_fades,
                                  args=(clip.id, f"fi_{fsig}", f"fo_{fsig}", f"cv_{fsig}"))


def settings_panel(prefix: str) -> None:
    """Presets + sliders for one part (whole song / vocals / instrumental)."""
    cols = st.columns(len(ap.PRESETS))
    for col, name in zip(cols, ap.PRESETS):
        col.button(name, key=f"preset_{prefix}_{name}" if prefix != sm.MASTER else f"preset_{name}",
                   on_click=cb_preset, args=(name, prefix), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Speed", SPEED_RANGE[0], SPEED_RANGE[1], step=0.05, format="%.2fx",
                  key=sm.key(prefix, "speed"), help="Faster or slower without changing the key.")
        st.slider("Pitch (semitones)", PITCH_RANGE[0], PITCH_RANGE[1], step=1,
                  key=sm.key(prefix, "pitch"), help="Higher or lower key without changing the speed.")
        st.slider("Volume", 0, 200, step=5, format="%d%%", key=sm.key(prefix, "volume"))
    with c2:
        for name in ("bass", "mid", "treble"):
            st.slider(name.title(), TONE_RANGE[0], TONE_RANGE[1], step=0.5, format="%+.1f dB",
                      key=sm.key(prefix, name))
    with st.expander("Advanced Equalizer"):
        cols = st.columns(4)
        for i, band in enumerate(EQ_BANDS):
            label = f"{band // 1000} kHz" if band >= 1000 else f"{band} Hz"
            cols[i % 4].slider(label, -12.0, 12.0, step=0.5, format="%+.1f dB",
                               key=sm.eq_key(band, prefix))


def tab_sound() -> None:
    show_flash("sound")
    project = st.session_state.project
    has_stems = bool(project and all(t in project.stems for t in (TRACK_VOCALS, TRACK_INSTRUMENTAL)))
    if not has_stems:
        st.caption("Tip: after Remove Vocals you can also change the vocals and the music separately.")
        settings_panel(sm.MASTER)
    else:
        whole, voc, inst = st.tabs(["🎵 Whole Song", "🎤 Vocals", "🎸 Instrumental"])
        with whole:
            st.caption("Applied on top of everything (vocals and instrumental together).")
            settings_panel(sm.MASTER)
        with voc:
            st.caption("Only changes the singing.")
            settings_panel("v")
        with inst:
            st.caption("Only changes the music.")
            settings_panel("i")
        stems = st.session_state.stem_settings
        if abs(stems[TRACK_VOCALS].speed - stems[TRACK_INSTRUMENTAL].speed) > 1e-6:
            st.warning("Vocals and Instrumental have different speeds, so they will drift apart "
                       "in the mix. Set both to the same speed to keep them together.")
        if st.session_state.current_track != TRACK_MIX:
            st.info(f"You are currently using: **{TRACK_LABELS[st.session_state.current_track]}**. "
                    "Choose “Vocals + Instrumental (mix)” in Preview to hear both parts together.")
    st.caption("Changes are applied when you click Render Preview or Export.")


def tab_preview() -> None:
    project = st.session_state.project
    if project is None:
        st.info("Upload a song first (Import tab).")
        return
    if project.stems:
        choose_track(key="track_preview_tab")
    clips = st.session_state.clips
    fp = current_fingerprint()
    speed = sm.effective_speed(st.session_state)
    st.write(f"Track: **{TRACK_LABELS[st.session_state.current_track]}** · {len(clips)} clip(s) · "
             f"length after speed change: **{editor.format_time(editor.timeline_duration(clips) / speed)}**")
    if st.button("▶ Render Preview", type="primary"):
        job_dir = file_utils.job_path(project.job_id)
        out = file_utils.unique_render_path(job_dir, "preview", "mp3")
        try:
            with st.spinner("Rendering preview..."):
                renderer.render_layers(clips, sm.render_layers(st.session_state), out, "preview")
        except ValidationError as e:
            st.error(str(e))
            return
        except Exception:  # noqa: BLE001
            log.exception("Preview failed")
            st.error(MSG["render"])
            return
        old = st.session_state.preview
        if old and Path(old["path"]).exists():
            Path(old["path"]).unlink(missing_ok=True)
        st.session_state.preview = {"path": str(out), "fingerprint": fp}

    prev = st.session_state.preview
    if prev and Path(prev["path"]).exists():
        if prev["fingerprint"] != fp:
            st.warning("You changed something since this preview. Click Render Preview to hear it.")
        st.audio(prev["path"])


MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac", "mp4": "video/mp4"}


def export_names(project) -> tuple[Path, str]:
    job_dir = file_utils.job_path(project.job_id)
    base = Path(file_utils.sanitize_filename(project.source_name, "song")).stem
    return job_dir, f"{base}_{TRACK_SUFFIX[st.session_state.current_track]}"


def render_audio_export(project, out: Path, fmt: str) -> Path:
    return renderer.render_layers(st.session_state.clips, sm.render_layers(st.session_state), out, fmt)


def render_video_export(project, out: Path) -> Path:
    job_dir, _ = export_names(project)
    audio = job_dir / "renders" / "video_audio.wav"
    try:
        render_audio_export(project, audio, "wav")
        return vp.export_video(project.info, st.session_state.clips,
                               sm.effective_speed(st.session_state), audio, out)
    finally:
        audio.unlink(missing_ok=True)


def tab_export() -> None:
    project = st.session_state.project
    if project is None:
        st.info("Upload a song first (Import tab).")
        return
    choices = {"MP3 (320 kbps)": "mp3", "WAV (CD quality)": "wav",
               "FLAC (lossless, about half the size of WAV)": "flac", "MP4 (video)": "mp4"}
    label = st.selectbox("Format", list(choices))
    fmt = choices[label]
    job_dir, name = export_names(project)
    if project.stems:
        choose_track(key="track_export_tab")
    st.write(f"Exporting: **{TRACK_LABELS[st.session_state.current_track]}**")

    if fmt in ("mp3", "wav", "flac"):
        if st.button("💾 Export Audio", type="primary"):
            out = job_dir / "renders" / f"{name}.{fmt}"
            try:
                with st.spinner("Exporting..."):
                    render_audio_export(project, out, fmt)
                st.session_state.exports[fmt] = str(out)
            except ValidationError as e:
                st.error(str(e))
            except Exception:  # noqa: BLE001
                log.exception("Audio export failed")
                st.error(MSG["render"])
    elif not project.info.has_video:
        st.info("Video export is available when you upload a video file.")
    elif st.button("🎬 Export Video", type="primary"):
        out = job_dir / "renders" / f"{name}.mp4"
        try:
            with st.spinner("Exporting video... this can take a while."):
                render_video_export(project, out)
            st.session_state.exports["mp4"] = str(out)
        except ValidationError as e:
            st.error(str(e))
        except renderer.RenderError:
            st.error(MSG["render"])
        except Exception:  # noqa: BLE001
            log.exception("Video export failed")
            st.error(MSG["video"])

    path = st.session_state.exports.get(fmt)
    if path and Path(path).exists():
        st.success(f"Your file is ready ({Path(path).stat().st_size / compressor.MB:.1f} MB).")
        st.download_button(f"⬇ Download {Path(path).name}", data=read_bytes(path),
                           file_name=Path(path).name, mime=MIME[fmt], type="primary")

    st.divider()
    whatsapp_section(project)


def whatsapp_section(project) -> None:
    st.subheader("📦 Compress for WhatsApp")
    st.caption("Makes **one file** under your size limit. It stays lossless (identical quality) "
               "whenever it fits; otherwise it is compressed to the best quality that fits. "
               "Tip: in WhatsApp send it as a **Document** (📎 → Document) so WhatsApp doesn't "
               "compress it again.")
    c1, c2 = st.columns([1, 2])
    limit = c1.number_input("Size limit (MB)", min_value=1.0, max_value=4000.0, value=180.0,
                            step=1.0, key="share_limit")
    kinds = ["🎵 Audio"] + (["🎬 Video"] if project.info.has_video else [])
    kind = c2.radio("Send", kinds, key="share_kind", horizontal=True)
    if st.button("📦 Compress", type="primary"):
        prepare_share(project, float(limit), kind.endswith("Video"))

    result = st.session_state.get("share_result")
    if result and Path(result["path"]).exists():
        path = Path(result["path"])
        if result["lossless"]:
            st.success(f"Ready: **{path.name}** — {result['mb']:.1f} MB, lossless (identical quality).")
        else:
            st.success(f"Ready: **{path.name}** — {result['mb']:.1f} MB, fits your "
                       f"{result['limit']:g} MB limit.")
            st.info(f"The lossless version would have been {result['lossless_mb']:.1f} MB, which "
                    "is over your limit, so this file is compressed (small quality loss).")
        st.download_button(f"⬇ Download {path.name}", data=read_bytes(path), file_name=path.name,
                           mime=MIME.get(path.suffix.lstrip("."), "application/octet-stream"),
                           key="share_dl", type="primary")


def video_is_unchanged(project) -> bool:
    """True when the export would just reproduce the original video file."""
    return (st.session_state.current_track == TRACK_ORIGINAL
            and editor.is_untouched(st.session_state.clips, project.info.duration)
            and st.session_state.audio_settings.is_neutral
            and Path(project.source_path).suffix.lower() == ".mp4")


def prepare_share(project, limit_mb: float, is_video: bool) -> None:
    job_dir, name = export_names(project)
    out_dir = job_dir / "share"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    st.session_state.share_result = None
    try:
        with st.spinner("Compressing..."):
            if is_video and video_is_unchanged(project):
                master = out_dir / f"{name}.mp4"
                shutil.copy(project.source_path, master)  # untouched original, bit for bit
            elif is_video:
                master = render_video_export(project, out_dir / f"{name}.mp4")
            else:
                master = render_audio_export(project, out_dir / f"{name}_master.wav", "wav")
            part = compressor.compress_single(master, out_dir, limit_mb, name, is_video)
            for leftover in out_dir.iterdir():
                if leftover.is_file() and leftover != part.path:
                    leftover.unlink()
    except (compressor.CompressionError, ValidationError) as e:
        st.error(str(e))
        return
    except Exception:  # noqa: BLE001
        log.exception("Compression failed")
        st.error(MSG["video"] if is_video else MSG["render"])
        return
    st.session_state.share_result = {"path": str(part.path), "mb": part.size_mb,
                                     "lossless": part.lossless, "limit": limit_mb,
                                     "lossless_mb": part.lossless_size / compressor.MB}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    runtime = boot()
    sm.init_state(st.session_state)
    if st.session_state.job_id:
        file_utils.touch_job(st.session_state.job_id)
    maybe_cleanup(runtime)

    st.title("🎵 Media Audio Studio")
    if not runtime["info"]["ffmpeg"]:
        st.error(MSG["ffmpeg"])
        st.stop()

    try:
        sm.sync_settings_from_widgets(st.session_state)
    except ValidationError:
        for prefix in sm.PREFIXES:
            sm.init_widgets(st.session_state, force=True, prefix=prefix)
        sm.sync_settings_from_widgets(st.session_state)

    tabs = st.tabs(["📂 Import", "🎤 Vocal Remover", "✂️ Editor", "🎚 Sound Controls",
                    "▶ Preview", "💾 Export"])
    with tabs[0]:
        tab_import()
    with tabs[1]:
        tab_vocals()
    with tabs[2]:
        tab_editor()
    with tabs[3]:
        tab_sound()
    with tabs[4]:
        tab_preview()
    with tabs[5]:
        tab_export()


main()
