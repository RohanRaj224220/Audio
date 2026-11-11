"""Media Audio Studio - Streamlit front end (UI only).

All processing lives in the backend modules; this file only wires widgets to
them and keeps per-user state in st.session_state.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import audio_processor as ap
import audio_separator as sep
import compressor
import downloader
import editor
import ffmpeg_utils
import file_utils
import lyrics
import renderer
import state_manager as sm
import video_processor as vp
import waveform
from models import (EQ_BANDS, INSTRUMENT_STEMS, PITCH_RANGE, SPEED_RANGE, TONE_RANGE,
                    TRACK_INSTRUMENTAL, TRACK_MIX, TRACK_ORIGINAL, TRACK_VOCALS, Clip, Overlay,
                    ValidationError)

st.set_page_config(page_title="Media Audio Studio", page_icon="🎵", layout="wide")

log = logging.getLogger("mas.ui")

PART_LABELS = {"drums": "🥁 Drums", "bass": "🎸 Bass", "guitar": "🎸 Guitar", "piano": "🎹 Piano",
               "other": "🎻 Other instruments"}
TRACK_LABELS = {TRACK_ORIGINAL: "Original", TRACK_INSTRUMENTAL: "Instrumental (no vocals)",
                TRACK_VOCALS: "Vocals only", TRACK_MIX: "Vocals + Instrumental (mix)",
                **{p: f"{label} only" for p, label in PART_LABELS.items()}}
TRACK_SUFFIX = {TRACK_ORIGINAL: "edited", TRACK_INSTRUMENTAL: "instrumental",
                TRACK_VOCALS: "vocals", TRACK_MIX: "mix", **{p: p for p in INSTRUMENT_STEMS}}

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


def download(label: str, path, name: str, mime: str, key: str, **kwargs) -> None:
    """Download button that only reads the file when it is clicked (keeps reruns fast)."""
    st.download_button(label, data=lambda p=str(path): Path(p).read_bytes(), file_name=name,
                       mime=mime, key=key, **kwargs)


def zip_bytes(files: dict[str, str]) -> bytes:
    """ZIP without re-compression (FLAC is already compressed)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, path in files.items():
            z.write(path, name)
    return buf.getvalue()


def flash(kind: str, text: str, where: str = "editor") -> None:
    st.session_state[f"flash_{where}"] = (kind, text)


def show_flash(where: str = "editor") -> None:
    msg = st.session_state.pop(f"flash_{where}", None)
    if msg:
        getattr(st, msg[0])(msg[1])


def current_fingerprint() -> str:
    return sm.fingerprint(st.session_state.clips, st.session_state.audio_settings,
                          st.session_state.current_track, st.session_state.stem_settings,
                          st.session_state.overlays)


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


def cb_redo() -> None:
    if not sm.redo(st.session_state):
        flash("info", "Nothing to redo.")


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


def cb_split(clip_id: str, key: str, key_n: str | None = None) -> None:
    try:
        pieces = int(st.session_state.get(key_n, 0) or 0) if key_n else 0
        if pieces >= 2:
            sm.apply_edit(st.session_state, editor.split_equal, clip_id, pieces)
            flash("success", f"Clip split into {pieces} equal pieces.")
            return
        positions = editor.parse_times(st.session_state[key])
        sm.apply_edit(st.session_state, editor.split_many, clip_id, positions)
        flash("success", f"Clip split into {len(positions) + 1} pieces.")
    except ValidationError as e:
        flash("warning", str(e))


def cb_silence(clip_id: str | None, key: str) -> None:
    do_edit(editor.add_silence, clip_id, float(st.session_state[key]))


def cb_fades(clip_id: str, key_in: str, key_out: str, key_vol: str, key_speed: str | None = None) -> None:
    try:
        new = editor.set_fades(st.session_state.clips, clip_id, float(st.session_state[key_in]),
                               float(st.session_state[key_out]))
        new = editor.set_clip_volume(new, clip_id, float(st.session_state[key_vol]) / 100.0)
        if key_speed:
            new = editor.set_clip_speed(new, clip_id, float(st.session_state[key_speed]))
        sm.push_undo(st.session_state)
        st.session_state.clips = new
        sm.save_project(st.session_state)
        flash("success", "Clip settings saved.")
    except ValidationError as e:
        flash("warning", str(e))


def cb_preset(name: str, prefix: str = sm.MASTER) -> None:
    sm.init_widgets(st.session_state, ap.preset_settings(name), force=True, prefix=prefix)
    part = sm.PREFIXES[prefix] or "whole song"
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
        download("⬇ Download MP4" if is_mp4 else "⬇ Download MP3", project.source_path,
                 Path(project.source_path).name, "video/mp4" if is_mp4 else "audio/mpeg",
                 key="dl_source_file")
    with st.expander("➕ Merge another song or video onto the end"):
        st.caption("Joins a second file after this one, so both can be edited as one project "
                   "(MP3, MP4 and the other accepted types can be mixed). "
                   "Do this before Remove Vocals — separated tracks are reset by a merge.")
        with st.form("merge_form", clear_on_submit=True, border=False):
            extra = st.file_uploader("File to add", type=sorted(file_utils.ALLOWED_EXTENSIONS),
                                     key="merge_file", label_visibility="collapsed")
            if st.form_submit_button("🔗 Merge onto the end") and extra is not None:
                merge_upload(project, extra)
    st.button("Start over with a different file", on_click=cb_new_project)


def merge_upload(project, up) -> None:
    if up.size > file_utils.MAX_UPLOAD_MB * 1024 * 1024:
        st.error(MSG["too_large"])
        return
    job_dir = file_utils.job_path(project.job_id)
    try:
        with st.spinner("Merging the files..."):
            merged, old_len = vp.merge_media(project, job_dir, up.name, up, up.size)
    except file_utils.FileTooLargeError:
        st.error(MSG["too_large"])
        return
    except file_utils.UnsupportedFormatError:
        st.error(MSG["unsupported"])
        return
    except Exception:  # noqa: BLE001
        log.exception("Merge failed for %s", up.name)
        st.error(MSG["unreadable"])
        return
    state = st.session_state
    sm.push_undo(state)
    state.project = merged
    state.clips = state.clips + [Clip(start=round(old_len, 3), end=merged.info.duration,
                                      source_path=merged.working_audio,
                                      name=Path(up.name).stem[:40] or "Added file")]
    state.current_track = TRACK_ORIGINAL
    state.preview, state.exports, state.share_result = None, {}, None
    state.stem_downloads, state.pieces = {}, {}
    sm.save_project(state)
    st.rerun()


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
        c1, c2 = st.columns(2)
        for col, track, label in ((c1, TRACK_INSTRUMENTAL, "Instrumental"),
                                  (c2, TRACK_VOCALS, "Vocals")):
            with col:
                piece_card(project, track, label)

        st.divider()
        st.subheader("Split the music into instruments")
        parts = [p for p in INSTRUMENT_STEMS if p in project.stems]
        st.write("Splits the instrumental into separate pieces: drums, bass, guitar, piano and "
                 "everything else. Only instruments that are in the song get a piece; "
                 "instruments without their own piece (flute, strings, synths…) are in "
                 "“Other instruments”. Takes about ½ minute per minute of song.")
        if st.button("🥁 Split Again" if parts else "🥁 Split into Instruments",
                     type="secondary" if parts else "primary"):
            run_instrument_split(project)
        if parts:
            cols = st.columns(3)
            for n, part in enumerate(parts):
                with cols[n % 3]:
                    piece_card(project, part, PART_LABELS[part])
            flacs = {f"{piece_base(project)}_{t}.flac": p
                     for t, p in st.session_state.pieces.items() if Path(p).exists()}
            if flacs:
                st.download_button("📦 Download every piece (ZIP, lossless FLAC)",
                                   data=lambda f=flacs: zip_bytes(f),
                                   file_name=f"{piece_base(project)}_pieces.zip",
                                   mime="application/zip", key="dl_pieces_zip", type="primary")
            st.caption("To put a piece somewhere else in the song, use **Add a piece on top of "
                       "the song** in the Editor tab. To remove or change one instrument, use "
                       "its tab in Sound Controls (volume 0% removes it).")
        st.divider()
        choose_track(key="track_vocals_tab")


def piece_base(project) -> str:
    return Path(file_utils.sanitize_filename(project.source_name, "song")).stem


def piece_card(project, track: str, label: str) -> None:
    """Player + MP3 and lossless downloads for one separated piece."""
    st.markdown(f"#### {label}")
    mp3 = st.session_state.stem_downloads.get(track)
    flac = st.session_state.pieces.get(track)
    has_mp3 = bool(mp3 and Path(mp3).exists())
    st.audio(mp3 if has_mp3 else project.stems[track])
    b1, b2 = st.columns(2)
    if has_mp3:
        download("⬇ MP3", mp3, f"{piece_base(project)}_{track}.mp3", "audio/mpeg",
                 key=f"dl_{track}", width="stretch")
    if flac and Path(flac).exists():
        with b2:
            download("⬇ FLAC (lossless)", flac, f"{piece_base(project)}_{track}.flac",
                     "audio/flac", key=f"dlf_{track}", width="stretch")


def make_piece_files(job_dir: Path, stems: dict) -> None:
    """MP3 (for playing/sharing) and FLAC (lossless) copies of each separated piece."""
    for track, path in stems.items():
        st.session_state.stem_downloads[track] = str(renderer.convert_audio(
            path, job_dir / "stems" / f"{track}.mp3", "mp3"))
        st.session_state.pieces[track] = str(renderer.convert_audio(
            path, job_dir / "stems" / f"{track}.flac", "flac"))


def run_instrument_split(project) -> None:
    job_dir = file_utils.job_path(project.job_id)
    st.session_state.preview = None
    bar = st.progress(0.0, text="Separating instruments...")
    try:
        parts = sep.split_instruments(
            Path(project.stems[TRACK_INSTRUMENTAL]), job_dir / "stems",
            progress=lambda p, text: bar.progress(min(p, 1.0), text=text or "Separating..."))
        for old in INSTRUMENT_STEMS:
            project.stems.pop(old, None)
            st.session_state.stem_downloads.pop(old, None)
            st.session_state.pieces.pop(old, None)
        make_piece_files(job_dir, parts)
    except sep.SeparationMemoryError:
        bar.empty()
        st.error(MSG["memory"])
        return
    except Exception:  # noqa: BLE001
        log.exception("Instrument split failed")
        bar.empty()
        st.error("Splitting into instruments failed.")
        return
    project.stems.update({k: str(v) for k, v in parts.items()})
    sm.save_project(st.session_state)
    st.rerun()


def choose_track(key: str) -> None:
    project = st.session_state.project
    options = [TRACK_ORIGINAL]
    if TRACK_INSTRUMENTAL in project.stems and TRACK_VOCALS in project.stems:
        options.append(TRACK_MIX)
    options += [t for t in (TRACK_INSTRUMENTAL, TRACK_VOCALS) + INSTRUMENT_STEMS if t in project.stems]
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
            st.session_state.stem_downloads, st.session_state.pieces = {}, {}
            make_piece_files(job_dir, stems)
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
    top = st.columns([3, 1, 1, 1.4])
    top[0].markdown(f"**{len(clips)} clip(s)** · total length **{editor.format_time(total)}** · "
                    f"editing: **{TRACK_LABELS[st.session_state.current_track]}**")
    top[1].button("↶ Undo", on_click=cb_undo, width="stretch",
                  disabled=not st.session_state.undo_stack)
    top[2].button("↷ Redo", on_click=cb_redo, width="stretch",
                  disabled=not st.session_state.redo_stack)
    top[3].button("🔗 Merge all joinable clips", on_click=do_edit, args=(editor.merge_all,),
                  width="stretch", help="Turns neighbouring pieces that continue into each other "
                                        "(for example after a split) back into one clip.",
                  disabled=not any(editor.can_merge(a, b) for a, b in zip(clips, clips[1:])))
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
    st.divider()
    overlay_section(project)


def overlay_sources(project) -> dict[str, str]:
    sources = {"Original song": project.working_audio}
    for track in (TRACK_VOCALS, TRACK_INSTRUMENTAL) + INSTRUMENT_STEMS:
        if track in project.stems:
            sources[PART_LABELS.get(track, track.title())] = project.stems[track]
    sources.update(st.session_state.extras)
    return sources


def cb_add_overlay(sources: dict) -> None:
    state = st.session_state
    try:
        name = state.ov_source
        path = sources[name]
        length = ffmpeg_utils.probe(path).duration
        start = editor.parse_time(state.ov_from)
        end = min(editor.parse_time(state.ov_to), round(length, 3)) if str(state.ov_to).strip() \
            else round(length, 3)
        position = editor.parse_time(state.ov_at)
        if end - start < editor.MIN_CLIP_SECONDS:
            raise ValidationError("The selected time range is invalid.")
        state.overlays = state.overlays + [Overlay(
            source_path=path, start=start, end=end, position=position,
            volume=float(state.ov_volume) / 100.0, name=name)]
        sm.save_project(state)
        flash("success", f"Added “{name}” at {editor.format_time(position)}.", where="overlay")
    except (ValidationError, KeyError, ffmpeg_utils.ProbeError) as e:
        flash("warning", str(e) or "That piece could not be added.", where="overlay")


def cb_remove_overlay(overlay_id: str) -> None:
    st.session_state.overlays = [o for o in st.session_state.overlays if o.id != overlay_id]
    sm.save_project(st.session_state)


def overlay_section(project) -> None:
    st.subheader("➕ Add a piece on top of the song")
    st.caption("Lay any sound over the song wherever you want: a separated instrument (for "
               "example the drums from one part placed in another part), or a new instrument or "
               "beat from a file of your own. The song underneath keeps playing.")
    show_flash("overlay")
    with st.expander("Use a sound from another file (new instrument, beat, voice…)"):
        with st.form("extra_form", clear_on_submit=True, border=False):
            up = st.file_uploader("Sound file", type=sorted(file_utils.ALLOWED_EXTENSIONS),
                                  key="extra_file", label_visibility="collapsed")
            if st.form_submit_button("Add to my sounds") and up is not None:
                try:
                    wav = vp.import_extra_audio(file_utils.job_path(project.job_id), up.name, up,
                                                up.size)
                    st.session_state.extras = {**st.session_state.extras,
                                               f"📎 {Path(up.name).stem[:40]}": str(wav)}
                except file_utils.FileTooLargeError:
                    st.error(MSG["too_large"])
                except Exception:  # noqa: BLE001
                    log.exception("Extra sound import failed")
                    st.error(MSG["unreadable"])
    sources = overlay_sources(project)
    with st.form("overlay_form", border=True):
        c = st.columns([2, 1, 1, 1, 1])
        c[0].selectbox("Sound", list(sources), key="ov_source")
        c[1].text_input("From", "0:00.000", key="ov_from", help="Where the piece starts in that sound.")
        c[2].text_input("To", "", key="ov_to", placeholder="end", help="Leave empty for the end.")
        c[3].text_input("Place at", "0:00.000", key="ov_at",
                        help="Where it starts playing in the finished song.")
        c[4].number_input("Volume (%)", 0, 200, 100, 5, key="ov_volume")
        st.form_submit_button("➕ Add to the song", on_click=cb_add_overlay, args=(sources,))
    for ov in st.session_state.overlays:
        row = st.columns([5, 1])
        row[0].markdown(f"**{ov.name}** · {editor.format_time(ov.start)} → "
                        f"{editor.format_time(ov.end)} · plays at **{editor.format_time(ov.position)}**"
                        f" · {int(round(ov.volume * 100))}%")
        row[1].button("🗑 Remove", key=f"ovdel_{ov.id}", on_click=cb_remove_overlay, args=(ov.id,),
                      width="stretch")


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

    clips = st.session_state.clips
    b = st.columns(5)
    b[4].button("🔗 Merge with next", key=f"mrg_{clip.id}", width="stretch",
                disabled=n == count or not editor.can_merge(clip, clips[n]),
                on_click=do_edit, args=(editor.merge, clip.id),
                help="Joins this clip and the next one into a single clip. Available when the "
                     "next clip continues where this one stops, or both are silence.")
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
        st.text_input("Split position(s)", editor.format_time((clip.start + clip.end) / 2),
                      key=f"sp_{sig}", help="Several positions at once: separate them with spaces, "
                                            "for example 0:30 1:05 1:40")
        st.number_input("…or into equal pieces (0 = use positions)", 0, 100, 0, 1, key=f"spn_{sig}")
        st.form_submit_button("Split", on_click=cb_split, args=(clip.id, f"sp_{sig}", f"spn_{sig}"))
    with t3.form(f"sil_{sig}", border=True):
        st.markdown("**Add Silence after this clip**")
        st.number_input("Seconds", 0.1, 600.0, 2.0, 0.5, key=f"sl_{sig}")
        st.form_submit_button("Add Silence", on_click=cb_silence, args=(clip.id, f"sl_{sig}"))

    if not clip.is_silence:
        fsig = f"{sig}_{clip.fade_in}_{clip.fade_out}_{clip.volume}_{clip.speed}"
        with st.form(f"fade_{fsig}", border=True):
            f1, f2, f3, f4 = st.columns(4)
            f4.number_input("Clip speed (x)", 0.5, 2.0, float(clip.speed), 0.05, key=f"cs_{fsig}",
                            help="Speed of this clip only (the key stays the same).")
            max_fade = float(max(0.0, min(30.0, clip.duration)))
            f1.number_input("Fade in (seconds)", 0.0, max_fade, float(min(clip.fade_in, max_fade)),
                            0.5, key=f"fi_{fsig}")
            f2.number_input("Fade out (seconds)", 0.0, max_fade, float(min(clip.fade_out, max_fade)),
                            0.5, key=f"fo_{fsig}")
            f3.number_input("Clip volume (%)", 0, 200, int(round(clip.volume * 100)), 5,
                            key=f"cv_{fsig}")
            st.form_submit_button("Save Fades, Volume & Speed", on_click=cb_fades,
                                  args=(clip.id, f"fi_{fsig}", f"fo_{fsig}", f"cv_{fsig}",
                                        f"cs_{fsig}"))


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
        parts = [p for p in INSTRUMENT_STEMS if p in project.stems]
        whole, voc, inst, *part_tabs = st.tabs(["🎵 Whole Song", "🎤 Vocals", "🎼 Instrumental"]
                                               + [PART_LABELS[p] for p in parts])
        for tab, part in zip(part_tabs, parts):
            with tab:
                st.caption("Only changes this instrument. Volume 0% removes it from the song.")
                settings_panel(sm.STEM_PREFIX[part])
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
        bus = stems[TRACK_INSTRUMENTAL].speed
        speeds = [stems[TRACK_VOCALS].speed] + ([bus * stems[p].speed for p in parts] or [bus])
        if max(speeds) - min(speeds) > 1e-6:
            st.warning("Some parts have different speeds, so they will drift apart "
                       "in the mix. Set them to the same speed to keep them together.")
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
                renderer.render_layers(clips, sm.render_layers(st.session_state), out, "preview",
                                       overlays=st.session_state.overlays)
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


LANGUAGES = {"Detect automatically": None, "English": "en", "Hindi": "hi", "Tamil": "ta",
             "Telugu": "te", "Malayalam": "ml", "Kannada": "kn", "Punjabi": "pa", "Bengali": "bn",
             "Urdu": "ur", "Spanish": "es", "French": "fr", "Portuguese": "pt", "Arabic": "ar",
             "Korean": "ko", "Japanese": "ja"}

KARAOKE_HTML = """
<style>
 body{margin:0;font-family:system-ui,sans-serif;background:#111;color:#777}
 audio{width:100%}
 #box{height:250px;overflow:hidden;text-align:center;padding:8px 12px}
 .l{font-size:20px;margin:10px 0;transition:all .2s}
 .l.on{font-size:28px;color:#fff;font-weight:600}
 .w.sung{color:#ffd400}
</style>
<audio id="a" controls src="data:audio/mpeg;base64,__AUDIO__"></audio>
<div id="box"></div>
<script>
 const L=__LINES__, a=document.getElementById('a'), box=document.getElementById('box');
 L.forEach((l,i)=>{const d=document.createElement('div');d.className='l';d.id='l'+i;
   l.words.forEach((w,j)=>{const s=document.createElement('span');s.className='w';s.id='w'+i+'_'+j;
     s.textContent=w[2]+' ';d.appendChild(s)});box.appendChild(d)});
 function tick(){const t=a.currentTime;let cur=-1;
   L.forEach((l,i)=>{if(t>=l.start-0.15)cur=i;
     l.words.forEach((w,j)=>document.getElementById('w'+i+'_'+j).classList.toggle('sung',t>=w[0]))});
   L.forEach((l,i)=>document.getElementById('l'+i).classList.toggle('on',i===cur));
   if(cur>=0){const e=document.getElementById('l'+cur);box.scrollTop=e.offsetTop-box.offsetTop-90}
   requestAnimationFrame(tick)}
 tick();
</script>
"""


def tab_lyrics() -> None:
    project = st.session_state.project
    if project is None:
        st.info("Upload a song first (Import tab).")
        return
    if not lyrics.available():
        st.warning("Lyrics are not available on this server.")
        return
    state = st.session_state
    has_vocals = TRACK_VOCALS in project.stems
    st.write("Listens to the singing and writes the words with the time of every word, like "
             "karaoke. Check the result and fix any wrong words below — singing is harder to "
             "understand than speech.")
    if not has_vocals:
        st.info("For much better results, run **Remove Vocals** first. The lyrics are then made "
                "from the voice alone.")
    c1, c2 = st.columns([2, 3])
    language = c1.selectbox("Language of the song", list(LANGUAGES), key="lyr_lang")
    if st.button("📝 Make Lyrics Again" if state.lyrics else "📝 Make Lyrics", type="primary"):
        if not lyrics.model_is_cached():
            st.info("Downloading the lyrics model on first use (about 500 MB). This happens once.")
        bar = st.progress(0.0, text="Listening to the words...")
        try:
            state.lyrics = lyrics.transcribe(
                Path(project.stems[TRACK_VOCALS] if has_vocals else project.working_audio),
                LANGUAGES[language], progress=lambda p, text: bar.progress(min(p, 1.0), text=text))
            sm.save_project(state)
        except lyrics.LyricsError as e:
            st.error(str(e))
        bar.empty()
    if not state.lyrics:
        return

    st.markdown("**Fix the words and timings** (add or delete rows as needed, then save)")
    rows = [{"Start": editor.format_time(l["start"]), "End": editor.format_time(l["end"]),
             "Words": l["text"]} for l in state.lyrics]
    edited = st.data_editor(rows, num_rows="dynamic", width="stretch", hide_index=True,
                            key=f"lyr_table_{len(state.lyrics)}_{hash(json.dumps(state.lyrics)) % 10**8}")
    if st.button("💾 Save lyric changes"):
        try:
            old = {l["text"]: l for l in state.lyrics}
            new = []
            for r in edited:
                if not str(r.get("Words") or "").strip():
                    continue
                start, end = editor.parse_time(r.get("Start")), editor.parse_time(r.get("End"))
                new.append(lyrics.retime(start, end, r["Words"], old.get(" ".join(r["Words"].split()))))
            state.lyrics = sorted(new, key=lambda l: l["start"])
            sm.save_project(state)
            st.rerun()
        except ValidationError as e:
            st.warning(str(e))

    # Timings follow the edited song: trims, reordering and speed are applied.
    timed = lyrics.map_to_timeline(state.lyrics, state.clips, sm.effective_speed(state))
    base = piece_base(project)
    st.markdown("**Download the lyrics** (timings match your edited song)")
    d = st.columns(4)
    d[0].download_button("⬇ Karaoke LRC (word by word)", lyrics.to_lrc(timed, True),
                         f"{base}_karaoke.lrc", "text/plain", key="lyr_lrc_w", width="stretch")
    d[1].download_button("⬇ LRC (line by line)", lyrics.to_lrc(timed, False), f"{base}.lrc",
                         "text/plain", key="lyr_lrc", width="stretch")
    d[2].download_button("⬇ SRT subtitles", lyrics.to_srt(timed), f"{base}.srt", "text/plain",
                         key="lyr_srt", width="stretch")
    d[3].download_button("⬇ ASS karaoke subtitles", lyrics.to_ass(timed, base), f"{base}.ass",
                         "text/plain", key="lyr_ass", width="stretch")

    prev = state.preview
    if st.toggle("🎤 Show karaoke player", key="lyr_player"):
        if not (prev and Path(prev["path"]).exists()):
            st.info("Click **Render Preview** in the Preview tab first; the karaoke player uses "
                    "that sound (choose the Instrumental track to sing along).")
        else:
            if prev["fingerprint"] != current_fingerprint():
                st.warning("You changed something since the preview was rendered. Render it again "
                           "so the words line up.")
            html = (KARAOKE_HTML
                    .replace("__AUDIO__", base64.b64encode(read_bytes(prev["path"])).decode())
                    .replace("__LINES__", json.dumps(timed)))
            components.html(html, height=330)


MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac", "mp4": "video/mp4"}


def export_names(project) -> tuple[Path, str]:
    job_dir = file_utils.job_path(project.job_id)
    base = Path(file_utils.sanitize_filename(project.source_name, "song")).stem
    return job_dir, f"{base}_{TRACK_SUFFIX[st.session_state.current_track]}"


def render_audio_export(project, out: Path, fmt: str) -> Path:
    return renderer.render_layers(st.session_state.clips, sm.render_layers(st.session_state), out, fmt,
                                  overlays=st.session_state.overlays)


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
    choices = {"FLAC (lossless, about half the size of WAV — best for sending)": "flac",
               "MP3 (320 kbps, smallest)": "mp3", "WAV (CD quality, uncompressed)": "wav",
               "MP4 (video)": "mp4"}
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

    if fmt == "mp4" and not project.info.has_video:
        st.caption("This project has no picture, so the MP4 shows a black screen with the sound.")
    path = st.session_state.exports.get(fmt)
    if path and Path(path).exists():
        st.success(f"Your file is ready ({Path(path).stat().st_size / compressor.MB:.1f} MB).")
        download(f"⬇ Download {Path(path).name}", path, Path(path).name, MIME[fmt],
                 key=f"dl_export_{fmt}", type="primary")

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
        download(f"⬇ Download {path.name}", path, path.name,
                 MIME.get(path.suffix.lstrip("."), "application/octet-stream"),
                 key="share_dl", type="primary")


def video_is_unchanged(project) -> bool:
    """True when the export would just reproduce the original video file."""
    return (st.session_state.current_track == TRACK_ORIGINAL
            and editor.is_untouched(st.session_state.clips, project.info.duration)
            and st.session_state.audio_settings.is_neutral
            and not st.session_state.overlays
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
                    "▶ Preview", "💾 Export", "📝 Lyrics"])
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
    with tabs[6]:
        tab_lyrics()


main()
