"""Streamlit app smoke tests using Streamlit's headless AppTest runner."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import editor
import file_utils
import state_manager as sm
import video_processor as vp

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def test_app_starts_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs][:6] == ["📂 Import", "🎤 Vocal Remover", "✂️ Editor",
                                             "🎚 Sound Controls", "▶ Preview", "💾 Export"]
    assert at.session_state.project is None


def test_preset_updates_sliders():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="preset_Slow Practice").click().run()
    assert not at.exception
    assert at.slider(key=sm.W_SPEED).value == 0.75
    at.button(key="preset_Original").click().run()
    assert at.slider(key=sm.W_SPEED).value == 1.0


def test_editor_preview_export_with_loaded_project(tone_wav):
    job_id, job_dir = file_utils.new_job()
    with open(tone_wav, "rb") as fh:
        project = vp.import_media(job_id, job_dir, "tone.wav", fh)
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state.project = project
    at.session_state.job_id = job_id
    at.session_state.clips = editor.initial_clips(project.working_audio, project.info.duration)
    at.run()
    assert not at.exception

    cid = at.session_state.clips[0].id
    at.button(key=f"dup_{cid}").click().run()
    assert not at.exception and len(at.session_state.clips) == 2
    at.button(key=f"dup_{cid}").click().run()
    assert len(at.session_state.clips) == 3
    undo = [b for b in at.button if b.label == "↶ Undo"][0]
    undo.click().run()
    assert len(at.session_state.clips) == 2

    at.slider(key=sm.W_SPEED).set_value(1.5).run()
    render = [b for b in at.button if "Render Preview" in b.label][0]
    render.click().run()
    assert not at.exception
    assert Path(at.session_state.preview["path"]).exists()

    export = [b for b in at.button if "Export Audio" in b.label][0]
    export.click().run()
    assert not at.exception
    assert Path(at.session_state.exports["mp3"]).exists()
