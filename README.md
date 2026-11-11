# 🎵 Media Audio Studio

A simple browser app to remove vocals, trim and rearrange songs, change speed, pitch and tone,
and export the result as MP3, WAV or MP4.

Built with Streamlit (UI) on top of a plain-Python backend that never imports Streamlit,
so the same backend can later be wrapped in a desktop app.

## Run it on your computer

Needs **Python 3.11** and **FFmpeg**.

```bash
# macOS
brew install ffmpeg
# Ubuntu / Debian
sudo apt-get install -y ffmpeg libsndfile1
# Windows: winget install Gyan.FFmpeg   (then reopen the terminal)

cd Audio
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run streamlit_app.py
```

The app opens at http://localhost:8501.
To keep it reachable only from this computer (not from other devices on your Wi-Fi), use
`./run_local.sh`, which adds `--server.address=localhost`.

The first time someone clicks **Remove Vocals**, the AI model (about 900 MB) downloads from Hugging Face.
After that, everything runs offline.

## Project layout

```
streamlit_app.py     UI only (tabs, widgets, session state wiring)
models.py            dataclasses: Project, Clip, AudioSettings, MediaInfo
state_manager.py     session/project state, undo, project.json persistence
editor.py            trim / split / delete / duplicate / reorder / silence / fades (metadata only)
audio_processor.py   speed, pitch, bass/mid/treble, EQ, volume, fades -> FFmpeg filters; librosa pitch fallback
renderer.py          renders the non-destructive timeline to WAV/MP3
video_processor.py   import media, extract audio, replace audio, re-render edited video
compressor.py        lossless FLAC (verified) / fit-to-size MP3 & two-pass H.264, one output file
downloader.py        YouTube / SoundCloud link -> MP3 (yt-dlp)
separation_worker.py separation subprocess (Mel-Band RoFormer, pieces + crossfade, bleed mask)
model_configs/       bundled model config (so only the checkpoint needs downloading)
audio_separator.py   vocal separation (Demucs htdemucs, Python API, CPU by default)
ffmpeg_utils.py      FFmpeg/ffprobe detection, probing, safe subprocess execution
file_utils.py        job folders, filename sanitizing, upload saving, cleanup, logging
waveform.py          waveform images (matplotlib)
tests/               pytest unit, integration and Streamlit app tests
```

Each upload gets its own folder, `workspace/jobs/<uuid>/` (with `source/`, `work/`, `stems/` and `renders/` inside),
plus a `project.json` that holds the clip list and settings. Job folders that haven't been used for 24 hours are
deleted automatically. A job in use by any open session gets touched on every interaction, so it is never deleted.

Logs go to `logs/application.log`, never to the screen.

## Import from YouTube or SoundCloud

On the **Import** tab, choose **Audio (MP3)** or **Video (MP4)**, paste a YouTube or SoundCloud link and click
**Download & Import**. The app downloads it using [yt-dlp](https://github.com/yt-dlp/yt-dlp), then imports it just
like an upload. A **Download MP3** or **Download MP4** button then appears so you can keep the file.

- **MP3:** audio only, converted to a 320 kbps MP3.
- **MP4:** video up to 1080p, chosen with `MAX_VIDEO_HEIGHT`. H.264 video with AAC sound is preferred because it
  plays on any device. Once imported, a video works with the editor, vocal removal and MP4 export.
  SoundCloud has no video, so video requests for SoundCloud links are refused with a message.

- Only single videos or tracks are accepted. Playlists, live streams, and anything longer than `MAX_DOWNLOAD_MINUTES`
  (default 30) are refused with a friendly message.
- YouTube needs a JavaScript runtime. `yt-dlp[default,deno]` in `requirements.txt` installs Deno through pip, so
  no system package is needed. Node, if it's installed, is used as a fallback.
- **On hosted servers YouTube often blocks downloads** from data-centre IP addresses ("Sign in to confirm you're not a
  bot"). The app then shows "YouTube blocked this download from this server". Downloads work reliably when the app
  runs on a home computer. SoundCloud is usually fine anywhere.
- Keep yt-dlp up to date (`pip install -U "yt-dlp[default,deno]"`), because sites change often.
- Only download content you own or have permission to use. Downloading may break a site's terms of service.

## Compress for WhatsApp (one file in, one file out)

At the bottom of the **Export** tab, type a size limit (default **180 MB**), choose Audio or Video, and click
**Compress**. You always get **exactly one file** under the limit:

1. **Lossless first.** Audio becomes FLAC, which is checked bit for bit by decoding it and comparing checksums with
   the original. Video that hasn't been edited is passed through byte-identical. If that fits the limit, this is
   the result.
2. **Otherwise, the best quality that fits.** Audio becomes MP3 at the highest constant bitrate that fits. Video
   goes through a two-pass H.264 encode aimed just under the limit. The size is checked, and if it's still over, the
   encode is retried at a slightly lower bitrate. The app says when this happened and how large the lossless
   version would have been.

Lossless compression can't promise a size, because it keeps every bit. That's why step 2 exists. ZIP-style
compression isn't used, because MP3 and MP4 files are already compressed and would only shrink by about 1%.
Typical sizes: a 4-minute song is about 40 MB as WAV, about 20–30 MB as FLAC, and about 9 MB as 320 kbps MP3.

Tip: in WhatsApp, send the file as a **Document** (📎 → Document). Otherwise WhatsApp re-compresses
videos and audio itself.

FLAC is also available as a normal export format.

## How it works

- **Non-destructive editing:** clips are only start/end times plus settings. Nothing is rendered until you click
  **Render Preview** or **Export**.
- **Vocal separation** uses the Mel-Band RoFormer model (Kim) through the `audio-separator` package, run in
  `separation_worker.py` as a separate process (memory is fully freed afterwards). Long songs are processed in
  24-second pieces with crossfades, which keeps memory around 1.8 GB. An optional **Extra bleed removal** soft
  mask (on by default) cuts more bleed, but can make the sound slightly thinner. On machines with less than 3.5 GB of
  RAM, or with `MAS_SEPARATION_MODE=fast`, the app uses the lighter Demucs htdemucs model instead.

  Measured against the true stems on 60 s of the MUSDB18 test set (bss_eval SIR: how far the bleed sits below the
  wanted sound; higher means less bleed):

  | Engine | Instrument bleed in vocals (SIR) | Vocal bleed in instrumental (SIR) | Time per minute of audio (M-series Mac) |
  |---|---|---|---|
  | Demucs htdemucs (old) | 17.7 dB | 21.7 dB | ~35 s |
  | Mel-Band RoFormer | 24.6 dB | 26.7 dB | ~85 s |
  | Mel-Band RoFormer + extra bleed removal (default) | **28.7 dB** | **30.5 dB** | ~85 s |

  No current separation model removes bleed completely. In very dense mixes, faint residue or slight
  "watery" artifacts can still be heard.
- **Separate controls for vocals and instrumental:** after Remove Vocals, the Sound Controls tab has three sub-tabs:
  *Whole Song*, *Vocals* and *Instrumental*. Each has its own speed, pitch, bass/mid/treble, volume, EQ and presets.
  Choose **Vocals + Instrumental (mix)** in Preview or Export to merge them. Each stem is rendered with its
  own settings, then the whole-song settings are applied on top, and the stems are summed and peak-limited.
  Different speeds for the vocals and the music make them drift apart, and the app warns about this.
- **Vocal separation runs once** per file (or again only when you click *Separate Again*). The stem paths are stored in the project, so moving sliders never re-runs it.
- **Speed without changing pitch:** FFmpeg `atempo`.
- **Pitch without changing speed:** FFmpeg `rubberband` if the installed FFmpeg has it (most Linux builds do).
  Otherwise the app uses `librosa.effects.pitch_shift` with high-quality soxr resampling. The app detects this at startup.
  The Homebrew FFmpeg on macOS does **not** include rubberband, so a Mac uses librosa.
- **Video:** if only the sound changed, the original picture is copied as-is. If you trimmed, split, reordered or changed the
  speed, the picture is cut and re-encoded (H.264) to match. Silence clips become black frames.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_MB` | `200` | Larger uploads are rejected with a friendly message |
| `PORT` | `8501` | Port used by the Docker image |
| `MAX_VIDEO_HEIGHT` | `1080` | Highest video resolution downloaded from YouTube |
| `MAX_DOWNLOAD_MINUTES` | `30` | Longest YouTube/SoundCloud item that can be downloaded |
| `MAS_SEPARATION_MODE` | `best` | `fast` = always use the lighter Demucs model |
| `MAS_MODEL_DIR` | `workspace/models` | Where separation models are stored (~1 GB) |
| `MAS_USE_MPS` | unset | `1` = try Apple GPU for vocal separation (experimental) |
| `FFMPEG_PATH` / `FFPROBE_PATH` | auto | Use a specific FFmpeg build |

`.streamlit/config.toml` sets Streamlit's own upload cap to 300 MB and hides error details from users.

## Tests

```bash
pytest -q                       # everything, incl. real vocal separation and YouTube/SoundCloud downloads
MAS_SKIP_DEMUCS=1 MAS_SKIP_NETWORK=1 pytest -q   # offline: skip model download and link downloads
```

## Deploy

All hosts use the same code. Pick one.

### 1. Streamlit Community Cloud (free, easiest)

1. Push this folder to a GitHub repo. `.gitignore` already excludes `.venv`, `workspace` and `logs`.
2. Go to https://share.streamlit.io, then **Create app**, then choose the repo and branch, with main file `streamlit_app.py`.
3. Under **Advanced settings**, choose **Python 3.11**.
4. Click **Deploy**. `packages.txt` installs FFmpeg. `requirements.txt` installs CPU-only PyTorch.
   The first build takes about 5–10 minutes.

Free-tier limits: about 1 GB of RAM (documented limits change over time) and shared CPU.
Editing, effects, preview and export work fine. **Free tiers have less than 3.5 GB of RAM, so the app automatically uses the lighter, bleedier Demucs model
there. Even that may run out of memory on a full song or take several minutes.** If it fails, the app shows "There is not enough memory to process this file" and keeps working.

### 2. Render (recommended fallback, Docker)

1. Push to GitHub.
2. On https://render.com, click **New**, then **Blueprint**, and pick the repo. `render.yaml` creates a Docker web service.
   (Or choose **New**, then **Web Service**, then Docker, with health check path `/_stcore/health`.)
3. Use at least the **Standard** plan (2 GB). The 512 MB plans are too small for vocal removal.
4. The Docker image downloads the AI model while it builds, so it doesn't download again on every restart.

The local disk on Render is temporary: uploaded jobs disappear on redeploy. That's fine for this app.

### 3. Railway

Create a new project from the GitHub repo. `railway.json` tells Railway to build with the `Dockerfile`.
Railway sets `PORT` automatically.

### 4. Hugging Face Spaces (Docker)

Create a **Docker** Space and push this repo to it. Then add this front-matter block at the very top of this
README in the Space's copy:

```yaml
---
title: Media Audio Studio
sdk: docker
app_port: 8501
---
```

The free CPU Space (16 GB RAM) has enough memory for vocal removal on full songs.

### Docker anywhere

```bash
docker build -t media-audio-studio .
docker run -p 8501:8501 -e PORT=8501 media-audio-studio
```

## Resource notes

Measured on an Apple Silicon Mac, CPU only:

- Best-quality vocal separation took about 85 s per minute of audio (a 4-minute song took 6 min), peaking at
  about 1.8 GB of RAM. The fast Demucs mode takes about 35 s per minute and uses about 1–1.5 GB.
- Hosted CPUs are usually slower, so expect 2 to 4 times longer.
- Rendering and export take a few seconds per song. Pitch changes via librosa add about 5–15 s per song.
  Edited video export re-encodes the picture, which can take about as long as the video itself.
