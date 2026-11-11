"""Download from a YouTube or SoundCloud link as MP3 audio or MP4 video (via yt-dlp).

Only single tracks/videos are accepted (no playlists, no live streams).
Users are responsible for only downloading content they have the right to use.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import ffmpeg_utils
import file_utils

log = logging.getLogger("mas.download")

MAX_DOWNLOAD_MINUTES = float(os.environ.get("MAX_DOWNLOAD_MINUTES", "30"))

SITES = {
    "YouTube": ("youtube.com", "youtu.be"),
    "SoundCloud": ("soundcloud.com",),
}


class DownloadError(Exception):
    """Friendly, user-facing message in str(e)."""


@dataclass
class DownloadResult:
    path: Path
    title: str
    site: str
    url: str
    duration: float


def detect_site(url: str) -> str:
    """Return 'YouTube' or 'SoundCloud', or raise DownloadError."""
    url = (url or "").strip()
    if not url:
        raise DownloadError("Please paste a link first.")
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise DownloadError("That doesn't look like a web link.")
    for site, domains in SITES.items():
        if any(host == d or host.endswith("." + d) for d in domains):
            return site
    raise DownloadError("Only YouTube and SoundCloud links are supported for now.")


def normalize_url(url: str) -> str:
    url = url.strip()
    return url if "://" in url else "https://" + url


def js_runtimes() -> dict:
    """YouTube needs a JavaScript runtime. Prefer the pip-installed deno, then node."""
    runtimes = {}
    try:
        import deno  # type: ignore

        runtimes["deno"] = {"path": deno.find_deno_bin()}
    except Exception:  # noqa: BLE001
        if shutil.which("deno"):
            runtimes["deno"] = {}
    node = shutil.which("node")
    if node:
        runtimes["node"] = {"path": node}
    return runtimes


class _Logger:
    def debug(self, msg):
        if not msg.startswith("[debug] "):
            log.debug(msg)

    def info(self, msg):
        log.debug(msg)

    def warning(self, msg):
        log.warning(msg)

    def error(self, msg):
        log.error(msg)


def _friendly(exc: Exception) -> str:
    text = str(exc).lower()
    if "sign in to confirm" in text or "not a bot" in text or "http error 429" in text:
        return ("YouTube blocked this download from this server. "
                "Try again later, or run the app on your own computer.")
    if "private" in text:
        return "This video or track is private."
    if "members-only" in text or "join this channel" in text or "premium" in text:
        return "This video or track requires a subscription."
    if "age" in text and ("confirm" in text or "restricted" in text):
        return "This video is age-restricted and can't be downloaded."
    if "not available in your country" in text or "geo" in text:
        return "This video or track is not available in this country."
    if "unsupported url" in text or "is not a valid url" in text:
        return "That link doesn't point to a single video or track."
    if "unavailable" in text or "404" in text or "does not exist" in text or "removed" in text:
        return "This video or track is unavailable."
    if "max_filesize" in text or "larger than max-filesize" in text:
        return f"This file is larger than {file_utils.MAX_UPLOAD_MB:g} MB."
    if "unable to download" in text or "timed out" in text or "connection" in text:
        return "Couldn't reach the website. Check the internet connection and try again."
    return "The download failed. Please check the link and try again."


KIND_AUDIO = "audio"
KIND_VIDEO = "video"
MAX_VIDEO_HEIGHT = int(os.environ.get("MAX_VIDEO_HEIGHT", "1080"))


def _format_selector(kind: str) -> str:
    if kind == KIND_AUDIO:
        return "bestaudio/best"
    h = MAX_VIDEO_HEIGHT
    # Prefer H.264 video + AAC audio: plays everywhere (older phones, TVs, Windows players).
    return (f"bv*[vcodec^=avc1][height<={h}]+ba[ext=m4a]/"
            f"bv*[height<={h}]+ba/b[height<={h}]/bv*+ba/b")


def download_audio(url: str, dest_dir: Path,
                   progress: Optional[Callable[[float, str], None]] = None) -> DownloadResult:
    """Download the audio of a single YouTube video / SoundCloud track as MP3 into dest_dir."""
    return download_media(url, dest_dir, KIND_AUDIO, progress)


def download_video(url: str, dest_dir: Path,
                   progress: Optional[Callable[[float, str], None]] = None) -> DownloadResult:
    """Download a single YouTube video as MP4 (up to MAX_VIDEO_HEIGHT) into dest_dir."""
    return download_media(url, dest_dir, KIND_VIDEO, progress)


def download_media(url: str, dest_dir: Path, kind: str = KIND_AUDIO,
                   progress: Optional[Callable[[float, str], None]] = None) -> DownloadResult:
    """Download one item as MP3 (kind="audio") or MP4 (kind="video") into dest_dir."""
    if kind not in (KIND_AUDIO, KIND_VIDEO):
        raise ValueError(kind)
    site = detect_site(url)
    if kind == KIND_VIDEO and site == "SoundCloud":
        raise DownloadError("SoundCloud only has audio. Choose “Audio (MP3)” for SoundCloud links.")
    url = normalize_url(url)
    try:
        import yt_dlp
    except ImportError as e:
        raise DownloadError("Downloading from links is not available on this server.") from e

    ffmpeg = ffmpeg_utils.find_ffmpeg()
    if not ffmpeg:
        raise DownloadError("FFmpeg is not available.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = "mp3" if kind == KIND_AUDIO else "mp4"
    # A video download fetches two files (picture, then sound); spread progress across both.
    parts = {"done": 0, "count": 1 if kind == KIND_AUDIO else 2}

    def hook(d: dict) -> None:
        if not progress:
            return
        share = 0.9 / parts["count"]
        base = 0.02 + share * min(parts["done"], parts["count"] - 1)
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            frac = (d.get("downloaded_bytes", 0) / total) if total else 0.0
            progress(min(0.92, base + frac * share), "Downloading...")
        elif d.get("status") == "finished":
            parts["done"] += 1
            if parts["done"] >= parts["count"]:
                progress(0.93, "Converting to MP3..." if kind == KIND_AUDIO else "Joining picture and sound...")

    opts = {
        "format": _format_selector(kind),
        "noplaylist": True,
        "outtmpl": str(dest_dir / "download.%(ext)s"),
        "max_filesize": int(file_utils.MAX_UPLOAD_MB * 1024 * 1024),
        "ffmpeg_location": str(Path(ffmpeg).parent),
        "progress_hooks": [hook],
        "logger": _Logger(),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 3,
        "js_runtimes": js_runtimes(),
        "cachedir": False,
    }
    if kind == KIND_AUDIO:
        opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3",
                                   "preferredquality": "320"}]
    else:
        opts["merge_output_format"] = "mp4"
        # Re-wrap single-file downloads (e.g. webm) as MP4 too.
        opts["postprocessors"] = [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}]
    log.info("Downloading %s %s link: %s (runtimes=%s)", site, kind, url, list(opts["js_runtimes"]))
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("_type") in ("playlist", "multi_video") or info.get("entries"):
                raise DownloadError("That's a playlist. Please paste a link to a single song or video.")
            if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
                raise DownloadError("Live streams can't be downloaded.")
            duration = float(info.get("duration") or 0)
            if duration > MAX_DOWNLOAD_MINUTES * 60:
                raise DownloadError(f"This is longer than {MAX_DOWNLOAD_MINUTES:g} minutes. "
                                    "Please choose a shorter one.")
            if kind == KIND_VIDEO and info.get("requested_formats") is None \
                    and info.get("vcodec") in (None, "none"):
                raise DownloadError("This link has no video picture. Choose “Audio (MP3)” instead.")
            if progress:
                progress(0.02, "Starting download...")
            info = ydl.process_ie_result(info, download=True)
    except DownloadError:
        raise
    except Exception as e:  # noqa: BLE001 - yt-dlp raises many types
        log.exception("Download failed for %s", url)
        raise DownloadError(_friendly(e)) from e

    produced = dest_dir / f"download.{ext}"
    if not produced.exists():
        log.error("Expected %s after download; found %s", produced, list(dest_dir.iterdir()))
        raise DownloadError("The download failed. Please check the link and try again.")
    title = info.get("title") or "download"
    final = dest_dir / file_utils.sanitize_filename(f"{title}.{ext}", f"download.{ext}")
    if final.exists() and final != produced:
        final.unlink()
    produced.rename(final)
    log.info("Downloaded %r -> %s (%.1fs)", title, final, duration)
    if progress:
        progress(1.0, "Done")
    return DownloadResult(path=final, title=title, site=site, url=url, duration=duration)
