"""FFmpeg detection, probing and command execution (no shell=True anywhere)."""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from models import MediaInfo

log = logging.getLogger("mas.ffmpeg")


class FFmpegNotFound(Exception):
    pass


class FFmpegError(Exception):
    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


class ProbeError(Exception):
    pass


def find_ffmpeg() -> str | None:
    """Locate ffmpeg. FFMPEG_PATH env var wins, then PATH, then common folders."""
    env = os.environ.get("FFMPEG_PATH")
    if env and Path(env).is_file():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if Path(candidate).is_file():
            return candidate
    return None


def find_ffprobe() -> str | None:
    env = os.environ.get("FFPROBE_PATH")
    if env and Path(env).is_file():
        return env
    ff = find_ffmpeg()
    if ff:
        sibling = Path(ff).with_name("ffprobe" + Path(ff).suffix)
        if sibling.is_file():
            return str(sibling)
    return shutil.which("ffprobe")


def ffmpeg_bin() -> str:
    path = find_ffmpeg()
    if not path:
        raise FFmpegNotFound("ffmpeg not found")
    return path


def ffprobe_bin() -> str:
    path = find_ffprobe()
    if not path:
        raise FFmpegNotFound("ffprobe not found")
    return path


@functools.lru_cache(maxsize=None)
def available_filters() -> frozenset:
    try:
        out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-filters"],
                             capture_output=True, text=True, timeout=30).stdout
    except (FFmpegNotFound, OSError, subprocess.SubprocessError):
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return frozenset(names)


def has_filter(name: str) -> bool:
    return name in available_filters()


@functools.lru_cache(maxsize=None)
def has_encoder(name: str) -> bool:
    try:
        out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30).stdout
    except (FFmpegNotFound, OSError, subprocess.SubprocessError):
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def ffmpeg_version() -> str:
    try:
        out = subprocess.run([ffmpeg_bin(), "-version"], capture_output=True, text=True, timeout=30)
        return out.stdout.splitlines()[0] if out.stdout else "unknown"
    except Exception:  # noqa: BLE001 - informational only
        return "unavailable"


def run_ffmpeg(args: Sequence[str], timeout: float | None = 3600) -> None:
    """Run ffmpeg with an argument list. Raises FFmpegError with stderr on failure."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *map(str, args)]
    log.info("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise FFmpegError("FFmpeg timed out", str(e)) from e
    if proc.returncode != 0:
        log.error("FFmpeg failed (%s): %s", proc.returncode, proc.stderr[-4000:])
        raise FFmpegError(f"FFmpeg exited with code {proc.returncode}", proc.stderr)


def _parse_fps(rate: str) -> float:
    try:
        num, _, den = rate.partition("/")
        return float(num) / float(den or 1) if float(den or 1) else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str | Path) -> MediaInfo:
    cmd = [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
           str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise ProbeError(str(e)) from e
    if proc.returncode != 0:
        log.error("ffprobe failed for %s: %s", path, proc.stderr[-2000:])
        raise ProbeError(proc.stderr.strip() or "ffprobe failed")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    # Cover art in MP3/M4A shows up as a video stream; ignore attached pictures.
    video = [s for s in streams if s.get("codec_type") == "video"
             and not s.get("disposition", {}).get("attached_pic")]
    duration = 0.0
    for candidate in [fmt.get("duration")] + [s.get("duration") for s in audio + video]:
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue
    info = MediaInfo(path=str(path), duration=duration, has_audio=bool(audio),
                     has_video=bool(video), format_name=fmt.get("format_name", ""))
    if video:
        v = video[0]
        info.width = int(v.get("width") or 0)
        info.height = int(v.get("height") or 0)
        info.fps = _parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1")
    return info
