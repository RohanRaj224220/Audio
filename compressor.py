"""Compress one file into ONE output file under a size limit (e.g. WhatsApp).

`compress_single` always tries lossless first:
  * audio -> FLAC (bit-exact, verified by decoding and comparing checksums);
  * video -> kept exactly as it is.
If that fits under the limit, it is the result. Otherwise one file is made that
fits the limit at the highest quality possible (MP3 at the highest constant
bitrate that fits, or two-pass H.264 for video). That fallback is lossy, but
the size is guaranteed and checked.

Lossless compression cannot promise a size: it keeps every bit, so how small it
gets depends on the content. Already-compressed files (MP3/MP4) shrink only ~1%
with general-purpose compressors like ZIP, which is why they are not used here.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import ffmpeg_utils

log = logging.getLogger("mas.compress")

MB = 1024 * 1024
# Leave a little room under the limit for container overhead and rounding.
SAFETY = 0.97
MP3_BITRATES = (320, 256, 224, 192, 160, 128, 112, 96, 80, 64, 48, 32)
VIDEO_AUDIO_KBPS = 128
MIN_VIDEO_KBPS = 150


class CompressionError(Exception):
    """User-facing message in str(e)."""


@dataclass
class SharePart:
    path: Path
    size: int
    lossless: bool
    lossless_size: int = 0  # size the lossless version would have had (when it didn't fit)

    @property
    def size_mb(self) -> float:
        return self.size / MB


def _duration(path: Path) -> float:
    d = ffmpeg_utils.probe(path).duration
    if d <= 0:
        raise CompressionError("The file could not be read.")
    return d


def pcm_md5(path: Path) -> str:
    """Checksum of the decoded 16-bit PCM audio (independent of the container/codec)."""
    cmd = [ffmpeg_utils.ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i",
           str(path), "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "md5", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if out.returncode != 0:
        raise CompressionError("Could not verify the file.")
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# Lossless
# ---------------------------------------------------------------------------
def to_flac(src_wav: Path, dest: Path, verify: bool = True) -> Path:
    """Encode 16-bit PCM WAV to FLAC (max compression) and verify it is bit-exact."""
    ffmpeg_utils.run_ffmpeg(["-i", str(src_wav), "-map", "0:a:0", "-c:a", "flac",
                             "-compression_level", "8", "-sample_fmt", "s16", str(dest)])
    if verify and pcm_md5(src_wav) != pcm_md5(dest):
        raise CompressionError("Lossless check failed.")
    log.info("FLAC %s -> %s (%.1f MB -> %.1f MB)", src_wav.name, dest.name,
             src_wav.stat().st_size / MB, dest.stat().st_size / MB)
    return dest


# ---------------------------------------------------------------------------
# Fit to size (lossy, single file)
# ---------------------------------------------------------------------------
def fit_audio(src: Path, dest: Path, limit_mb: float) -> SharePart:
    """MP3 at the highest constant bitrate whose size fits the limit."""
    duration = _duration(src)
    budget = limit_mb * MB * SAFETY
    for kbps in MP3_BITRATES:
        if kbps * 1000 / 8 * duration > budget:
            continue
        ffmpeg_utils.run_ffmpeg(["-i", str(src), "-map", "0:a:0", "-c:a", "libmp3lame",
                                 "-b:a", f"{kbps}k", "-ar", "44100", str(dest)])
        if dest.stat().st_size <= limit_mb * MB:
            log.info("Fit audio %s at %d kbps -> %.1f MB", src.name, kbps, dest.stat().st_size / MB)
            return SharePart(dest, dest.stat().st_size, False)
    raise CompressionError("This is too long to fit in that size, even at low quality.")


def fit_video(src: Path, dest: Path, limit_mb: float) -> SharePart:
    """Two-pass H.264 at the bitrate that fills (but does not exceed) the limit."""
    duration = _duration(src)
    total_kbps = limit_mb * MB * 8 * SAFETY / duration / 1000
    video_kbps = total_kbps - VIDEO_AUDIO_KBPS
    codec = "libx264" if ffmpeg_utils.has_encoder("libx264") else "mpeg4"
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(4):
            if video_kbps < MIN_VIDEO_KBPS:
                raise CompressionError("This video is too long to fit in that size with a "
                                       "watchable picture. Try a larger size limit.")
            passlog = str(Path(tmp) / "pass")
            common = ["-i", str(src), "-map", "0:v:0", "-c:v", codec, "-b:v", f"{int(video_kbps)}k",
                      "-pix_fmt", "yuv420p", "-passlogfile", passlog]
            if codec == "libx264":
                common += ["-preset", "medium"]
            ffmpeg_utils.run_ffmpeg([*common, "-pass", "1", "-an", "-f", "null", "-"])
            ffmpeg_utils.run_ffmpeg([*common, "-pass", "2", "-map", "0:a:0?", "-c:a", "aac",
                                     "-b:a", f"{VIDEO_AUDIO_KBPS}k", "-movflags", "+faststart",
                                     str(dest)])
            if dest.stat().st_size <= limit_mb * MB:
                log.info("Fit video %s at %d kbps -> %.1f MB", src.name, video_kbps,
                         dest.stat().st_size / MB)
                return SharePart(dest, dest.stat().st_size, False)
            video_kbps *= 0.9  # overshot: try again a little lower
    raise CompressionError("Could not fit the video under the size limit.")


# ---------------------------------------------------------------------------
# One file in, one file out
# ---------------------------------------------------------------------------
def compress_single(src: Path, out_dir: Path, limit_mb: float, stem: str,
                    is_video: bool) -> SharePart:
    """Return ONE file under limit_mb: lossless if it fits, otherwise the best quality that fits.

    Audio `src` must be a 16-bit PCM WAV master; video `src` is an MP4.
    """
    if limit_mb < 1:
        raise CompressionError("Please enter a size limit of at least 1 MB.")
    limit = int(limit_mb * MB)
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_video:
        size = src.stat().st_size
        if size <= limit:
            return SharePart(src, size, True)
        part = fit_video(src, out_dir / f"{stem}_compressed.mp4", limit_mb)
        part.lossless_size = size
        return part
    flac = to_flac(src, out_dir / f"{stem}.flac")
    size = flac.stat().st_size
    if size <= limit:
        return SharePart(flac, size, True)
    flac.unlink()
    part = fit_audio(src, out_dir / f"{stem}_compressed.mp3", limit_mb)
    part.lossless_size = size
    return part
