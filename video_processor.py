"""Video helpers: extract audio, replace audio, export an edited video."""

from __future__ import annotations

import logging
from pathlib import Path

import audio_processor as ap
import editor
import ffmpeg_utils
from models import Clip, MediaInfo

log = logging.getLogger("mas.video")


class VideoExportError(Exception):
    pass


def extract_audio(src: Path, dest: Path) -> Path:
    """Decode the first audio stream to 44.1 kHz stereo 16-bit WAV."""
    ffmpeg_utils.run_ffmpeg(["-i", str(src), "-map", "0:a:0", "-vn", "-ac", "2",
                             "-ar", str(ap.SAMPLE_RATE), "-c:a", "pcm_s16le", str(dest)])
    return dest


def _video_codec_args() -> list[str]:
    if ffmpeg_utils.has_encoder("libx264"):
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p"]
    return ["-c:v", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p"]


def replace_audio(video: Path, audio: Path, out: Path) -> Path:
    """Keep the original picture untouched and swap in new audio."""
    base = ["-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
            "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart"]
    try:
        ffmpeg_utils.run_ffmpeg([*base, "-c:v", "copy", str(out)])
    except ffmpeg_utils.FFmpegError:
        log.warning("Video stream copy failed, re-encoding instead")
        ffmpeg_utils.run_ffmpeg([*base, *_video_codec_args(), str(out)])
    return out


def rerender_video(info: MediaInfo, clips: list[Clip], speed: float, audio: Path, out: Path) -> Path:
    """Cut/reorder the picture to match the clip list, then add the new audio."""
    w = max(2, info.width - info.width % 2) if info.width else 1280
    h = max(2, info.height - info.height % 2) if info.height else 720
    fps = info.fps if 1 <= info.fps <= 120 else 30.0
    args: list[str] = []
    parts: list[str] = []
    norm = (f"fps={fps:.4f},scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p")
    for i, clip in enumerate(clips):
        if clip.is_silence:
            args += ["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps:.4f}:d={clip.duration:.3f}"]
            chain = norm
        else:
            args += ["-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}", "-i", info.path]
            chain = norm
            if abs(clip.speed - 1.0) > 1e-6:
                chain += f",setpts=PTS/{clip.speed:.6g}"
        parts.append(f"[{i}:v]{chain},setpts=PTS-STARTPTS[v{i}]")
    labels = "".join(f"[v{i}]" for i in range(len(clips)))
    tail = f",setpts=PTS/{speed:.6g}" if abs(speed - 1.0) > 1e-6 else ""
    parts.append(f"{labels}concat=n={len(clips)}:v=1:a=0{tail}[vout]")
    audio_idx = len(clips)
    args += ["-i", str(audio)]
    ffmpeg_utils.run_ffmpeg([*args, "-filter_complex", ";".join(parts), "-map", "[vout]",
                             "-map", f"{audio_idx}:a:0", *_video_codec_args(), "-c:a", "aac",
                             "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out)])
    return out


def export_video(info: MediaInfo, clips: list[Clip], speed: float, audio: Path,
                 out: Path) -> Path:
    """Copy the picture when only the sound changed; re-render it after edits."""
    if not info.has_video:
        raise VideoExportError("Source has no video")
    try:
        if editor.is_untouched(clips, info.duration) and abs(speed - 1.0) < 1e-6:
            log.info("Video export: picture unchanged, replacing audio only")
            return replace_audio(Path(info.path), audio, out)
        log.info("Video export: re-rendering picture for %d clips", len(clips))
        return rerender_video(info, clips, speed, audio, out)
    except ffmpeg_utils.FFmpegError as e:
        raise VideoExportError(str(e)) from e


def import_media(job_id: str, job_dir: Path, filename: str, stream, size: int | None = None):
    """Save an upload to <job>/source/, then import it (see `import_file`).

    Raises file_utils.UnsupportedFormatError, file_utils.FileTooLargeError,
    ffmpeg_utils.ProbeError or FFmpegError.
    """
    import file_utils

    ext = file_utils.check_extension(filename)
    safe = f"{file_utils.sanitize_filename(Path(filename).stem, 'upload')}.{ext}"
    src = file_utils.save_stream(stream, job_dir / "source" / safe, size)
    return import_file(job_id, job_dir, src, display_name=filename)


def import_file(job_id: str, job_dir: Path, src: Path, display_name: str | None = None,
                origin_url: str = ""):
    """Probe a file already inside <job>/source/ and extract working audio.

    Returns a `models.Project`.
    """
    import soundfile as sf

    from models import Project

    info = ffmpeg_utils.probe(src)
    if not info.has_audio:
        raise ffmpeg_utils.ProbeError("File has no audio stream")
    wav = extract_audio(src, job_dir / "work" / "original.wav")
    # Use the decoded audio length as the single source of truth for timing.
    info.duration = round(sf.info(str(wav)).duration, 3)
    if info.duration <= 0:
        raise ffmpeg_utils.ProbeError("Decoded audio is empty")
    log.info("Imported %s: %.2fs, video=%s, %sx%s@%.2f", src.name, info.duration, info.has_video,
             info.width, info.height, info.fps)
    return Project(job_id=job_id, source_name=display_name or src.name, source_path=str(src),
                   working_audio=str(wav), info=info, origin_url=origin_url)
