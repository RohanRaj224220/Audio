"""Render a non-destructive timeline (clips + sound settings) to an audio file."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import audio_processor as ap
import ffmpeg_utils
from models import AudioSettings, Clip, Overlay, ValidationError

log = logging.getLogger("mas.render")

FORMATS = {
    "wav": ["-c:a", "pcm_s16le"],
    "flac": ["-c:a", "flac", "-compression_level", "8", "-sample_fmt", "s16"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
    "preview": ["-c:a", "libmp3lame", "-b:a", "192k"],
    "aac": ["-c:a", "aac", "-b:a", "256k"],
    "float_wav": ["-c:a", "pcm_f32le"],
}


class RenderError(Exception):
    pass


def _timeline_command(clips: list[Clip], track_path: str, extra_filters: list[str],
                      use_rubberband: bool) -> tuple[list[str], str]:
    """Inputs + filter_complex that produce the label [out]."""
    if not clips:
        raise ValidationError("The timeline is empty.")
    args: list[str] = []
    parts: list[str] = []
    norm = f"aformat=sample_fmts=fltp:sample_rates={ap.SAMPLE_RATE}:channel_layouts=stereo"
    for i, clip in enumerate(clips):
        if clip.duration <= 0:
            raise ValidationError("The selected time range is invalid.")
        if clip.is_silence:
            args += ["-f", "lavfi", "-t", f"{clip.duration:.3f}",
                     "-i", f"anullsrc=r={ap.SAMPLE_RATE}:cl=stereo"]
        else:
            # Every audio clip reads from the chosen track (original / instrumental / vocals).
            # Stems are time-aligned with the original, so clip times stay valid.
            args += ["-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}", "-i", track_path]
        chain = [norm] + ap.clip_filters(clip, use_rubberband)
        parts.append(f"[{i}:a]{','.join(chain)}[c{i}]")
    labels = "".join(f"[c{i}]" for i in range(len(clips)))
    parts.append(f"{labels}concat=n={len(clips)}:v=0:a=1[cat]")
    tail = ",".join(extra_filters) if extra_filters else "anull"
    parts.append(f"[cat]{tail}[out]")
    return args, ";".join(parts)


def _settings_list(settings) -> list[AudioSettings]:
    items = list(settings) if isinstance(settings, (list, tuple)) else [settings]
    for s in items:
        s.validate()
    return items


def render_timeline(clips: list[Clip], track_path: str, settings, out_path: Path,
                    fmt: str = "wav") -> Path:
    """Render clips from one track to `out_path`. fmt: wav | mp3 | preview | aac.

    `settings` is one AudioSettings or a list applied in order
    (e.g. [vocal settings, whole-song settings]).
    """
    return render_layers(clips, [(track_path, settings)], out_path, fmt)


def render_layers(clips: list[Clip], layers: list[tuple[str, object]], out_path: Path,
                  fmt: str = "wav", overlays: list[Overlay] | None = None) -> Path:
    """Render the same clip timeline from several time-aligned tracks and mix them.

    `layers` is [(track_path, settings_or_list), ...]. Each layer gets its own
    speed/pitch/tone/volume; the layers are then summed (no auto-normalising)
    and peak-limited. `overlays` are extra pieces of sound mixed on top at their
    own position in the finished song (they are not affected by the settings).
    """
    overlays = list(overlays or [])
    for ov in overlays:
        if ov.duration <= 0 or ov.position < 0 or not Path(ov.source_path).exists():
            raise ValidationError(f"The added layer “{ov.name}” is not valid.")
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt}")
    if not layers:
        raise ValidationError("Nothing to render.")
    layers = [(path, _settings_list(settings)) for path, settings in layers]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_rb = ap.rubberband_available()
    plans = [(path, *ap.chain_filters(sl, use_rb)) for path, sl in layers]
    log.info("Render %d clips x %d layer(s) -> %s (fmt=%s, rubberband=%s, librosa=%s)",
             len(clips), len(layers), out_path, fmt, use_rb, [p[2] for p in plans])
    encode = ["-ar", str(ap.SAMPLE_RATE), "-ac", "2", *FORMATS[fmt]]
    try:
        if len(plans) == 1 and not plans[0][2] and not overlays:
            path, filters, _ = plans[0]
            inputs, graph = _timeline_command(clips, path, filters + [ap.limiter_filter()], use_rb)
            ffmpeg_utils.run_ffmpeg([*inputs, "-filter_complex", graph, "-map", "[out]",
                                     *encode, str(out_path)])
            return out_path
        with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp:
            stems = []
            for n, (path, filters, semitones) in enumerate(plans):
                stage = Path(tmp) / f"layer{n}.wav"
                inputs, graph = _timeline_command(clips, path, filters, use_rb)
                ffmpeg_utils.run_ffmpeg([*inputs, "-filter_complex", graph, "-map", "[out]",
                                         "-ar", str(ap.SAMPLE_RATE), "-ac", "2",
                                         *FORMATS["float_wav"], str(stage)])
                if semitones:
                    shifted = Path(tmp) / f"layer{n}_pitch.wav"
                    ap.pitch_shift_file(stage, shifted, semitones)
                    stage = shifted
                stems.append(stage)
            args: list[str] = []
            for stage in stems:
                args += ["-i", str(stage)]
            labels = "".join(f"[{i}:a]" for i in range(len(stems)))
            chains = []
            for k, ov in enumerate(overlays):
                args += ["-ss", f"{ov.start:.3f}", "-t", f"{ov.duration:.3f}", "-i", ov.source_path]
                chain = [f"aformat=sample_fmts=fltp:sample_rates={ap.SAMPLE_RATE}:channel_layouts=stereo",
                         *ap.volume_filters(ov.volume)]
                if ov.position > 0:
                    chain.append(f"adelay={int(round(ov.position * 1000))}:all=1")
                chains.append(f"[{len(stems) + k}:a]{','.join(chain)}[ov{k}]")
                labels += f"[ov{k}]"
            total = len(stems) + len(overlays)
            if total == 1:
                graph = f"[0:a]{ap.limiter_filter()}[out]"
            else:
                graph = ";".join(chains + [f"{labels}amix=inputs={total}:normalize=0:"
                                           f"duration=longest,{ap.limiter_filter()}[out]"])
            ffmpeg_utils.run_ffmpeg([*args, "-filter_complex", graph, "-map", "[out]", *encode,
                                     str(out_path)])
        return out_path
    except (ffmpeg_utils.FFmpegError, OSError, RuntimeError) as e:
        log.exception("Rendering failed")
        raise RenderError(str(e)) from e


def convert_audio(src: Path, dest: Path, fmt: str) -> Path:
    """Plain format conversion (used for stem downloads)."""
    ffmpeg_utils.run_ffmpeg(["-i", str(src), "-vn", "-ar", str(ap.SAMPLE_RATE), "-ac", "2",
                             *FORMATS[fmt], str(dest)])
    return dest
