"""Stand-alone vocal separation worker (run as a subprocess by audio_separator.py).

Runs in its own process so that
  * the third-party `audio_separator` package does not clash with our own
    audio_separator.py module, and
  * the several GB of model memory is fully released after every job.

Protocol: prints one JSON object per line on stdout:
  {"event": "progress", "value": 0.0-1.0, "stage": "..."}
  {"event": "done", "vocals": "<path>", "instrumental": "<path>"}
  {"event": "error", "kind": "memory"|"failed", "message": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Drop the app folder from sys.path *before* importing the separation package.
_APP_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _APP_DIR]

import logging  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

# Chosen from measurements on the MUSDB18 test set (see README): Mel-Band RoFormer (Kim)
# cut bleed by ~6 dB in the vocals and ~4 dB in the instrumental vs htdemucs. A second
# pass through the same model gave no further improvement, so it is not used.
VOCAL_MODEL = "vocals_mel_band_roformer.ckpt"
FAST_MODEL = "htdemucs.yaml"
# Soft-mask exponent for "extra bleed removal" (+~2.3 dB less bleed, slightly thinner sound).
MASK_EXPONENT = 2.0
# Long songs are separated in pieces so memory stays flat (whole-file processing made an
# 8 GB Mac swap heavily: 9 min for 1 min of audio vs ~1.5 s per second in pieces).
SEGMENT_SECONDS = 24.0
OVERLAP_SECONDS = 2.0


def emit(**data) -> None:
    print(json.dumps(data), flush=True)


class Runner:
    def __init__(self, model_dir: Path, work: Path, device_log: logging.Logger):
        from audio_separator.separator import Separator

        self.Separator = Separator
        self.model_dir = model_dir
        self.work = work
        self.log = device_log
        self.loaded: dict[str, object] = {}

    def run(self, model: str, src: Path, tag: str) -> dict[str, Path]:
        out = self.work / tag
        out.mkdir(parents=True, exist_ok=True)
        sep = self.loaded.get(model)
        if sep is None:
            # Free the previous model before loading the next one (keeps peak RAM down).
            self.loaded.clear()
            sep = self.Separator(model_file_dir=str(self.model_dir), output_dir=str(out),
                                 output_format="WAV", log_level=logging.WARNING)
            sep.load_model(model_filename=model)
            self.loaded[model] = sep
        sep.output_dir = str(out)
        if getattr(sep, "model_instance", None) is not None:
            sep.model_instance.output_dir = str(out)
        files = sep.separate(str(src))
        stems: dict[str, Path] = {}
        for f in files:
            p = Path(f) if Path(f).is_absolute() else out / f
            name = p.name.lower()
            for stem in ("vocals", "instrumental", "other", "drums", "bass"):
                if f"({stem})" in name:
                    stems[stem] = p
        if "instrumental" not in stems:
            rest = [p for k, p in stems.items() if k != "vocals"]
            if not rest:
                raise RuntimeError(f"{model} produced no accompaniment: {files}")
            total = sum(sf.read(str(p), dtype="float32", always_2d=True)[0] for p in rest)
            stems["instrumental"] = out / "instrumental_sum.wav"
            sf.write(str(stems["instrumental"]), total, 44100, subtype="FLOAT")
        return stems

    def run_chunked(self, model: str, src: Path, tag: str, on_progress,
                    segment: float = SEGMENT_SECONDS, overlap: float = OVERLAP_SECONDS
                    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Separate in short pieces with crossfades so memory stays flat for long songs.

        Returns (vocals, instrumental, sample_rate) as float32 arrays.
        """
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        n = len(data)
        seg, ov = int(segment * sr), int(overlap * sr)
        step = seg - ov
        starts = list(range(0, max(1, n - ov), step)) or [0]
        out = {k: np.zeros_like(data) for k in ("vocals", "instrumental")}
        weight = np.zeros((n, 1), dtype=np.float32)
        for k, start in enumerate(starts):
            end = min(n, start + seg)
            piece = self.work / tag / f"chunk{k:04d}.wav"
            piece.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(piece), data[start:end], sr, subtype="FLOAT")
            stems = self.run(model, piece, f"{tag}/c{k:04d}")
            # Linear fade-in/out over the overlap so neighbouring pieces blend smoothly.
            length = end - start
            w = np.ones(length, dtype=np.float32)
            fade = min(ov, length // 2)
            if fade and start > 0:
                w[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
            if fade and end < n:
                w[-fade:] = np.minimum(w[-fade:], np.linspace(1, 0, fade, dtype=np.float32))
            for name in out:
                est, _ = sf.read(str(stems[name]), dtype="float32", always_2d=True)
                est = est[:length]
                out[name][start:start + len(est)] += est * w[:len(est), None]
            weight[start:end, 0] += w
            shutil.rmtree(self.work / tag / f"c{k:04d}", ignore_errors=True)
            piece.unlink(missing_ok=True)
            on_progress((k + 1) / len(starts))
        weight[weight == 0] = 1
        return out["vocals"] / weight, out["instrumental"] / weight, sr


def bleed_mask(vocals: np.ndarray, inst: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Re-split the two estimates with a soft time-frequency mask.

    Each STFT bin is shared between vocals and instrumental according to which
    estimate is louder there, pushing leftover bleed into the other stem.
    """
    import librosa

    n = min(len(vocals), len(inst))
    out_v = np.zeros((n, vocals.shape[1]), dtype=np.float32)
    out_i = np.zeros_like(out_v)
    for c in range(vocals.shape[1]):
        V = librosa.stft(vocals[:n, c], n_fft=4096, hop_length=1024)
        I = librosa.stft(inst[:n, c], n_fft=4096, hop_length=1024)
        pv, pi = np.abs(V) ** MASK_EXPONENT, np.abs(I) ** MASK_EXPONENT
        m = pv / (pv + pi + 1e-12)
        del pv, pi
        out_v[:, c] = librosa.istft(V * m, hop_length=1024, length=n)
        out_i[:, c] = librosa.istft(I * (1 - m), hop_length=1024, length=n)
    return out_v, out_i


def separate(src: Path, out_dir: Path, mode: str, model_dir: Path, clean: bool) -> dict[str, Path]:
    work = out_dir / "_work"
    runner = Runner(model_dir, work, logging.getLogger("worker"))
    model = FAST_MODEL if mode == "fast" else VOCAL_MODEL
    emit(event="progress", value=0.03, stage="Loading the AI model...")
    vocals, inst, sr = runner.run_chunked(
        model, src, "p1",
        lambda f: emit(event="progress", value=0.05 + 0.83 * f,
                       stage=f"Separating vocals... {int(f * 100)}%"))
    runner.loaded.clear()
    if clean:
        emit(event="progress", value=0.9, stage="Removing leftover bleed...")
        vocals, inst = bleed_mask(vocals, inst, sr)
    result = {}
    for name, data in (("vocals", vocals), ("instrumental", inst)):
        dest = out_dir / f"{name}.wav"
        sf.write(str(dest), np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")
        result[name] = dest
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--mode", choices=("best", "fast"), default="best")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--clean", action="store_true", help="extra bleed removal")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        args.outdir.mkdir(parents=True, exist_ok=True)
        stems = separate(args.input, args.outdir, args.mode, args.model_dir, args.clean)
    except MemoryError as e:
        emit(event="error", kind="memory", message=str(e))
        return 2
    except Exception as e:  # noqa: BLE001
        text = str(e).lower()
        oom = "out of memory" in text or "can't allocate" in text or "failed to allocate" in text
        emit(event="error", kind="memory" if oom else "failed", message=f"{type(e).__name__}: {e}")
        return 2
    emit(event="progress", value=1.0, stage="Done")
    emit(event="done", **{k: str(v) for k, v in stems.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
