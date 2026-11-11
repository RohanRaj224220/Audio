"""AI vocal separation.

Two engines:
  * "best" (default when installed): RoFormer models from the `audio-separator`
    package, run as a 4-pass pipeline in `separation_worker.py` (subprocess).
    Much less bleed than Demucs - see README for measured numbers.
  * "fast" / fallback: Demucs htdemucs in-process via its Python API (audio I/O
    through soundfile, so no torchaudio backend is needed).
"""

from __future__ import annotations

import functools
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from file_utils import WORKSPACE

log = logging.getLogger("mas.separator")

MODEL_NAME = os.environ.get("MAS_DEMUCS_MODEL", "htdemucs")
MODEL_MARKER = WORKSPACE / f".model_ready_{MODEL_NAME}"
MODEL_DIR = Path(os.environ.get("MAS_MODEL_DIR", WORKSPACE / "models"))
WORKER = Path(__file__).resolve().parent / "separation_worker.py"
# "best" = RoFormer pipeline, "fast" = Demucs. Hosted servers with little RAM can set fast.
DEFAULT_MODE = os.environ.get("MAS_SEPARATION_MODE", "best")
BEST_MODELS = ("vocals_mel_band_roformer.ckpt",)
BEST_CONFIG = "config_vocals_mel_band_roformer_kim.yaml"
# The same checkpoint, from its author's Hugging Face repo (more reliable than GitHub releases).
BEST_HF_REPO, BEST_HF_FILE = "KimberleyJSN/melbandroformer", "MelBandRoformer.ckpt"
BUNDLED_CONFIGS = Path(__file__).resolve().parent / "model_configs"
# The best engine peaks at ~1.8 GB RAM; below this much total RAM use the lighter Demucs model.
MIN_RAM_GB_FOR_BEST = float(os.environ.get("MAS_MIN_RAM_GB_FOR_BEST", "3.5"))


class SeparationError(Exception):
    pass


class SeparationUnavailable(SeparationError):
    pass


class SeparationMemoryError(SeparationError):
    pass


def separation_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "demucs"))


def best_available() -> bool:
    """True when the high-quality RoFormer engine (audio-separator package) is installed."""
    try:
        importlib.metadata.version("audio-separator")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def total_ram_gb() -> float:
    """Physical RAM, or the container memory limit if lower (cgroups on Linux hosts)."""
    ram = 0.0
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    for limit_file in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(limit_file).read_text().strip()
            if text.isdigit():
                ram = min(ram or float("inf"), int(text) / 1e9)
        except OSError:
            continue
    return ram


def active_mode() -> str:
    if DEFAULT_MODE != "best" or not best_available():
        return "fast"
    ram = total_ram_gb()
    if ram and ram < MIN_RAM_GB_FOR_BEST:
        return "fast"
    return "best"


def ensure_best_models() -> None:
    """Make sure the best-quality model files are on disk (downloads ~900 MB once)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config = MODEL_DIR / BEST_CONFIG
    if not config.exists() and (BUNDLED_CONFIGS / BEST_CONFIG).exists():
        shutil.copy(BUNDLED_CONFIGS / BEST_CONFIG, config)
    ckpt = MODEL_DIR / BEST_MODELS[0]
    if not ckpt.exists():
        from huggingface_hub import hf_hub_download

        log.info("Downloading %s/%s", BEST_HF_REPO, BEST_HF_FILE)
        cached = hf_hub_download(BEST_HF_REPO, BEST_HF_FILE)
        tmp = ckpt.with_suffix(".part")
        shutil.copy(cached, tmp)
        tmp.replace(ckpt)


def model_is_cached() -> bool:
    if active_mode() == "best":
        return all((MODEL_DIR / m).exists() for m in BEST_MODELS)
    return MODEL_MARKER.exists()


def compute_device() -> str:
    """CUDA if present, else CPU. (Apple MPS is opt-in: MAS_USE_MPS=1.)"""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("MAS_USE_MPS") == "1" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@functools.lru_cache(maxsize=1)
def load_model(name: str = MODEL_NAME):
    """Load (and on first use, download) a Demucs model. Keeps the last one in memory."""
    if not separation_available():
        raise SeparationUnavailable("torch/demucs not installed")
    from demucs.pretrained import get_model

    log.info("Loading separation model %s", name)
    model = get_model(name)
    model.eval()
    if name == MODEL_NAME:
        MODEL_MARKER.parent.mkdir(parents=True, exist_ok=True)
        MODEL_MARKER.touch()
    return model


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in text or "can't allocate" in text \
        or "failed to allocate" in text


def separate(wav_path: Path, out_dir: Path,
             progress: Optional[Callable[[float, str], None]] = None,
             mode: Optional[str] = None, clean: bool = True) -> dict[str, Path]:
    """Split `wav_path` into instrumental.wav + vocals.wav inside `out_dir`.

    progress(value 0..1, stage text) is called as work advances.
    """
    mode = mode or active_mode()
    if mode == "best" and best_available():
        return _separate_worker(Path(wav_path), Path(out_dir), progress, "best", clean)
    return _separate_demucs(Path(wav_path), Path(out_dir),
                            (lambda v: progress(v, "Separating vocals...")) if progress else None)


def _separate_worker(wav_path: Path, out_dir: Path,
                     progress: Optional[Callable[[float, str], None]], mode: str,
                     clean: bool = True) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not model_is_cached() and progress:
            progress(0.01, "Downloading the AI model (about 900 MB, only the first time)...")
        ensure_best_models()
    except Exception as e:  # noqa: BLE001
        log.exception("Model download failed")
        raise SeparationError(f"model download failed: {e}") from e
    cmd = [sys.executable, str(WORKER), "--input", str(wav_path), "--outdir", str(out_dir),
           "--mode", mode, "--model-dir", str(MODEL_DIR)] + (["--clean"] if clean else [])
    log.info("Separation worker: %s", " ".join(cmd))
    err_log = out_dir / "worker.log"
    result: dict = {}
    with open(err_log, "w") as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True,
                                cwd=str(MODEL_DIR))
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("event") == "progress" and progress:
                progress(float(msg.get("value", 0)), msg.get("stage", ""))
            elif msg.get("event") in ("done", "error"):
                result = msg
        code = proc.wait()
    shutil.rmtree(out_dir / "_work", ignore_errors=True)
    if result.get("event") == "done" and code == 0:
        stems = {k: Path(result[k]) for k in ("vocals", "instrumental")}
        log.info("Separation finished: %s", stems)
        return stems
    tail = err_log.read_text(errors="replace")[-3000:] if err_log.exists() else ""
    log.error("Separation worker failed (code %s): %s\n%s", code, result, tail)
    # A killed process (-9) is almost always the OS running out of memory.
    if result.get("kind") == "memory" or code in (-9, 137):
        raise SeparationMemoryError(result.get("message", "killed"))
    raise SeparationError(result.get("message", f"worker exited with {code}"))


# 6-stem Demucs: drums, bass, guitar, piano, other (+ a vocals output we fold into "other").
INSTRUMENT_MODEL = os.environ.get("MAS_INSTRUMENT_MODEL", "htdemucs_6s")
# A piece that never gets louder than this (RMS over any 1 s) is treated as "not in this song".
SILENT_PIECE_RMS = 0.006


def split_instruments(instrumental_wav: Path, out_dir: Path,
                      progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Path]:
    """Split the instrumental into drums / bass / guitar / piano / other WAV files.

    Only instruments that are actually present get a file; anything too quiet to be a
    real part is folded into "other", so the pieces always add back up to the instrumental.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sources, sr = _demucs_sources(
            Path(instrumental_wav), INSTRUMENT_MODEL,
            (lambda v: progress(0.02 + 0.93 * v, f"Separating instruments... {int(v * 100)}%"))
            if progress else None)
        # "other" is everything the named pieces don't cover (including the model's leftover
        # "vocals" output), so the pieces always add back up to exactly the instrumental.
        sources.pop("other", None)
        sources.pop("vocals", None)
        other, _ = sf.read(str(instrumental_wav), dtype="float32", always_2d=True)
        if other.shape[1] == 1:
            other = np.repeat(other, 2, axis=1)
        other = other[:, :2].copy()
        result: dict[str, Path] = {}
        win = sr
        for name, data in list(sources.items()):
            usable = data[: len(data) // win * win].reshape(-1, win, data.shape[1])
            loudest = float(np.sqrt((usable ** 2).mean(axis=(1, 2))).max()) if len(usable) else 0.0
            if loudest < SILENT_PIECE_RMS:
                log.info("Instrument %s not present (loudest second %.4f); left in other",
                         name, loudest)
                continue
            result[name] = _write_stem(out_dir / f"{name}.wav", data, sr)
            other -= sf.read(str(result[name]), dtype="float32", always_2d=True)[0]
        result["other"] = _write_stem(out_dir / "other.wav", other, sr)
        if progress:
            progress(1.0, "Done")
        log.info("Instrument split finished: %s", result)
        return result
    except SeparationError:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("Instrument split failed")
        if _is_oom(e):
            raise SeparationMemoryError(str(e)) from e
        raise SeparationError(str(e)) from e


def _write_stem(path: Path, data: np.ndarray, sr: int) -> Path:
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")
    return path


def _demucs_sources(wav_path: Path, model_name: str,
                    progress: Optional[Callable[[float], None]] = None
                    ) -> tuple[dict[str, np.ndarray], int]:
    """Run a Demucs model; returns ({source name: float32 array (samples, 2)}, sample rate)."""
    if not separation_available():
        raise SeparationUnavailable("torch/demucs not installed")
    import torch
    from demucs.apply import apply_model

    model = load_model(model_name)
    device = compute_device()
    log.info("Demucs %s on %s (%s, torch %s)", model_name, wav_path, device, torch.__version__)
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if sr != model.samplerate:
        raise SeparationError(f"Expected {model.samplerate} Hz input, got {sr}")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    mix = torch.from_numpy(np.ascontiguousarray(data[:, :2].T))
    length = mix.shape[-1]
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    mix = (mix - mean) / std

    def _cb(d: dict) -> None:
        if progress and d.get("state") == "end":
            progress(min(1.0, (d.get("segment_offset", 0) + 1) / max(1, length)))

    with torch.no_grad():
        sources = apply_model(model, mix[None], device=device, shifts=1, split=True,
                              overlap=0.25, progress=False, num_workers=0, callback=_cb)[0]
    sources = sources * std  # the (tiny) mean offset stays in the remainder, not in every piece
    return {n: sources[i].cpu().numpy().T for i, n in enumerate(model.sources)}, sr


def _separate_demucs(wav_path: Path, out_dir: Path,
                     progress: Optional[Callable[[float], None]] = None) -> dict[str, Path]:
    """Demucs htdemucs, in-process (fast mode / fallback)."""
    if not separation_available():
        raise SeparationUnavailable("torch/demucs not installed")
    import torch
    from demucs.apply import apply_model

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = load_model()
        device = compute_device()
        log.info("Separating %s on %s (torch %s)", wav_path, device, torch.__version__)
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        if sr != model.samplerate:
            raise SeparationError(f"Expected {model.samplerate} Hz input, got {sr}")
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        mix = torch.from_numpy(np.ascontiguousarray(data[:, :2].T))
        length = mix.shape[-1]
        ref = mix.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        mix = (mix - mean) / std

        def _cb(d: dict) -> None:
            if progress and d.get("state") == "end":
                progress(min(1.0, (d.get("segment_offset", 0) + 1) / max(1, length)))

        with torch.no_grad():
            sources = apply_model(model, mix[None], device=device, shifts=1, split=True,
                                  overlap=0.25, progress=False, num_workers=0, callback=_cb)[0]
        sources = sources * std + mean
        names = list(model.sources)
        vocals = sources[names.index("vocals")]
        instrumental = sum(sources[i] for i, n in enumerate(names) if n != "vocals")
        result = {}
        for name, tensor in (("instrumental", instrumental), ("vocals", vocals)):
            arr = np.clip(tensor.cpu().numpy().T, -1.0, 1.0)
            path = out_dir / f"{name}.wav"
            sf.write(str(path), arr, sr, subtype="PCM_16")
            result[name] = path
        log.info("Separation finished: %s", result)
        return result
    except SeparationError:
        raise
    except Exception as e:  # noqa: BLE001 - convert to friendly categories
        log.exception("Separation failed")
        if _is_oom(e):
            raise SeparationMemoryError(str(e)) from e
        raise SeparationError(str(e)) from e
