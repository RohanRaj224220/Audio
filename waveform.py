"""Lightweight waveform image for a section of an audio file."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

MAX_POINTS = 2000


def envelope(path: str | Path, start: float = 0.0, end: float | None = None,
             points: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, min, max) arrays for the given range, downsampled."""
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        first = max(0, int(start * sr))
        last = f.frames if end is None else min(f.frames, int(end * sr))
        f.seek(first)
        data = f.read(max(0, last - first), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.size else np.zeros(1, dtype=np.float32)
    bucket = max(1, len(mono) // points)
    usable = (len(mono) // bucket) * bucket or len(mono)
    blocks = mono[:usable].reshape(-1, bucket) if usable >= bucket else mono.reshape(1, -1)
    times = start + (np.arange(len(blocks)) * bucket) / sr
    return times, blocks.min(axis=1), blocks.max(axis=1)


def waveform_figure(path: str | Path, start: float = 0.0, end: float | None = None,
                    color: str = "#4C78A8", markers: tuple[float, ...] = ()):
    times, lo, hi = envelope(path, start, end)
    fig, ax = plt.subplots(figsize=(10, 1.8), dpi=100)
    ax.fill_between(times, lo, hi, color=color, linewidth=0)
    for m in markers:
        ax.axvline(m, color="#E45756", linewidth=1.5)
    ax.set_ylim(-1, 1)
    ax.set_xlim(times[0] if len(times) else start, times[-1] if len(times) else (end or 1))
    ax.set_yticks([])
    ax.set_xlabel("seconds")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def silence_figure(duration: float):
    fig, ax = plt.subplots(figsize=(10, 1.8), dpi=100)
    ax.axhline(0, color="#999999")
    ax.set_xlim(0, duration)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("seconds (silence)")
    fig.tight_layout()
    return fig


def close(fig) -> None:
    plt.close(fig)
