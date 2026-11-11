"""Vocal separation engine tests (the real models are slow; skip with MAS_SKIP_DEMUCS=1)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import audio_separator as sep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import separation_worker as worker  # noqa: E402

SLOW = pytest.mark.skipif(os.environ.get("MAS_SKIP_DEMUCS") == "1" or not sep.best_available(),
                          reason="separation models skipped or not installed")


def test_bleed_mask_only_removes_energy():
    """The mask scales each time-frequency bin by 0..1, so it never adds sound."""
    rng = np.random.default_rng(0)
    v = rng.normal(0, 0.1, (44100 * 2, 2)).astype("float32")
    i = rng.normal(0, 0.1, (44100 * 2, 2)).astype("float32")
    v2, i2 = worker.bleed_mask(v, i, 44100)
    assert v2.shape == v.shape and i2.shape == i.shape
    assert (v2 ** 2).sum() <= (v ** 2).sum() * 1.001
    assert (i2 ** 2).sum() <= (i ** 2).sum() * 1.001


def test_bleed_mask_moves_quiet_bleed_to_other_stem():
    sr = 44100
    t = np.arange(sr * 2) / sr
    voice, music = np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 3000 * t)
    # Vocal estimate contains a little of the music (bleed) and vice versa.
    v = np.stack([voice + 0.1 * music] * 2, 1).astype("float32")
    i = np.stack([music + 0.1 * voice] * 2, 1).astype("float32")
    v2, _ = worker.bleed_mask(v, i, sr)
    spec = np.abs(np.fft.rfft(v2[sr // 2: sr * 3 // 2, 0]))
    freqs = np.fft.rfftfreq(sr, 1 / sr)
    level = lambda f: spec[np.argmin(np.abs(freqs - f))]
    assert level(3000) < level(440) * 0.01  # bleed reduced by > 40 dB here


def test_mode_selection(monkeypatch):
    monkeypatch.setattr(sep, "total_ram_gb", lambda: 2.0)
    assert sep.active_mode() == "fast"
    monkeypatch.setattr(sep, "total_ram_gb", lambda: 16.0)
    assert sep.active_mode() == ("best" if sep.best_available() else "fast")


@SLOW
def test_best_engine_end_to_end(tmp_path):
    """Runs the real worker on 30 s of audio (crosses a piece boundary)."""
    sr = 44100
    t = np.arange(sr * 30) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 0.25 * t)) / 2
    src = tmp_path / "in.wav"
    sf.write(src, np.stack([x, x], 1).astype("float32"), sr, subtype="PCM_16")
    seen = []
    stems = sep.separate(src, tmp_path / "out", progress=lambda v, s: seen.append(v), mode="best",
                         clean=False)
    for name in ("vocals", "instrumental"):
        info = sf.info(stems[name])
        assert info.samplerate == sr and info.duration == pytest.approx(30, abs=0.01)
    assert seen[-1] == 1.0 and not (tmp_path / "out" / "_work").exists()
    v, _ = sf.read(stems["vocals"], always_2d=True)
    i, _ = sf.read(stems["instrumental"], always_2d=True)
    # Without the extra filter the two stems add back up to the input (no seams at piece joins).
    err = (v + i) - np.stack([x, x], 1)
    assert np.sqrt((err ** 2).mean()) < 0.01
