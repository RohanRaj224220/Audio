import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    """Point job folders at a temp dir so tests never touch real jobs."""
    import file_utils

    monkeypatch.setattr(file_utils, "WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr(file_utils, "JOBS_DIR", tmp_path / "workspace" / "jobs")
    yield


@pytest.fixture
def tone_wav(tmp_path) -> Path:
    """4 s stereo test tone: 440 Hz left, 660 Hz right, 44.1 kHz."""
    sr = 44100
    t = np.arange(int(4 * sr)) / sr
    data = np.stack([0.4 * np.sin(2 * np.pi * 440 * t), 0.4 * np.sin(2 * np.pi * 660 * t)], axis=1)
    path = tmp_path / "tone.wav"
    sf.write(path, data.astype("float32"), sr, subtype="PCM_16")
    return path


@pytest.fixture
def test_video(tmp_path) -> Path:
    import ffmpeg_utils

    path = tmp_path / "clip.mp4"
    subprocess.run([ffmpeg_utils.ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=4",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                    "-c:v", "libx264" if ffmpeg_utils.has_encoder("libx264") else "mpeg4",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def duration_of(path) -> float:
    import ffmpeg_utils

    return ffmpeg_utils.probe(path).duration


def dominant_freq(path, channel=0) -> float:
    data, sr = sf.read(str(path), always_2d=True)
    x = data[:, channel]
    x = x[len(x) // 4: 3 * len(x) // 4]
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spec)]


def rms(path) -> float:
    data, _ = sf.read(str(path), always_2d=True)
    return float(np.sqrt(np.mean(data ** 2)))
