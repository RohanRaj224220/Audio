import subprocess

import numpy as np
import pytest
import soundfile as sf

import compressor as c
import ffmpeg_utils

MB = c.MB


def wav(path, seconds, noise=False, sr=44100):
    t = np.arange(int(seconds * sr)) / sr
    if noise:
        x = np.random.default_rng(1).uniform(-0.5, 0.5, (len(t), 2))
    else:
        x = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, 1)
    sf.write(path, x.astype("float32"), sr, subtype="PCM_16")
    return path


def noisy_video(path, seconds):
    subprocess.run([ffmpeg_utils.ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=25:duration={seconds}",
                    "-f", "lavfi", "-i", f"anoisesrc=d={seconds}:a=0.3",
                    "-vf", "noise=alls=40:allf=t", "-c:v", "libx264", "-g", "25", "-b:v", "3M",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def test_flac_is_bit_exact_and_smaller(tmp_path):
    src = wav(tmp_path / "a.wav", 10)
    out = c.to_flac(src, tmp_path / "a.flac")
    assert c.pcm_md5(src) == c.pcm_md5(out)
    assert out.stat().st_size < src.stat().st_size * 0.5


def test_single_audio_lossless_when_it_fits(tmp_path):
    src = wav(tmp_path / "a.wav", 10)
    part = c.compress_single(src, tmp_path / "out", 180, "song", is_video=False)
    assert part.path.suffix == ".flac" and part.lossless and part.size <= 180 * MB
    assert c.pcm_md5(part.path) == c.pcm_md5(src)
    assert list((tmp_path / "out").iterdir()) == [part.path]  # exactly one output file


def test_single_audio_falls_back_to_one_file_that_fits(tmp_path):
    src = wav(tmp_path / "noise.wav", 30, noise=True)  # FLAC ~4.5 MB, barely compressible
    part = c.compress_single(src, tmp_path / "out", 1.5, "song", is_video=False)
    assert part.path.suffix == ".mp3" and not part.lossless
    assert part.size <= 1.5 * MB and part.lossless_size > 1.5 * MB
    assert list((tmp_path / "out").iterdir()) == [part.path]
    assert ffmpeg_utils.probe(part.path).duration == pytest.approx(30, abs=0.2)


def test_single_video_kept_as_is_when_it_fits(tmp_path):
    src = noisy_video(tmp_path / "v.mp4", 10)
    part = c.compress_single(src, tmp_path / "out", 180, "clip", is_video=True)
    assert part.path == src and part.lossless


def test_single_video_compressed_to_fit(tmp_path):
    src = noisy_video(tmp_path / "v.mp4", 20)
    limit = max(1.0, src.stat().st_size / MB / 3)
    part = c.compress_single(src, tmp_path / "out", limit, "clip", is_video=True)
    info = ffmpeg_utils.probe(part.path)
    assert part.size <= limit * MB and not part.lossless and part.path.suffix == ".mp4"
    assert info.has_video and info.has_audio and info.duration == pytest.approx(20, abs=0.5)


def test_fit_audio_respects_limit(tmp_path):
    src = wav(tmp_path / "a.wav", 60)
    part = c.fit_audio(src, tmp_path / "a.mp3", 1.0)
    assert part.size <= 1.0 * MB and not part.lossless
    with pytest.raises(c.CompressionError):
        c.fit_audio(wav(tmp_path / "long.wav", 600), tmp_path / "b.mp3", 1.0)


def test_fit_video_respects_limit(tmp_path):
    src = noisy_video(tmp_path / "v.mp4", 20)
    part = c.fit_video(src, tmp_path / "small.mp4", 2.0)
    info = ffmpeg_utils.probe(part.path)
    assert part.size <= 2.0 * MB and info.has_video and info.has_audio
    assert info.duration == pytest.approx(20, abs=0.5)


def test_limit_validation(tmp_path):
    with pytest.raises(c.CompressionError):
        c.compress_single(wav(tmp_path / "a.wav", 2), tmp_path / "o", 0.5, "s", False)
