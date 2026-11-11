import os

import pytest

import downloader
import editor
import file_utils
import video_processor as vp
from conftest import duration_of


@pytest.mark.parametrize("url,site", [
    ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "YouTube"),
    ("youtube.com/watch?v=abc", "YouTube"),
    ("https://youtu.be/jNQXAC9IVRw", "YouTube"),
    ("https://music.youtube.com/watch?v=abc", "YouTube"),
    ("https://m.youtube.com/watch?v=abc", "YouTube"),
    ("https://soundcloud.com/artist/track", "SoundCloud"),
    ("https://on.soundcloud.com/abc123", "SoundCloud"),
])
def test_detect_site(url, site):
    assert downloader.detect_site(url) == site


@pytest.mark.parametrize("url", [
    "", "   ", "not a link", "https://vimeo.com/123", "https://evil-youtube.com/watch",
    "https://youtube.com.evil.net/x", "ftp://youtube.com/x", "file:///etc/passwd",
])
def test_detect_site_rejects(url):
    with pytest.raises(downloader.DownloadError):
        downloader.detect_site(url)


def test_friendly_messages():
    f = downloader._friendly
    assert "blocked" in f(Exception("Sign in to confirm you're not a bot"))
    assert "private" in f(Exception("ERROR: Private video"))
    assert "unavailable" in f(Exception("Video unavailable"))
    assert "failed" in f(Exception("something odd"))


def test_js_runtime_available():
    assert downloader.js_runtimes(), "YouTube needs deno or node"


NETWORK = pytest.mark.skipif(os.environ.get("MAS_SKIP_NETWORK") == "1", reason="network skipped")


def _download_and_import(url, expected_seconds):
    job_id, job_dir = file_utils.new_job()
    seen = []
    result = downloader.download_audio(url, job_dir / "source",
                                       progress=lambda p, m: seen.append(p))
    assert result.path.suffix == ".mp3" and result.path.exists()
    assert seen and seen[-1] == 1.0
    project = vp.import_file(job_id, job_dir, result.path, origin_url=result.url)
    assert project.info.duration == pytest.approx(expected_seconds, abs=2)
    assert editor.initial_clips(project.working_audio, project.info.duration)
    return result


@NETWORK
def test_youtube_download():
    # "Me at the zoo" - 19 s, the first video ever uploaded to YouTube.
    r = _download_and_import("https://www.youtube.com/watch?v=jNQXAC9IVRw", 19)
    assert r.site == "YouTube" and r.title == "Me at the zoo"


@NETWORK
def test_soundcloud_download():
    # A Creative Commons (CC BY-SA) instrumental, ~3 min.
    r = _download_and_import(
        "https://soundcloud.com/unwritten-stories/in-the-fog-creative-commons-free-hip-hop-instrumental-download",
        184)
    assert r.site == "SoundCloud"


@NETWORK
def test_playlist_rejected():
    job_id, job_dir = file_utils.new_job()
    with pytest.raises(downloader.DownloadError, match="playlist"):
        downloader.download_audio(
            "https://www.youtube.com/playlist?list=PLbpi6ZahtOH6Blw3RGYpWkSByi_T7Rygb",
            job_dir / "source")


def test_soundcloud_video_rejected():
    job_id, job_dir = file_utils.new_job()
    with pytest.raises(downloader.DownloadError, match="only has audio"):
        downloader.download_video("https://soundcloud.com/artist/track", job_dir / "source")


@NETWORK
def test_youtube_video_download():
    import ffmpeg_utils

    job_id, job_dir = file_utils.new_job()
    seen = []
    r = downloader.download_video("https://www.youtube.com/watch?v=jNQXAC9IVRw", job_dir / "source",
                                  progress=lambda p, m: seen.append(p))
    assert r.path.suffix == ".mp4" and r.path.exists()
    assert seen == sorted(seen) and seen[-1] == 1.0  # progress never jumps backwards
    info = ffmpeg_utils.probe(r.path)
    assert info.has_video and info.has_audio and info.width > 0
    project = vp.import_file(job_id, job_dir, r.path, origin_url=r.url)
    assert project.info.has_video and project.info.duration == pytest.approx(19, abs=2)
