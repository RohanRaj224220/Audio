"""Paths, filenames, job folders, cleanup and logging setup."""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
import unicodedata
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import BinaryIO, Iterable

APP_ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("MAS_WORKSPACE", APP_ROOT / "workspace"))
JOBS_DIR = WORKSPACE / "jobs"
LOG_DIR = Path(os.environ.get("MAS_LOG_DIR", APP_ROOT / "logs"))
LOG_FILE = LOG_DIR / "application.log"

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "200"))
JOB_MAX_AGE_SECONDS = 24 * 3600
TOUCH_FILE = ".last_used"

AUDIO_EXTENSIONS = {"mp3", "wav", "flac", "m4a", "aac", "ogg"}
VIDEO_EXTENSIONS = {"mp4", "mov", "mkv", "webm"}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

log = logging.getLogger("mas.files")


class FileTooLargeError(Exception):
    pass


class UnsupportedFormatError(Exception):
    pass


def setup_logging() -> logging.Logger:
    """Configure the 'mas' logger once. Safe to call many times."""
    root = logging.getLogger("mas")
    if getattr(root, "_mas_configured", False):
        return root
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    root._mas_configured = True  # type: ignore[attr-defined]
    return root


def sanitize_filename(name: str, default: str = "file") -> str:
    """Return a safe file name: no paths, no odd characters, limited length."""
    name = unicodedata.normalize("NFKD", name or "")
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" ._")
    name = re.sub(r"[ _]+", "_", name)
    if not name:
        return default
    stem, dot, ext = name.rpartition(".")
    if dot and stem:
        return f"{stem[:80]}.{ext[:10].lower()}"
    return name[:80]


def file_extension(name: str) -> str:
    return Path(name).suffix.lower().lstrip(".")


def check_extension(name: str) -> str:
    ext = file_extension(name)
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(ext)
    return ext


def new_job() -> tuple[str, Path]:
    job_id = uuid.uuid4().hex
    job_dir = job_path(job_id)
    for sub in ("source", "work", "stems", "renders"):
        (job_dir / sub).mkdir(parents=True, exist_ok=True)
    touch_job(job_id)
    log.info("Created job %s", job_id)
    return job_id, job_dir


def job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id or ""):
        raise ValueError("Invalid job id")
    return JOBS_DIR / job_id


def touch_job(job_id: str) -> None:
    """Mark a job as in use so cleanup leaves it alone."""
    try:
        p = job_path(job_id)
        if p.exists():
            (p / TOUCH_FILE).touch()
    except (OSError, ValueError):
        pass


def save_stream(stream: BinaryIO, dest: Path, size_hint: int | None = None) -> Path:
    """Copy an upload-like stream to disk in chunks, enforcing MAX_UPLOAD_MB."""
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    if size_hint is not None and size_hint > limit:
        raise FileTooLargeError(size_hint)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    try:
        if hasattr(stream, "seek"):
            stream.seek(0)
        with open(tmp, "wb") as out:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise FileTooLargeError(written)
                out.write(chunk)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    log.info("Saved upload to %s (%d bytes)", dest, written)
    return dest


def cleanup_old_jobs(active_ids: Iterable[str] = (), max_age: float = JOB_MAX_AGE_SECONDS,
                     now: float | None = None) -> list[str]:
    """Delete job folders not used for `max_age` seconds. Never touches active ids."""
    now = time.time() if now is None else now
    active = set(active_ids)
    removed = []
    if not JOBS_DIR.exists():
        return removed
    for d in JOBS_DIR.iterdir():
        if not d.is_dir() or d.name in active:
            continue
        marker = d / TOUCH_FILE
        try:
            last = marker.stat().st_mtime if marker.exists() else d.stat().st_mtime
        except OSError:
            continue
        if now - last > max_age:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
    if removed:
        log.info("Cleaned up %d old jobs", len(removed))
    return removed


def unique_render_path(job_dir: Path, stem: str, ext: str) -> Path:
    return job_dir / "renders" / f"{sanitize_filename(stem, 'output')}_{uuid.uuid4().hex[:6]}.{ext}"
