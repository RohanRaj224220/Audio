"""Karaoke lyrics: transcribe the vocals with word timings and export LRC / SRT / ASS.

Uses faster-whisper (CPU). Works best on the separated vocals. Sung words are harder
to recognise than speech, so the result is meant to be corrected by hand.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Callable, Optional

from file_utils import WORKSPACE

log = logging.getLogger("mas.lyrics")

MODEL_SIZE = os.environ.get("MAS_WHISPER_MODEL", "small")
MODEL_DIR = Path(os.environ.get("MAS_MODEL_DIR", WORKSPACE / "models")) / "whisper"
MAX_LINE_WORDS = 9
MAX_GAP_SECONDS = 1.2


class LyricsError(Exception):
    pass


def available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def model_is_cached() -> bool:
    return MODEL_DIR.exists() and any(MODEL_DIR.rglob("model.bin"))


def transcribe(audio: Path, language: str | None = None,
               progress: Optional[Callable[[float, str], None]] = None) -> list[dict]:
    """Return lyric lines: {"start", "end", "text", "words": [[start, end, word], ...]}."""
    if not available():
        raise LyricsError("Lyrics are not available on this server.")
    from faster_whisper import WhisperModel

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(0.02, "Loading the lyrics model...")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8",
                             download_root=str(MODEL_DIR))
        segments, info = model.transcribe(str(audio), language=language or None,
                                          word_timestamps=True, vad_filter=True,
                                          condition_on_previous_text=False)
        words: list[list] = []
        for seg in segments:
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    words.append([round(float(w.start), 3), round(float(w.end), 3), text])
            if progress and info.duration:
                progress(min(0.98, 0.05 + 0.93 * seg.end / info.duration), "Listening to the words...")
    except Exception as e:  # noqa: BLE001
        log.exception("Transcription failed")
        raise LyricsError("The lyrics could not be made.") from e
    lines = group_lines(words)
    log.info("Transcribed %s: %d words, %d lines (%s)", audio, len(words), len(lines),
             getattr(info, "language", "?"))
    if progress:
        progress(1.0, "Done")
    return lines


def group_lines(words: list[list]) -> list[dict]:
    """Break a flat word list into singable lines (at pauses, punctuation or length)."""
    lines, cur = [], []
    for w in words:
        if cur and (w[0] - cur[-1][1] > MAX_GAP_SECONDS or len(cur) >= MAX_LINE_WORDS
                    or cur[-1][2][-1:] in ".!?"):
            lines.append(_line(cur))
            cur = []
        cur.append(list(w))
    if cur:
        lines.append(_line(cur))
    return lines


def _line(words: list[list]) -> dict:
    return {"start": words[0][0], "end": words[-1][1], "text": " ".join(w[2] for w in words),
            "words": words}


def retime(start: float, end: float, text: str, old: dict | None = None) -> dict:
    """A line after hand editing. Word timings are kept when only the line moved,
    and spread evenly over the line when the words changed."""
    text = " ".join(str(text or "").split())
    start, end = float(start), max(float(end), float(start) + 0.05)
    tokens = text.split()
    if old and [w[2] for w in old["words"]] == tokens and old["end"] > old["start"]:
        scale = (end - start) / (old["end"] - old["start"])
        words = [[round(start + (a - old["start"]) * scale, 3),
                  round(start + (b - old["start"]) * scale, 3), t] for a, b, t in old["words"]]
    else:
        weights = [len(t) + 1 for t in tokens]
        total, t, words = sum(weights) or 1, start, []
        for tok, wt in zip(tokens, weights):
            nxt = t + (end - start) * wt / total
            words.append([round(t, 3), round(nxt, 3), tok])
            t = nxt
    return {"start": round(start, 3), "end": round(end, 3), "text": text, "words": words}


def shift(lines: list[dict], offset: float = 0.0, scale: float = 1.0) -> list[dict]:
    """Move/stretch every timing (e.g. after a speed change: scale = 1 / speed)."""
    f = lambda t: round(max(0.0, t * scale + offset), 3)  # noqa: E731
    return [{"start": f(l["start"]), "end": f(l["end"]), "text": l["text"],
             "words": [[f(a), f(b), w] for a, b, w in l["words"]]} for l in lines]


def map_to_timeline(lines: list[dict], clips, speed: float = 1.0) -> list[dict]:
    """Lyric timings of the original song -> timings in the edited song.

    Follows trims, reordering, duplicates and per-clip speed; `speed` is the overall
    speed applied afterwards. Lines that were cut out are dropped.
    """
    out, offset = [], 0.0
    for clip in clips:
        cs = clip.speed or 1.0
        if not clip.is_silence:
            conv = lambda t: round((offset + (min(max(t, clip.start), clip.end) - clip.start) / cs)  # noqa: E731
                                   / speed, 3)
            for l in lines:
                if clip.start - 1e-3 <= l["start"] < clip.end:
                    out.append({"start": conv(l["start"]), "end": conv(l["end"]), "text": l["text"],
                                "words": [[conv(a), conv(b), w] for a, b, w in l["words"]
                                          if a < clip.end]})
        offset += clip.duration / cs
    return out


def _stamp(t: float, sep: str = ".", hours: bool = False, digits: int = 2) -> str:
    unit = 10 ** digits
    total = int(round(max(0.0, t) * unit))
    whole, frac = divmod(total, unit)
    m, s = divmod(whole, 60)
    if hours:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}{sep}{frac:0{digits}d}"
    return f"{m:02d}:{s:02d}{sep}{frac:0{digits}d}"


def to_lrc(lines: list[dict], word_level: bool = True) -> str:
    """LRC lyrics. word_level=True writes the enhanced (A2) format with a time per word."""
    out = []
    for l in lines:
        if word_level and l["words"]:
            body = " ".join(f"<{_stamp(a)}>{w}" for a, _, w in l["words"])
            out.append(f"[{_stamp(l['start'])}]{body} <{_stamp(l['end'])}>")
        else:
            out.append(f"[{_stamp(l['start'])}]{l['text']}")
    return "\n".join(out) + "\n"


def to_srt(lines: list[dict]) -> str:
    out = []
    for n, l in enumerate(lines, 1):
        a = "0" + _stamp(l["start"], ",", hours=True, digits=3)
        b = "0" + _stamp(l["end"], ",", hours=True, digits=3)
        out.append(f"{n}\n{a} --> {b}\n{l['text']}\n")
    return "\n".join(out)


def to_ass(lines: list[dict], title: str = "Karaoke") -> str:
    """ASS subtitles with karaoke fill (\\kf) per word; plays in VLC, mpv, Aegisub."""
    head = (f"[Script Info]\nTitle: {title}\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
            "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Karaoke,Arial,54,&H0000E5FF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,"
            "0,0,1,3,1,2,60,60,60,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    rows = []
    for l in lines:
        text, t = "", l["start"]
        for a, b, w in l["words"] or [[l["start"], l["end"], l["text"]]]:
            gap = int(round((a - t) * 100))
            if gap > 0:
                text += f"{{\\k{gap}}}"
            text += f"{{\\kf{max(1, int(round((b - a) * 100)))}}}{w} "
            t = b
        rows.append(f"Dialogue: 0,{_stamp(l['start'], hours=True)},{_stamp(l['end'], hours=True)},"
                    f"Karaoke,,0,0,0,,{text.strip()}")
    return head + "\n".join(rows) + "\n"
