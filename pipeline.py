"""Media prep, chunk planning, and the look-ahead scheduler.

The whole trick: a local video's audio track is random-access once demuxed to
raw PCM, so "10 seconds ahead of the playhead" is just an index into a
memmap. No streaming ASR, no ring buffer, no VAD.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CACHE_DIR = Path(os.environ.get("LT_CACHE", Path.home() / ".cache" / "live-transcribe"))

CHUNK_TARGET = 30.0  # whisper's native window
CHUNK_MAX = 30.0
SILENCE_SNAP = 6.0  # how far a boundary may drift to land on a silence
LOOKAHEAD = float(os.environ.get("LT_LOOKAHEAD", "10.0"))

VIDEO_EXT = {
    ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi",
    ".ts", ".flv", ".wmv", ".mpg", ".mpeg",
}
# Containers the browser can play as-is, given browser-safe codecs inside.
CONTAINER_OK = {".mp4", ".m4v", ".mov", ".webm"}
BROWSER_VIDEO = {"h264", "vp8", "vp9", "av1"}
BROWSER_AUDIO = {"aac", "mp3", "opus", "vorbis"}


@dataclass
class Cue:
    """One subtitle line. `target` is empty when no translation ran."""

    start: float
    end: float
    source: str
    target: str = ""


# --------------------------------------------------------------------------
# ffmpeg media prep
# --------------------------------------------------------------------------


def _cache_key(video: Path) -> str:
    st = video.stat()
    raw = f"{video.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def extract_pcm(video: Path) -> Path:
    """Demux the audio track to mono 16k float32, cached by file identity."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{_cache_key(video)}.f32"
    meta = out.with_suffix(".json")
    if out.exists() and meta.exists() and out.stat().st_size > 0:
        return out
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", str(out)],
        check=True,
    )
    meta.write_text(json.dumps({"source": str(video)}))
    return out


def probe_codecs(video: Path) -> dict[str, str]:
    """First video/audio codec name per stream type, e.g. {"video": "hevc"}."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    codecs: dict[str, str] = {}
    for stream in json.loads(proc.stdout).get("streams", []):
        codecs.setdefault(stream.get("codec_type", ""), stream.get("codec_name", ""))
    return codecs


def _needs_conversion(video: Path, codecs: dict[str, str]) -> str | None:
    """Why this file can't be played as-is, or None if it can."""
    if video.suffix.lower() not in CONTAINER_OK:
        return "container"
    video_codec = codecs.get("video")
    if video_codec not in BROWSER_VIDEO:
        return f"video codec '{video_codec or 'none'}'"
    audio_codec = codecs.get("audio")
    if audio_codec is not None and audio_codec not in BROWSER_AUDIO:
        return f"audio codec '{audio_codec}'"
    return None


@lru_cache(maxsize=1)
def _video_encoder() -> str:
    """Hardware H.264 when the ffmpeg build has it, else libx264."""
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                          capture_output=True, text=True)
    return "h264_videotoolbox" if "h264_videotoolbox" in proc.stdout else "libx264"


def _duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


class PlaybackPrep:
    """Produces a browser-playable mp4, in the background if it must transcode.

    Codec matters as much as container: an HEVC .mp4 looks playable by
    extension but Chromium refuses the streams, so the player just dies. We
    probe first and only re-encode what the browser can't already decode.
    """

    def __init__(self, source: Path):
        self.source = source
        self.output: Path | None = None
        self.error: str | None = None
        self.reason: str | None = None
        self.progress = 0.0
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if os.environ.get("LT_REMUX", "1") == "0":
            return self._finish(self.source)
        try:
            codecs = probe_codecs(self.source)
            self.reason = _needs_conversion(self.source, codecs)
        except Exception as exc:
            self.error = f"could not probe streams: {exc}"
            self._ready.set()
            return

        if self.reason is None:
            return self._finish(self.source)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / f"{_cache_key(self.source)}.mp4"
        if cache.exists() and cache.stat().st_size > 0 and self._cache_is_good(cache):
            return self._finish(cache)
        cache.unlink(missing_ok=True)

        self._thread = threading.Thread(target=self._run, args=(codecs, cache), daemon=True)
        self._thread.start()

    def _finish(self, output: Path) -> None:
        self.output = output
        self._ready.set()

    def _cache_is_good(self, cache: Path) -> bool:
        """Never trust a cache written by an older ffmpeg invocation."""
        try:
            return _needs_conversion(cache, probe_codecs(cache)) is None
        except Exception:
            return False

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def state(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
            "reason": self.reason,
            "progress": round(self.progress, 3),
            "converted": bool(self.output) and self.output != self.source,
        }

    # -- conversion --------------------------------------------------------

    def _run(self, codecs: dict[str, str], cache: Path) -> None:
        part = cache.with_name(cache.stem + ".part.mp4")
        try:
            duration = _duration(self.source)
            proc = subprocess.Popen(
                _convert_args(self.source, part, codecs),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            for line in proc.stdout:
                if line.startswith("out_time_ms=") and duration > 0:
                    micros = int(line.split("=", 1)[1] or 0)
                    self.progress = min(1.0, micros / 1_000_000 / duration)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr.read() or "ffmpeg failed").strip()[-300:])
            part.replace(cache)
            self.progress = 1.0
            self.output = cache
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            part.unlink(missing_ok=True)
        finally:
            self._ready.set()


def _convert_args(video: Path, out: Path, codecs: dict[str, str]) -> list[str]:
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(video),
            "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn"]
    if codecs.get("video") in BROWSER_VIDEO:
        args += ["-c:v", "copy"]
    elif _video_encoder() == "h264_videotoolbox":
        # ponytail: bitrate-capped hardware encode, no quality tuning.
        # Swap to libx264 -crf if a file comes out visibly soft.
        args += ["-c:v", "h264_videotoolbox", "-b:v", "8M"]
    else:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
    audio = codecs.get("audio")
    if audio is None:
        pass
    elif audio in BROWSER_AUDIO:
        args += ["-c:a", "copy"]
    else:
        args += ["-c:a", "aac", "-b:a", "192k"]
    return args + ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(out)]


_SILENCE = re.compile(r"silence_(start|end):\s*(-?[\d.]+)")


def detect_silences(video: Path) -> list[float]:
    """Midpoints of silence regions, used to avoid cutting mid-word."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(video),
         "-af", "silencedetect=n=-35dB:d=0.4", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts: list[float] = []
    mids: list[float] = []
    for kind, value in _SILENCE.findall(proc.stderr):
        t = float(value)
        if t < 0:
            continue
        if kind == "start":
            starts.append(t)
        elif starts:
            mids.append((starts.pop(0) + t) / 2.0)
    return mids


def plan_chunks(
    duration: float,
    silences: list[float],
    target: float = CHUNK_TARGET,
    max_len: float = CHUNK_MAX,
    snap: float = SILENCE_SNAP,
) -> list[tuple[float, float]]:
    """Contiguous [t0, t1) chunks of at most `max_len`, edges snapped to silence."""
    points = sorted(silences)
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 0.05:
        if duration - start <= max_len:
            chunks.append((start, duration))
            break
        target_end = start + target
        end = _snap(target_end, points, snap)
        if not (start + 1.0 < end <= start + max_len):
            end = min(target_end, duration)
        chunks.append((start, end))
        start = end
    return chunks


def _snap(t: float, points: list[float], tol: float) -> float:
    best, best_d = t, tol
    for p in points:
        d = abs(p - t)
        if d <= best_d:
            best, best_d = p, d
    return best


class AudioSource:
    """Random-access float32 mono audio."""

    def __init__(self, pcm_path: Path):
        self.data = np.memmap(pcm_path, dtype="<f4", mode="r")

    @property
    def duration(self) -> float:
        return len(self.data) / SAMPLE_RATE

    def slice(self, t0: float, t1: float) -> np.ndarray:
        a = max(0, int(t0 * SAMPLE_RATE))
        b = min(len(self.data), int(t1 * SAMPLE_RATE))
        return np.array(self.data[a:b], dtype=np.float32)


def build_media(video: Path) -> tuple[AudioSource, float, list[tuple[float, float]]]:
    pcm = extract_pcm(video)
    source = AudioSource(pcm)
    duration = source.duration
    return source, duration, plan_chunks(duration, detect_silences(video))


# --------------------------------------------------------------------------
# look-ahead scheduler
# --------------------------------------------------------------------------


class LookaheadScheduler:
    """Fills the transcript to the end, like a progressive download.

    The playhead window (`playhead + lookahead`) is always next in the queue
    so playback never waits; remaining chunks fill in afterwards. Results are
    cached by chunk index, so seeking backwards re-serves instantly.
    """

    def __init__(
        self,
        chunks: list[tuple[float, float]],
        run_chunk,
        lookahead: float = LOOKAHEAD,
        on_cues=None,
        poll: float = 0.2,
    ):
        self.chunks = list(chunks)
        self.run_chunk = run_chunk
        self.lookahead = lookahead
        self.on_cues = on_cues or (lambda idx, cues: None)
        self.poll = poll

        self.cache: dict[int, list[Cue]] = {}
        self.next_idx = 0
        self.playhead = 0.0
        self.error: str | None = None
        self.warming = bool(self.chunks)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def set_playhead(self, t: float) -> None:
        with self._lock:
            self.playhead = t
            idx = self._index_at(t)
            if idx is not None:
                self.next_idx = idx

    def state(self) -> dict:
        with self._lock:
            until = self._transcribed_until()
            done = len(self.cache)
            return {
                "playhead": round(self.playhead, 2),
                "transcribed_until": round(until, 2),
                "ahead": round(until - self.playhead, 2),
                "lookahead": self.lookahead,
                "chunks_done": done,
                "chunks_total": len(self.chunks),
                "cached": sorted(self.cache),
                "finished": done >= len(self.chunks),
                "error": self.error,
                "warming": self.warming and done == 0 and self.error is None,
            }

    def all_cues(self) -> list[Cue]:
        with self._lock:
            out: list[Cue] = []
            for idx in sorted(self.cache):
                out.extend(self.cache[idx])
        return sorted(out, key=lambda c: c.start)

    # -- internals ---------------------------------------------------------

    def _index_at(self, t: float) -> int | None:
        for i, (a, b) in enumerate(self.chunks):
            if a <= t < b:
                return i
        return None

    def _transcribed_until(self) -> float:
        """End of the contiguous transcribed run containing the playhead."""
        idx = self._index_at(self.playhead)
        if idx is None:
            return self.chunks[-1][1] if self.chunks else 0.0
        j = idx
        while j in self.cache:
            j += 1
        return self.chunks[j - 1][1] if j > idx else self.chunks[idx][0]

    def _pick(self) -> int | None:
        with self._lock:
            idx = self.next_idx
            while idx < len(self.chunks) and idx in self.cache:
                idx += 1
            self.next_idx = idx
            if idx < len(self.chunks) and self.chunks[idx][0] <= self.playhead + self.lookahead:
                return idx
            for i, _ in enumerate(self.chunks):
                if i not in self.cache:
                    return i
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            idx = self._pick()
            if idx is None:
                self._stop.wait(self.poll)
                continue

            t0, t1 = self.chunks[idx]
            try:
                cues = self.run_chunk(idx, t0, t1)
            except Exception as exc:  # keep the worker alive; surface to the UI
                self.warming = False
                self.error = f"chunk {idx}: {type(exc).__name__}: {exc}"
                self._stop.wait(1.0)
                continue

            self.warming = False
            self.error = None
            with self._lock:
                self.cache[idx] = cues
                if self.next_idx == idx:
                    self.next_idx = idx + 1
            try:
                self.on_cues(idx, cues)
            except Exception:
                pass
