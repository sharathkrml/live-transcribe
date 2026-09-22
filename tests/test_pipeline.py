import time
from pathlib import Path

from pipeline import (
    Cue,
    LookaheadScheduler,
    PlaybackPrep,
    _needs_conversion,
    plan_chunks,
    reflow_cues,
)


# ------------------------------------------------------- playback formats


def test_playable_files_are_left_alone():
    mp4 = Path("clip.mp4")
    assert _needs_conversion(mp4, {"video": "h264", "audio": "aac"}) is None
    assert _needs_conversion(mp4, {"video": "vp9", "audio": "opus"}) is None
    assert _needs_conversion(mp4, {"video": "h264"}) is None  # no audio track


def test_unplayable_codecs_are_detected():
    mp4 = Path("clip.mp4")
    # The regression: an HEVC .mp4 looks fine by extension but Chromium
    # refuses the streams, so the player dies with a demuxer error.
    assert _needs_conversion(mp4, {"video": "hevc", "audio": "aac"}) == "video codec 'hevc'"
    assert _needs_conversion(mp4, {"video": "h264", "audio": "ac3"}) == "audio codec 'ac3'"
    assert _needs_conversion(mp4, {"video": "h264", "audio": "eac3"}) == "audio codec 'eac3'"
    assert _needs_conversion(mp4, {}) == "video codec 'none'"


def test_unplayable_containers_are_detected():
    assert _needs_conversion(Path("clip.mkv"), {"video": "h264", "audio": "aac"}) == "container"
    assert _needs_conversion(Path("clip.avi"), {"video": "h264", "audio": "mp3"}) == "container"


# -------------------------------------------------------------- cancelling


class _FakeProc:
    def __init__(self):
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True


def test_cancel_kills_in_flight_conversion():
    prep = PlaybackPrep(Path("clip.mkv"))
    proc = _FakeProc()
    prep._proc = proc
    prep.cancel()
    assert proc.killed is True
    assert prep._cancelled.is_set()


def test_cancel_is_safe_without_a_conversion():
    prep = PlaybackPrep(Path("clip.mkv"))
    prep.cancel()
    prep.cancel()
    assert prep._cancelled.is_set()
    assert prep.error is None


# --------------------------------------------------------------- planning


def test_plan_respects_max_len_and_covers_duration():
    chunks = plan_chunks(100.0, [29.0], target=30, max_len=30, snap=6)
    assert chunks[0] == (0.0, 29.0)
    assert chunks[-1][1] == 100.0
    assert all(b - a <= 30.0 for a, b in chunks)
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert next_start == prev_end, "chunks must be contiguous"


def test_plan_snaps_to_nearest_silence():
    chunks = plan_chunks(60.0, [27.0, 29.5], target=30, max_len=30, snap=6)
    assert chunks[0][1] == 29.5


def test_plan_rejects_snap_past_max_len():
    """A silence beyond whisper's 30s window must not stretch the chunk."""
    chunks = plan_chunks(60.0, [33.0], target=30, max_len=30, snap=6)
    assert chunks[0][1] == 30.0


def test_plan_ignores_distant_silence():
    chunks = plan_chunks(60.0, [5.0], target=30, max_len=30, snap=6)
    assert chunks[0][1] == 30.0


def test_plan_short_and_empty():
    assert plan_chunks(0.0, []) == []
    assert plan_chunks(12.0, []) == [(0.0, 12.0)]


# -------------------------------------------------------------- scheduling


def make_chunks(n, length=30.0):
    return [(i * length, (i + 1) * length) for i in range(n)]


def make_scheduler(chunks, log, lookahead=10.0):
    def run_chunk(idx, t0, t1):
        log.append(idx)
        return [Cue(t0, t1, f"chunk {idx}")]

    scheduler = LookaheadScheduler(
        chunks, run_chunk, lookahead=lookahead, poll=0.01
    )
    return scheduler


def wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_fills_to_end():
    log = []
    scheduler = make_scheduler(make_chunks(4), log)
    scheduler.start()
    assert wait_until(lambda: scheduler.state()["finished"])
    scheduler.stop()
    assert log == [0, 1, 2, 3]


def test_seek_jumps_the_queue():
    log = []
    hold = True

    def run_chunk(idx, t0, t1):
        log.append(idx)
        while hold and idx == 0:
            time.sleep(0.005)
        return [Cue(t0, t1, f"chunk {idx}")]

    scheduler = LookaheadScheduler(make_chunks(10), run_chunk, poll=0.01)
    scheduler.start()
    assert wait_until(lambda: 0 in log)
    scheduler.set_playhead(95.0)
    hold = False
    assert wait_until(lambda: 3 in log)
    scheduler.stop()
    assert log[0] == 0
    assert log.index(3) < log.index(1)


def test_seek_back_serves_from_cache():
    log = []
    scheduler = make_scheduler(make_chunks(4), log)
    scheduler.start()
    assert wait_until(lambda: 0 in log)
    scheduler.set_playhead(95.0)
    assert wait_until(lambda: scheduler.state()["finished"])
    scheduler.set_playhead(5.0)
    time.sleep(0.05)
    scheduler.stop()
    assert log.count(0) == 1, "cached chunks must not be re-run"


def test_warming_clears_after_first_chunk():
    log = []
    scheduler = make_scheduler(make_chunks(3), log)
    assert scheduler.state()["warming"] is True
    scheduler.start()
    assert wait_until(lambda: scheduler.state()["warming"] is False)
    scheduler.stop()
    assert 0 in log


def test_state_reports_ahead():
    log = []
    scheduler = make_scheduler(make_chunks(10), log)
    scheduler.start()
    assert wait_until(lambda: len(log) >= 1)
    scheduler.set_playhead(5.0)
    assert wait_until(lambda: scheduler.state()["ahead"] >= 25.0)
    state = scheduler.state()
    scheduler.stop()
    assert state["transcribed_until"] >= 30.0
    assert state["error"] is None


def test_errors_are_surfaced_not_fatal():
    def boom(idx, t0, t1):
        raise RuntimeError("model exploded")

    scheduler = LookaheadScheduler(make_chunks(3), boom, poll=0.01)
    scheduler.start()
    assert wait_until(lambda: scheduler.state()["error"] is not None)
    scheduler.stop()
    assert "model exploded" in scheduler.state()["error"]


def test_all_cues_are_ordered():
    log = []
    scheduler = make_scheduler(make_chunks(4), log)
    scheduler.start()
    assert wait_until(lambda: len(log) >= 1)
    scheduler.set_playhead(95.0)
    assert wait_until(lambda: len(scheduler.all_cues()) >= 2)
    cues = scheduler.all_cues()
    scheduler.stop()
    assert [c.start for c in cues] == sorted(c.start for c in cues)


# ------------------------------------------------------------------ reflow


def test_reflow_splits_long_cue_into_two_line_pieces():
    text = " ".join(["word"] * 40)
    cues = reflow_cues([Cue(0.0, 9.0, text, text.upper())])
    assert len(cues) >= 2
    for cue in cues:
        for side in (cue.source, cue.target):
            assert side.count("\n") <= 1
            assert all(len(line) <= 44 for line in side.split("\n"))
    assert cues[0].start == 0.0
    assert cues[-1].end == 9.0
    for prev, nxt in zip(cues, cues[1:]):
        assert abs(prev.end - nxt.start) < 1e-9


def test_reflow_leaves_short_cues_alone():
    cues = reflow_cues([Cue(0.0, 2.0, "hello world")])
    assert [(c.start, c.end, c.source) for c in cues] == [(0.0, 2.0, "hello world")]


def test_reflow_handles_cjk_without_spaces():
    cues = reflow_cues([Cue(0.0, 6.0, "あ" * 100)])
    assert len(cues) >= 2
    assert all(len(line) <= 42 for c in cues for line in c.source.split("\n"))
