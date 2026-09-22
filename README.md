# live-transcribe

> **Subtitles that pull up before you do.** Play a local video, and the words
> are *already on screen* by the time the scene reaches them. No cloud, no
> upload, no upload spinner. Just your Mac's GPU being the main character.

A local video player that keeps a transcript **at least 10 seconds ahead of the
playhead**, on Apple Silicon. Captions overlay the video, the transcript panel
is click-to-seek, and the whole thing exports as SRT / VTT / TXT.

Everything stays on your machine. The *only* time it touches the network is the
one-time model download from Hugging Face. After that it's fully offline, lowkey
feral, and proud of it.

---

## ok but what does it actually do

You know how normal subtitles feel like they're buffering at the exact moment
someone says something important? This flips that. The model runs *ahead* of
where you're watching, so by the time the video catches up, the caption is
sitting there like it's been waiting for you. It's giving "I got here early and
saved us seats."

- Play a local video → captions show up ahead of the playhead.
- The side panel logs every line as it lands; click any line to jump there.
- Auto-detects the spoken language and translates it to English in one pass.
- Export the whole thing as SRT, VTT, or TXT whenever.

---

## the lore (how it works, fr)

Most "live" transcription tries to do the hard thing: stream audio into a model
in real time and hope it keeps up. This app is delulu in a *productive* way — it
refuses to be real-time. A local file's audio is random-access, so "10 seconds
ahead" is literally just arithmetic. Here's the pipeline, top to bottom:

```
ffmpeg  ──►  mono 16 kHz float32 PCM (cached)  ──►  np.memmap
ffmpeg  ──►  silencedetect ──► chunk boundaries snapped to silence
worker thread  ──►  transcribe chunk i ONLY while chunks[i].start <= playhead + 10s
```

### 1. Demux once, then it's just vibes and indexes

`ffmpeg` rips the audio track down to mono 16 kHz float32 PCM
(`pipeline.extract_pcm`). That buffer is cached by file identity
(`path + size + mtime`), so reopening the same video costs nothing. It's loaded
as a `np.memmap`, which means **seeking to any timestamp is an array index** —
`SAMPLE_RATE * seconds`. No decode-on-the-fly, no ring buffer, no VAD. The audio
is sitting in memory-mapped storage and we just point at it.

### 2. Chunks that snap to silence (no mid-vowel violence)

Whisper eats ~30-second windows, so `plan_chunks` cuts the PCM into ~30s chunks.
But cutting on a hard 30s mark is how you slice a word in half and get cursed
subtitles. So before planning, `ffmpeg silencedetect` finds the quiet spots
(≥0.4s below −35 dB), and every chunk edge gets nudged up to ±6s to land on a
silence. Whole words enter, whole words leave.

### 3. The look-ahead scheduler (the actual flex)

`LookaheadScheduler` is one worker thread with one rule:

> transcribe chunk `i` only while `chunks[i].start <= playhead + 10s`.

- **Playing** → it keeps the next chunk (and the `LOOKAHEAD` window) hot.
- **Paused** → the window freezes; it doesn't burn your GPU into the floor.
- **Rewound** → that chunk is already in the cache, so it re-serves instantly
  and never re-runs the model.
- **Playhead window first** → the chunk you're about to hit always jumps the
  queue; the rest of the file fills in after, like a progressive download.

On an 88s clip with the playhead at 0, exactly one of its three chunks is
transcribed. The scheduler literally idles until you need it. That's the whole
trick, and it's not even a trick.

### 4. Whisper, but it translates too

One backend: `mlx-community/whisper-large-v3-mlx` running through MLX on the
Apple GPU. Whisper's built-in `task="translate"` turns any spoken language into
English **in a single pass** — no separate "detect language, then translate"
stage. It uses full `large-v3`, *not* `large-v3-turbo`, because turbo is a
menace: it silently ignores `task="translate"` and just transcribes instead.
(delulu behavior. don't trust it.) Point `LT_TRANSLATE_MODEL` at another repo if
you want to swap.

### 5. Make the browser accept the file (or it just dies)

The browser decides what plays, and the *extension lies*. An HEVC `.mp4` looks
perfectly playable and then Chromium refuses the streams. So every file is
probed first (`probe_codecs`): if it's a browser-safe container
(mp4/mov/m4v/webm) with browser-safe codecs (H.264/VP8/VP9/AV1 + AAC/MP3/Opus/
Vorbis), it's served untouched. Otherwise `PlaybackPrep` transcodes it once to
H.264/AAC mp4 in a **background thread with a live progress bar**, and caches
it. On Apple Silicon that's `h264_videotoolbox` (hardware, fast). If conversion
faceplants, the original is served anyway and the UI tells you why —
transcription still works even when playback can't.

### 6. Reflow, so captions don't read like a Twitter rant

Raw Whisper segments can run long. `reflow_cues` splits them into **at most 2
balanced lines** (`≤ 42` chars a line), and re-times the split proportionally to
text length on each side. So a long line doesn't just get chopped — it gets
divided into pieces that land roughly when they're spoken, and the shorter
language side gets sliced to match. CJK gets hard-wrapped (no spaces to lean
on). The result reads clean over the video instead of wrapping into a 3rd line
that covers the actor's face.

---

## why it slaps

Because "live transcription" is usually a lie that stutters. This one is honest:
transcribe *ahead*, cache *everything*, and let a memmap index make seeking
instant. The scheduler's bound is the whole performance model — it does the
minimum work to guarantee you never wait for a caption.

---

## setup

Requires **macOS on Apple Silicon**, plus `ffmpeg` and `uv`.

```sh
brew install ffmpeg uv
make setup              # install deps
make run                # http://localhost:8000
```

Open <http://localhost:8000> and hit **Open Video…**. That's a *native macOS
open panel*, not a web upload — so nothing gets copied, nothing leaves the
machine, no `input type=file` nonsense. First run waits for the model to load
(~3 GB download, one time); after that it's resident and offline.

---

## controls

The player has a custom transport bar that auto-hides while you watch:

`play/pause` · `−10s` · `+10s` · `volume + mute` · `time` · `speed` · `CC` ·
`PiP` · `fullscreen`

And a full keyboard:

| Keys | Action |
| --- | --- |
| `⌘O` | Open a video |
| `Space` / `K` | Play / pause |
| `←` / `→` | Seek 5s (`⇧` for 30s) |
| `J` / `L` | Seek 10s |
| `↑` / `↓` | Volume |
| `M` | Mute |
| `0`–`9` | Jump to 0–90% |
| `Home` / `End` | Start / end |
| `,` / `.` | Frame step (while paused) |
| `⇧,` / `⇧.` (`<` / `>`) | Playback speed |
| `C` | Toggle captions |
| `F` | Fullscreen |
| `P` | Picture-in-picture |
| `?` | Show the shortcut list |
| `Esc` | Cancel an in-progress open |

The timeline under the video is drag-scrubbable and doubles as a pipeline meter:
one cell per transcription chunk, lighting up as the words land. Click any
transcript line to jump there; hover for a **Copy** button, or **Copy** the
whole transcript from the panel header.

---

## language

One profile: **Auto → English**. `whisper-large-v3` auto-detects the spoken
language and its built-in `task=translate` renders the transcript in English,
all in a single pass. Point `LT_TRANSLATE_MODEL` at another repo id to swap the
model — `large-v3-turbo` will not work here because it silently ignores
`task="translate"` and transcribes instead.

---

## environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | — | Hugging Face token; also accepted as `HUGGING_FACE_HUB_TOKEN` |
| `LT_LOOKAHEAD` | `10.0` | seconds of transcript to keep ahead of the playhead |
| `LT_TRANSLATE_MODEL` | `mlx-community/whisper-large-v3-mlx` | ASR repo (must be translate-capable) |
| `LT_REMUX` | `1` | convert unplayable files to a browser-safe mp4 (`0` disables) |
| `LT_CACHE` | `~/.cache/live-transcribe` | derived PCM + remuxed mp4 |

```sh
HF_TOKEN=hf_xxx LT_LOOKAHEAD=15 make run
```

---

## notes and limits (the honest part)

- **Playback formats.** The browser decides what plays, and the *container is
  not the whole story* — see the remux section above. `LT_REMUX=0` disables
  conversion and serves the original no matter what.
- **Cached conversions are re-probed before use**, so a cache written by an
  older version is discarded instead of served.
- **Conversion failure isn't fatal.** The original is served anyway and the
  reason is shown; transcription still works even when playback can't.
- **The buffer overshoots by up to one chunk (30s)**, since whole chunks are
  transcribed. It guarantees *at least* `LT_LOOKAHEAD` seconds ahead.
- **One video at a time.** `/media` serves the current one; opening another
  cancels whatever was in flight.
- **The file panel is driven by `osascript`** (`app.py:choose_file`). macOS may
  ask once for permission to control System Events, which is only used to bring
  the panel to the front; if denied, the panel still works but may open behind
  the browser. Off macOS, the UI falls back to a path field.
- **Chunk edges snap to detected silence** within ±6s of each 30s mark, and
  never extend past Whisper's 30s window.
- **Chunk planning re-runs `silencedetect` on every open** (a fast audio-only
  pass); the PCM extraction itself is cached by file identity.
- **Settings are stateless.** Volume, speed, captions and the sidebar reset per
  session — no localStorage, nothing persisted. Close the tab and it's gone.

---

## the UI

The interface follows Apple's fluid-interface guidance (WWDC *Designing Fluid
Interfaces*): feedback fires on pointer-*down* rather than release, the timeline
tracks the pointer 1:1 with `setPointerCapture`, chrome is a translucent
`backdrop-filter` material rather than opaque bars, and tracking is size-specific
— tight on the large display type, near zero on body, tabular numerals on every
clock so it never jitters.

It also honors `prefers-reduced-motion`, `prefers-reduced-transparency` and
`prefers-contrast`, and adapts to light and dark via `prefers-color-scheme`.
Transcript rows are keyboard-reachable, the transport is fully labeled for
screen readers, and every control has a focus ring.

---

## tests

```sh
make test
```

Covers chunk planning, the scheduler (look-ahead bound, idling, seek re-serving
from cache, error surfacing), cue reflow, playback-format detection, and
conversion cancellation — against a fake backend, so no model or ffmpeg needed.

```sh
make help                        # list targets
make dev                         # auto-reload
make run PORT=9000               # different port
HF_TOKEN=hf_xxx make run         # pass a token
make test                        # run the suite
make check                       # byte-compile
make clean                       # drop .venv and caches
make cache-clean                 # drop derived PCM / remuxed mp4
```

All Python goes through `uv` — no manual venv activation.

---

## license

[MIT](LICENSE) — do whatever, just keep the copyright notice. Go build something
cool with it.
