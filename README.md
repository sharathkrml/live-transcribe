# live-transcribe

Local video player that keeps a transcript **at least 10 seconds ahead of the
playhead**, on Apple Silicon. Subtitles overlay the video, the transcript panel
is click-to-seek, and the result exports as SRT / VTT / TXT.

Everything runs locally. No cloud, no upload — the only network access is the
one-time model download from Hugging Face.

## Why it's fast

A local file's audio is random-access once demuxed, so "10 seconds ahead" is
just an index into a memory-mapped PCM buffer:

```
ffmpeg -> mono 16k float32 PCM (cached)  ->  np.memmap
ffmpeg silencedetect -> chunk boundaries snapped to silences
worker thread: transcribe chunk i only while chunks[i].start <= playhead + 10s
```

The scheduler idles once the buffer is full. On an 88s clip with the playhead
at 0, exactly one of three chunks is transcribed; rewinding serves from cache
and never re-runs the model.

## Setup

Requires macOS on Apple Silicon, plus `ffmpeg` and `uv`.

```sh
brew install ffmpeg uv
make setup              # en-en / ja-ja / ja-en-fast profiles
make setup-mt           # + torch/transformers for the ja-en profile
make run                # http://localhost:8000
```

Open <http://localhost:8000> and click **Open Video…** — that's a native macOS
open panel, not a web upload, so nothing is copied and nothing leaves the
machine. The first transcription waits for the model to load; after that it
stays resident.

Keyboard: `⌘O` opens a video, `Space` plays/pauses, `←`/`→` seek 5s
(`⇧` for 30s), and the timeline under the video is drag-scrubbable.

All Python goes through `uv` — no manual venv activation.

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

## Language pairs

Pick from the dropdown. Adding a pair is one entry in `PROFILES`
(`backends.py`):

| Profile | Path |
| --- | --- |
| `en-en` | `whisper-large-v3-turbo`, transcribe |
| `en-en-parakeet` | `parakeet-tdt-0.6b-v2` via MLX (English-only, fast) |
| `ja-ja` | `kotoba-whisper-v2.0-mlx`, transcribe |
| `ja-en` | kotoba transcribes, then NLLB-600M translates |
| `ja-en-fast` | `whisper-large-v3` built-in `task=translate` (no `--extra mt`) |
| `auto-en` | `whisper-large-v3` auto-detects the language, then built-in `task=translate` to English |

More models need no code change — point at anything in the local
Hugging Face cache (or any repo id / local snapshot dir):

```sh
# engine is inferred ("parakeet" in the name -> Parakeet, else Whisper/en)
LT_ASR_MODELS="small=mlx-community/whisper-small-mlx,ft=./models/my-parakeet" make run
```

`LT_PARAKEET_MODEL` swaps the repo behind `en-en-parakeet`, and
`LT_ASR_MODEL` / `LT_JA_ASR_MODEL` do the same for the Whisper profiles.
`LT_TRANSLATE_MODEL` backs the translate profiles (`auto-en`, `ja-en-fast`) —
it defaults to full `whisper-large-v3` because `large-v3-turbo` silently
ignores `task="translate"` and transcribes instead.

Switching profiles reloads that profile's models and restarts transcription
from the current position.

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | — | Hugging Face token; also accepted as `HUGGING_FACE_HUB_TOKEN`, and used by both mlx-whisper and transformers |
| `LT_LOOKAHEAD` | `10.0` | seconds of transcript to keep ahead of the playhead |
| `LT_PROFILE` | `en-en` | profile to start with |
| `LT_ASR_MODEL` | `mlx-community/whisper-large-v3-turbo` | primary ASR repo |
| `LT_TRANSLATE_MODEL` | `mlx-community/whisper-large-v3-mlx` | translate-profile ASR repo |
| `LT_JA_ASR_MODEL` | `kaiinui/kotoba-whisper-v2.0-mlx` | Japanese ASR repo |
| `LT_PARAKEET_MODEL` | `mlx-community/parakeet-tdt-0.6b-v2` | Parakeet ASR repo |
| `LT_ASR_MODELS` | — | extra `name=repo` profiles, comma-separated (local dirs OK) |
| `LT_MT_MODEL` | `facebook/nllb-200-distilled-600M` | translation repo |
| `LT_MT_DEVICE` | `cpu` | torch device for NLLB (`mps` is flaky with seq2seq) |
| `LT_REMUX` | `1` | convert unplayable files to a browser-safe mp4 (`0` disables) |
| `LT_CACHE` | `~/.cache/live-transcribe` | derived PCM + remuxed mp4 |

```sh
HF_TOKEN=hf_xxx LT_LOOKAHEAD=15 make run
```

## Notes and limits

- **Playback formats.** The browser decides what plays, and the *container is
  not the whole story* — an HEVC `.mp4` is refused by Chromium even though the
  extension looks fine. So every file is probed first: if it's already a
  browser-safe container (mp4/mov/m4v/webm) with browser-safe codecs
  (H.264/VP8/VP9/AV1 + AAC/MP3/Opus/Vorbis) it is served untouched. Otherwise
  it's converted to H.264/AAC mp4 once and cached; on Apple Silicon that uses
  `h264_videotoolbox`. Conversion runs in the background with a progress
  readout, and the UI waits for it rather than blocking the open call.
  `LT_REMUX=0` disables this and serves the original.
- Cached conversions are re-probed before use, so a cache written by an older
  version is discarded instead of served.
- If conversion fails, the original is served anyway and the reason is shown:
  transcription still works even when playback can't.
- The buffer overshoots by up to one chunk (30s), since whole chunks are
  transcribed. It guarantees *at least* `LT_LOOKAHEAD` seconds ahead.
- Only one video is open at a time; `/media` serves the current one.
- The file panel is driven by `osascript` (`app.py:choose_file`). macOS may ask
  once for permission to control System Events, which is only used to bring the
  panel to the front; if denied, the panel still works but may open behind the
  browser. Off macOS, the UI falls back to a path field.
- Chunk edges snap to detected silence within ±6s of each 30s mark, and never
  extend past Whisper's 30s window.
- Chunk planning re-runs `silencedetect` on every open (a fast audio-only
  pass); the PCM extraction itself is cached by file identity.

## Interface

The UI follows Apple's fluid-interface guidance (WWDC *Designing Fluid
Interfaces*): feedback fires on pointer-*down* rather than release, the
timeline tracks the pointer 1:1 with `setPointerCapture`, chrome is a
translucent `backdrop-filter` material rather than opaque bars, and tracking is
size-specific — tight on the large display type, near zero on body, tabular
numerals on every clock so it never jitters.

It also honors `prefers-reduced-motion`, `prefers-reduced-transparency` and
`prefers-contrast`, and adapts to light and dark via `prefers-color-scheme`.

## Tests

```sh
make test
```

Covers chunk planning and the scheduler (look-ahead bound, idling, seek
re-serving from cache, error surfacing) against a fake backend, so no model or
ffmpeg is needed.
