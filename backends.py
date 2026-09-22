"""Whisper backend: auto-detect the spoken language, translate to English.

Whisper's built-in `task="translate"` does any-language -> English in one pass,
so there is a single backend and no separate translation stage.
"""

from __future__ import annotations

import os
from pathlib import Path

from pipeline import Cue

# large-v3-turbo silently ignores task="translate" (it just transcribes), so use
# the full large-v3 weights, which are translate-capable.
TRANSLATE_ASR = os.environ.get("LT_TRANSLATE_MODEL", "mlx-community/whisper-large-v3-mlx")


def hf_token() -> str | None:
    """HF_TOKEN (or the legacy HUGGING_FACE_HUB_TOKEN)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def resolve_model(repo_or_path: str) -> str:
    """A local dir (e.g. an HF snapshot) is used as-is; otherwise the repo id.

    mlx-whisper (`path_or_hf_repo`) accepts either, so anything already sitting
    in the HF cache on this machine works offline with no download.
    """
    p = Path(os.path.expanduser(repo_or_path))
    return str(p) if p.is_dir() else repo_or_path


# mlx-whisper calls huggingface_hub internally and reads the token from the
# environment, so make sure a legacy var is visible under the standard name.
if hf_token():
    os.environ.setdefault("HF_TOKEN", hf_token())


class WhisperASR:
    """Whisper via MLX. `task="translate"` is Whisper's built-in X->English."""

    def __init__(self, model: str = TRANSLATE_ASR, language: str | None = None,
                 task: str = "translate"):
        self.model = model
        self.language = language
        self.task = task
        self._prompt: str | None = None

    def run(self, audio, offset: float) -> list[Cue]:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=resolve_model(self.model),
            language=self.language,
            task=self.task,
            initial_prompt=self._prompt,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )

        cues: list[Cue] = []
        for seg in result.get("segments", []):
            if (seg.get("no_speech_prob") or 0.0) > 0.6:
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            cues.append(Cue(offset + seg["start"], offset + seg["end"], text))

        if cues:
            tail = " ".join(c.source for c in cues[-8:])
            self._prompt = tail[-200:]
        return cues


_backend: WhisperASR | None = None


def get_backend() -> WhisperASR:
    """The one resident model: auto-detect + translate to English."""
    global _backend
    if _backend is None:
        _backend = WhisperASR(TRANSLATE_ASR, None, "translate")
    return _backend
