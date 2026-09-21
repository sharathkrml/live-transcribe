"""ASR + translation backends and the language-pair profile registry.

Adding a language pair is one entry in PROFILES. Every backend exposes
`run(audio, offset) -> list[Cue]` so the scheduler doesn't care what's behind
it.
"""

from __future__ import annotations

import os

from pipeline import Cue

DEFAULT_ASR = os.environ.get("LT_ASR_MODEL", "mlx-community/whisper-large-v3-turbo")
KOTOBA_ASR = os.environ.get("LT_JA_ASR_MODEL", "kaiinui/kotoba-whisper-v2.0-mlx")
NLLB_MODEL = os.environ.get("LT_MT_MODEL", "facebook/nllb-200-distilled-600M")


def hf_token() -> str | None:
    """HF_TOKEN (or the legacy HUGGING_FACE_HUB_TOKEN)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# mlx-whisper calls huggingface_hub internally and reads the token from the
# environment, so make sure a legacy var is visible under the standard name.
if hf_token():
    os.environ.setdefault("HF_TOKEN", hf_token())


class WhisperASR:
    """Whisper via MLX. `task="translate"` is Whisper's built-in X->English."""

    def __init__(self, model: str = DEFAULT_ASR, language: str = "en", task: str = "transcribe"):
        self.model = model
        self.language = language
        self.task = task
        self._prompt: str | None = None

    def run(self, audio, offset: float) -> list[Cue]:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
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


class NLLBTranslator:
    """Sentence-level translation with NLLB. CPU by default; MPS is flaky here."""

    def __init__(self, model: str = NLLB_MODEL, src: str = "jpn_Jpan", tgt: str = "eng_Latn"):
        self.model_id = model
        self.src = src
        self.tgt = tgt
        self.device = os.environ.get("LT_MT_DEVICE", "cpu")
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        token = hf_token()
        self._tok = AutoTokenizer.from_pretrained(self.model_id, src_lang=self.src, token=token)
        self._mt = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, token=token)
        self._mt = self._mt.to(self.device).eval()
        self._bos = self._tok.convert_tokens_to_ids(self.tgt)
        self._loaded = True

    def run(self, cues: list[Cue]) -> list[Cue]:
        if not cues:
            return cues
        self._load()
        texts = [c.source for c in cues]
        batch = self._tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with self._torch.inference_mode():
            out = self._mt.generate(
                **batch,
                forced_bos_token_id=self._bos,
                num_beams=1,  # latency over quality; the look-ahead hides little
            )
        for cue, ids in zip(cues, out):
            cue.target = self._tok.batch_decode([ids], skip_special_tokens=True)[0].strip()
        return cues


class PassthroughStage:
    """ASR only: the source text is the displayed text."""

    def __init__(self, asr: WhisperASR):
        self.asr = asr

    def run(self, audio, offset: float) -> list[Cue]:
        return [
            Cue(c.start, c.end, c.source, c.source)
            for c in self.asr.run(audio, offset)
        ]


class TwoStage:
    """ASR in the source language, then translate the chunk's lines in one batch."""

    def __init__(self, asr: WhisperASR, mt: NLLBTranslator):
        self.asr = asr
        self.mt = mt

    def run(self, audio, offset: float) -> list[Cue]:
        return self.mt.run(self.asr.run(audio, offset))


PROFILES = {
    "en-en": lambda: PassthroughStage(WhisperASR(DEFAULT_ASR, "en", "transcribe")),
    "ja-ja": lambda: PassthroughStage(WhisperASR(KOTOBA_ASR, "ja", "transcribe")),
    "ja-en": lambda: TwoStage(
        WhisperASR(KOTOBA_ASR, "ja", "transcribe"),
        NLLBTranslator(),
    ),
    "ja-en-fast": lambda: WhisperASR(DEFAULT_ASR, "ja", "translate"),
}

PROFILE_INFO = {
    "en-en": {"label": "English", "detail": "Whisper large-v3 turbo", "needs_mt": False},
    "ja-ja": {"label": "Japanese", "detail": "Kotoba Whisper", "needs_mt": False},
    "ja-en": {"label": "Japanese → English", "detail": "Kotoba + NLLB", "needs_mt": True},
    "ja-en-fast": {"label": "Japanese → English (fast)", "detail": "Whisper built-in translate", "needs_mt": False},
}

_cache: dict[str, object] = {}


def get_profile(name: str):
    """Backends are singletons, so switching back and forth doesn't reload weights."""
    if name not in PROFILES:
        raise KeyError(f"unknown profile: {name}")
    if name not in _cache:
        _cache[name] = PROFILES[name]()
    return _cache[name]
