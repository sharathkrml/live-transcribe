"""ASR + translation backends and the language-pair profile registry.

Adding a language pair is one entry in PROFILES. Every backend exposes
`run(audio, offset) -> list[Cue]` so the scheduler doesn't care what's behind
it.
"""

from __future__ import annotations

import os
from pathlib import Path

from pipeline import Cue

DEFAULT_ASR = os.environ.get("LT_ASR_MODEL", "mlx-community/whisper-large-v3-turbo")
KOTOBA_ASR = os.environ.get("LT_JA_ASR_MODEL", "kaiinui/kotoba-whisper-v2.0-mlx")
PARAKEET_ASR = os.environ.get("LT_PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v2")
NLLB_MODEL = os.environ.get("LT_MT_MODEL", "facebook/nllb-200-distilled-600M")


def hf_token() -> str | None:
    """HF_TOKEN (or the legacy HUGGING_FACE_HUB_TOKEN)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def resolve_model(repo_or_path: str) -> str:
    """A local dir (e.g. an HF snapshot) is used as-is; otherwise the repo id.

    Both mlx-whisper (`path_or_hf_repo`) and parakeet-mlx (`hf_id_or_path`)
    accept either, so anything already sitting in the HF cache on this machine
    works offline with no download.
    """
    p = Path(os.path.expanduser(repo_or_path))
    return str(p) if p.is_dir() else repo_or_path


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


class ParakeetASR:
    """NVIDIA Parakeet via MLX (`parakeet-mlx`). English-only, no word prompt."""

    def __init__(self, model: str = PARAKEET_ASR):
        self.model = model
        self._m = None

    def _load(self):
        if self._m is None:
            from parakeet_mlx import from_pretrained

            self._m = from_pretrained(resolve_model(self.model))
        return self._m

    def run(self, audio, offset: float) -> list[Cue]:
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        if len(audio) < 160:  # shorter than one hop: nothing to decode
            return []
        m = self._load()
        mel = get_logmel(mx.array(audio, dtype=mx.float32), m.preprocessor_config)
        result = m.generate(mel)[0]
        cues: list[Cue] = []
        for s in result.sentences:
            text = (s.text or "").strip()
            if text:
                cues.append(Cue(offset + s.start, offset + s.end, text))
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

    def __init__(self, asr: WhisperASR | ParakeetASR):
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
    "en-en-parakeet": lambda: PassthroughStage(ParakeetASR(PARAKEET_ASR)),
    "ja-ja": lambda: PassthroughStage(WhisperASR(KOTOBA_ASR, "ja", "transcribe")),
    "ja-en": lambda: TwoStage(
        WhisperASR(KOTOBA_ASR, "ja", "transcribe"),
        NLLBTranslator(),
    ),
    "ja-en-fast": lambda: WhisperASR(DEFAULT_ASR, "ja", "translate"),
}

PROFILE_INFO = {
    "en-en": {"label": "English", "detail": "Whisper large-v3 turbo", "needs_mt": False},
    "en-en-parakeet": {"label": "English (Parakeet)", "detail": "Parakeet TDT 0.6B", "needs_mt": False},
    "ja-ja": {"label": "Japanese", "detail": "Kotoba Whisper", "needs_mt": False},
    "ja-en": {"label": "Japanese → English", "detail": "Kotoba + NLLB", "needs_mt": True},
    "ja-en-fast": {"label": "Japanese → English (fast)", "detail": "Whisper built-in translate", "needs_mt": False},
}


def _short(repo_or_path: str) -> str:
    return repo_or_path.rstrip("/").rsplit("/", 1)[-1]


def build_extra_profiles(specs: str) -> dict[str, tuple]:
    """Parse `LT_ASR_MODELS` (`name=repo, ...`) into profile entries.

    Engine is inferred: anything with "parakeet" in the repo/path gets
    ParakeetASR, everything else WhisperASR (English transcribe). Both accept
    HF repo ids (served from the local HF cache when present) and local
    snapshot dirs. Built-in names are never overridden.
    """
    out: dict[str, tuple] = {}
    for spec in specs.split(","):
        spec = spec.strip()
        if not spec or "=" not in spec:
            continue
        name, model = (part.strip() for part in spec.split("=", 1))
        if not name or not model or name in PROFILES:
            continue
        if "parakeet" in model.lower():
            out[name] = (
                lambda m=model: PassthroughStage(ParakeetASR(m)),
                {"label": f"English ({_short(model)})", "detail": _short(model),
                 "needs_mt": False},
            )
        else:
            out[name] = (
                lambda m=model: PassthroughStage(WhisperASR(m, "en", "transcribe")),
                {"label": f"English ({_short(model)})", "detail": _short(model),
                 "needs_mt": False},
            )
    return out


def _register_extra() -> None:
    for name, (factory, info) in build_extra_profiles(os.environ.get("LT_ASR_MODELS", "")).items():
        PROFILES[name] = factory
        PROFILE_INFO[name] = info


_register_extra()
del _register_extra

_cache: dict[str, object] = {}


def get_profile(name: str):
    """Backends are singletons, so switching back and forth doesn't reload weights."""
    if name not in PROFILES:
        raise KeyError(f"unknown profile: {name}")
    if name not in _cache:
        _cache[name] = PROFILES[name]()
    return _cache[name]
