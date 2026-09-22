"""No-model tests for the Whisper backend: path resolution and cue mapping."""

import sys
import types

import numpy as np

import backends
from backends import WhisperASR, resolve_model


# ------------------------------------------------------------------ resolve


def test_resolve_model_prefers_existing_local_dir(tmp_path):
    assert resolve_model(str(tmp_path)) == str(tmp_path)


def test_resolve_model_passes_through_repo_ids():
    assert resolve_model("mlx-community/whisper-large-v3-mlx") == \
        "mlx-community/whisper-large-v3-mlx"


# ------------------------------------------------------------------ backend


def test_backend_auto_detects_and_translates():
    asr = backends.get_backend()
    assert isinstance(asr, WhisperASR)
    assert asr.language is None  # None -> whisper auto-detects
    assert asr.task == "translate"
    assert asr.model == backends.TRANSLATE_ASR


def test_whisper_maps_segments_with_offset_and_skips_silence(monkeypatch):
    fake = types.ModuleType("mlx_whisper")
    fake.transcribe = lambda audio, **kw: {
        "segments": [
            {"start": 0.5, "end": 1.5, "text": " hello ", "no_speech_prob": 0.1},
            {"start": 1.5, "end": 2.0, "text": " hmm ", "no_speech_prob": 0.9},
            {"start": 2.0, "end": 2.5, "text": "   ", "no_speech_prob": 0.1},
        ]
    }
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    cues = WhisperASR("whatever").run(np.zeros(16000, dtype=np.float32), 10.0)
    assert [(c.start, c.end, c.source) for c in cues] == [(10.5, 11.5, "hello")]
