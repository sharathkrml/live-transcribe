"""No-model tests for the ASR registry: spec parsing, path resolution,
and Parakeet sentence -> Cue mapping (fake model, no weights)."""

from dataclasses import dataclass

import numpy as np

import backends
from backends import ParakeetASR, build_extra_profiles, resolve_model


# ------------------------------------------------------------------ resolve


def test_resolve_model_prefers_existing_local_dir(tmp_path):
    assert resolve_model(str(tmp_path)) == str(tmp_path)


def test_resolve_model_passes_through_repo_ids():
    assert resolve_model("mlx-community/parakeet-tdt-0.6b-v2") == \
        "mlx-community/parakeet-tdt-0.6b-v2"


# -------------------------------------------------------------------- specs


def test_extra_specs_pick_engine_by_name():
    extra = build_extra_profiles(
        "sm=mlx-community/whisper-small-mlx,"
        "pk=mlx-community/parakeet-tdt-0.6b-v2"
    )
    assert set(extra) == {"sm", "pk"}
    assert isinstance(extra["sm"][0]().asr, backends.WhisperASR)
    assert isinstance(extra["pk"][0]().asr, backends.ParakeetASR)
    assert extra["pk"][1]["needs_mt"] is False


def test_extra_specs_skip_blanks_and_builtins():
    extra = build_extra_profiles(" ,noequals,en-en=foo/bar,en-en = foo/bar, ok = mlx-community/whisper-tiny ")
    assert set(extra) == {"ok"}


def test_empty_specs_give_nothing():
    assert build_extra_profiles("") == {}


# ------------------------------------------------------------------ profiles


def test_auto_en_profile_auto_detects_and_translates():
    asr = backends.PROFILES["auto-en"]()
    assert isinstance(asr, backends.WhisperASR)
    assert asr.language is None  # None -> whisper auto-detects
    assert asr.task == "translate"
    assert backends.PROFILE_INFO["auto-en"]["needs_mt"] is False


# ------------------------------------------------------------- parakeet run


@dataclass
class _Sentence:
    text: str
    start: float
    end: float


@dataclass
class _Result:
    sentences: list


class _FakeModel:
    sentences = [_Sentence(" hello world ", 0.5, 1.5), _Sentence("  ", 1.5, 2.0)]

    def __init__(self):
        from parakeet_mlx.audio import PreprocessArgs

        # Real mel config so get_logmel runs; decode itself is faked below.
        self.preprocessor_config = PreprocessArgs(
            sample_rate=16000, normalize="per_feature", window_size=0.025,
            window_stride=0.01, window="hann", features=128, n_fft=512,
            dither=0.0,
        )

    def generate(self, mel):
        assert mel is not None
        return [_Result(self.sentences)]


def test_parakeet_maps_sentences_with_offset():
    asr = ParakeetASR("whatever")
    asr._m = _FakeModel()
    cues = asr.run(np.zeros(16000, dtype=np.float32), 10.0)
    assert [(c.start, c.end, c.source) for c in cues] == [(10.5, 11.5, "hello world")]


def test_parakeet_short_audio_skipped_without_loading():
    asr = ParakeetASR("whatever")
    assert asr.run(np.zeros(10, dtype=np.float32), 0.0) == []
    assert asr._m is None  # model must not load for a <10ms sliver
