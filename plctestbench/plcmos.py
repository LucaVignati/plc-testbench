"""
https://github.com/microsoft/PLC-Challenge/blob/main/PLCMOS/plc_mos.py
"""

import os
import math

import numpy as np
from numpy.fft import rfft
from numpy.lib.stride_tricks import as_strided

import onnxruntime as ort


class PLCMOSEstimator:
    def __init__(self, model_version=2, embed_rounds=15):
        """
        Initialize a PLC-MOS model of a given version. There are currently three model versions available, v0alpha,
        (intrusive), v0 (both non-intrusive and intrusive available, renamed from v1 to match naming in paper),
        and v2 (non-intrusive, includes validation model as well as final release version).

        The default is to use the final v2 model. This is the no-holdout version of the model described in the
        PLCMOS paper published at INTERSPEECH 2023.
        """
        self.model_version = str(model_version)
        model_paths = {
            # v0alpha model: two encoders (no shared weights) plus dense, ~.93 PCC on v1 test set
            "0alpha": [("../dl_models/plcmos_v0.onnx", 999999999999), (None, 0)],
            # v0 models:
            # * Intrusive (old default): ~.0.997 PCC on v1 test set, ~0.927 PCC / ~0.857 SRCC on v2 test set
            # * Nonintrusive: ~.0.987 PCC on v1 test set
            "0": [
                ("../dl_models/plcmos_v1_intrusive.onnx", 768),
                ("../dl_models/plcmos_v1_nonintrusive.onnx", 999999999999),
            ],
            # v2 model for validation, "cool helmet"
            # ~0.97 PCC / ~0.95 SRCC / ~0.09 MAE on v2 test set
            "2-val": [(None, 0), ("../dl_models/plcmos_v2_val.onnx", 999999999999)],
            # v2 model final run (all data - nothing held out), "lucid garden", current default
            # Not reporting metrics (invalid, since no holdout) but reasonable to assume as good as or better than v2-val
            "2": [(None, 0), ("../dl_models/plcmos_v2.onnx", 999999999999)],
        }
        model_use_embed = {
            "0alpha": False,
            "0": False,
            "2-val": True,  # v2 models use embeds
            "2": True,  # v2 models use embeds
        }

        self.sessions = []
        self.max_lens = []
        for path, max_len in model_paths[self.model_version]:
            if path is not None:
                file_dir = os.path.dirname(os.path.realpath(__file__))
                self.sessions.append(ort.InferenceSession(os.path.join(file_dir, path)))
                self.max_lens.append(max_len)
            else:
                self.sessions.append(None)
                self.max_lens.append(0)
        self.use_embed = False
        self.embed_rounds = 1
        if model_use_embed[self.model_version]:
            self.use_embed = True
            self.embed_rounds = embed_rounds

    def logpow_dns(self, sig, floor=-30.0):
        """
        Compute log power of complex spectrum.

        Floor any -`np.inf` value to (nonzero minimum + `floor`) dB.
        If all values are 0s, floor all values to -80 dB.
        """
        log10e = np.log10(np.e)
        pspec = sig.real**2 + sig.imag**2
        zeros = pspec == 0
        logp = np.empty_like(pspec)
        if np.any(~zeros):
            logp[~zeros] = np.log(pspec[~zeros])
            logp[zeros] = np.log(pspec[~zeros].min()) + floor / 10 / log10e
        else:
            logp.fill(-80 / 10 / log10e)

        return logp

    def hop2hsize(self, wind, hop):
        """
        Convert hop fraction to integer size if necessary.
        """
        if hop >= 1:
            assert isinstance(hop, int), "Hop size must be integer!"
            return hop
        else:
            assert 0 < hop < 1, "Hop fraction has to be in range (0,1)!"
            return int(len(wind) * hop)

    def stana(self, sig, sr, wind, hop, synth=False, center=False):
        """
        Short term analysis by windowing
        """
        ssize = len(sig)
        fsize = len(wind)
        hsize = self.hop2hsize(wind, hop)
        if synth:
            sstart = hsize - fsize  # int(-fsize * (1-hfrac))
        elif center:
            sstart = -int(len(wind) / 2)  # odd window centered at exactly n=0
        else:
            sstart = 0
        send = ssize

        nframe = math.ceil((send - sstart) / hsize)
        # Calculate zero-padding sizes
        zpleft = -sstart
        zpright = (nframe - 1) * hsize + fsize - zpleft - ssize
        if zpleft > 0 or zpright > 0:
            sigpad = np.zeros(ssize + zpleft + zpright, dtype=sig.dtype)
            sigpad[zpleft : len(sigpad) - zpright] = sig
        else:
            sigpad = sig

        return (
            as_strided(
                sigpad,
                shape=(nframe, fsize),
                strides=(sig.itemsize * hsize, sig.itemsize),
            )
            * wind
        )

    def stft(self, sig, sr, wind, hop, nfft):
        """
        Compute STFT: window + rfft
        """
        frames = self.stana(sig, sr, wind, hop, synth=True)
        return rfft(frames, n=nfft)

    def stft_transform(self, audio, dft_size=512, hop_fraction=0.5, sr=16000):
        """
        Compute STFT parameters, then compute STFT
        """
        window = np.hamming(dft_size + 1)
        window = window[:-1]
        amp = np.abs(self.stft(audio, sr, window, hop_fraction, dft_size))
        feat = self.logpow_dns(amp, floor=-120.0)
        return feat / 20.0

    def run(
        self,
        audio_degraded,
        sr_degraded,
        audio_clean=None,
        combined=True,
        return_intermediate_scores=False,
    ):
        """
        Run the PLCMOS model and return the estimated MOS for the given audio. Only the degraded audio is used for the v2
        models, clean audio is not neccesary unless a legacy model is intended to be used.

        Audio data should be 16kHz, mono, [-1, 1] range.

        While all models are trained to work on 16khz audio, anecdotally, at least the v2 models seem to work
        quite well on downsampled higher-rate audio as well, but we have not validated this usage - ymmv.
        """
        assert sr_degraded == 16000
        np.random.seed(23)
        audio_features_degraded = np.float32(self.stft_transform(audio_degraded))[
            np.newaxis, np.newaxis, ...
        ]
        mos = 0
        intermediate_scores = {}
        for i in range(self.embed_rounds):
            rater_embed = np.random.normal(size=(1, 64))
            if audio_clean is not None:
                session = self.sessions[0]
                assert (
                    session is not None
                ), "Intrusive model not available for this model version."
                audio_features_clean = np.float32(self.stft_transform(audio_clean))[
                    np.newaxis, np.newaxis, ...
                ]
                assert (
                    len(audio_features_clean) <= self.max_lens[0]
                ), "Maximum input length exceeded"
                assert (
                    len(audio_features_degraded) <= self.max_lens[0]
                ), "Maximum input length exceeded"
                if self.use_embed:
                    onnx_inputs = {
                        "degraded_audio": audio_features_degraded,
                        "clean_audio": audio_features_clean,
                        "rater_embed": np.array(rater_embed, dtype=np.float32).reshape(
                            1, -1
                        ),
                    }
                else:
                    onnx_inputs = {
                        "degraded_audio": audio_features_degraded,
                        "clean_audio": audio_features_clean,
                    }
                mos_val = float(session.run(None, onnx_inputs)[0])
                intermediate_scores[str(i) + "_int"] = mos_val
                mos += mos_val

            if audio_clean is None or (not self.sessions[1] is None and combined):
                session = self.sessions[1]
                assert (
                    session is not None
                ), "Nonintrusive model not available for this model version."
                assert (
                    len(audio_features_degraded) <= self.max_lens[1]
                ), "Maximum input length exceeded"
                if self.use_embed:
                    onnx_inputs = {
                        "degraded_audio": audio_features_degraded,
                        "rater_embed": np.array(rater_embed, dtype=np.float32).reshape(
                            1, -1
                        ),
                    }
                else:
                    onnx_inputs = {"degraded_audio": audio_features_degraded}
                mos_val = float(session.run(None, onnx_inputs)[0])
                intermediate_scores[str(i) + "_nonint"] = mos_val
                mos += mos_val

            if (
                combined
                and (not self.sessions[0] is None or self.sessions[1] is None)
                and not audio_clean is None
            ):
                mos /= 2.0
        if not return_intermediate_scores:
            return mos / self.embed_rounds
        else:
            return mos / self.embed_rounds, intermediate_scores
