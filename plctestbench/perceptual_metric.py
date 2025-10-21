from __future__ import annotations

import librosa
import numpy as np

try:
    from essentia.standard import NSGConstantQ

    _cqt_impl_class = NSGConstantQ

except ImportError:

    class LibrosaCQTWrapper:
        def __init__(
            self,
            inputSize: int,
            minFrequency: float,
            maxFrequency: float,
            binsPerOctave: int,
            minimumWindow: int,
            sampleRate: int,
        ):
            self.inputSize = inputSize
            self.minFrequency = minFrequency
            self.maxFrequency = maxFrequency
            self.binsPerOctave = binsPerOctave
            self.minimumWindow = minimumWindow
            self.sampleRate = sampleRate

        def __call__(self, y):
            n_bins = int(
                self.binsPerOctave * np.log2(self.maxFrequency / self.minFrequency)
            )

            return librosa.cqt(
                y,
                sr=self.sampleRate,
                hop_length=512,
                fmin=self.minFrequency,
                n_bins=n_bins,
                bins_per_octave=self.binsPerOctave,
                window="hann",
                filter_scale=1,
                norm=1,
                pad_mode="constant",
                dtype=np.complex64,
            )

    _cqt_impl_class = LibrosaCQTWrapper


class PerceptualMetric(object):

    def __init__(
        self,
        min_frequency: float,
        max_frequency: float,
        bins_per_octave: int,
        n_bins: int,
        minimum_window: int,
        input_size: int,
        fs: int,
        intorno_length: int,
    ) -> None:
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.bins_per_octave = bins_per_octave
        self.n_bins = n_bins
        self.minimum_window = minimum_window
        self.input_size = input_size
        self.fs = fs
        self.intorno_length = intorno_length

        cqt = _cqt_impl_class(
            inputSize=input_size,
            minFrequency=min_frequency,
            maxFrequency=max_frequency,
            binsPerOctave=bins_per_octave,
            minimumWindow=minimum_window,
            sampleRate=fs,
        )
        self.transform = lambda original, reconstructed: {
            "original": cqt(original)[0],
            "reconstructed": cqt(reconstructed)[0],
        }
        self.frequency_axis = librosa.cqt_frequencies(
            input_size, fmin=min_frequency, bins_per_octave=bins_per_octave
        )

    def spectrogram(self, original, reconstructed):
        return self.transform(original, reconstructed)

    def __call__(self, spectrogram):
        spectrogram_difference = spectrogram["reconstructed"] - spectrogram["original"]
        spectrogram_difference_mag = np.abs(spectrogram_difference)
        ref_value = max(
            np.max(np.abs(spectrogram["original"])),
            np.max(np.abs(spectrogram["reconstructed"])),
        )
        spectrogram_original_db = librosa.amplitude_to_db(
            np.abs(spectrogram["original"]), ref=ref_value
        )
        spectrogram_difference_db = librosa.amplitude_to_db(
            spectrogram_difference_mag, ref=ref_value
        )
        spectrogram_reconstructed_residual = (
            spectrogram_difference_db - spectrogram_original_db
        )

        spectrogram_reconstructed_residual = np.where(
            spectrogram_reconstructed_residual < 0,
            0,
            spectrogram_reconstructed_residual,
        )
        perc_metric = np.sum(spectrogram_reconstructed_residual) / 10000

        return perc_metric
