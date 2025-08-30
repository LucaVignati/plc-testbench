import librosa
import numpy as np
from essentia.standard import NSGConstantQ


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

        cqt = NSGConstantQ(
            minFrequency=min_frequency,
            maxFrequency=max_frequency,
            binsPerOctave=bins_per_octave,
            minimumWindow=minimum_window,
            inputSize=input_size,
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
