import numpy as np

from plctestbench.settings import (
    CrossfadeFunction,
    CrossfadeSettings,
    CrossfadeType,
    Settings,
)

from .filters import LinkwitzRileyCrossover
from .settings import CrossfadeFunction, CrossfadeType
from .utils import recursive_split_audio


def power_crossfade(settings: CrossfadeSettings, length_in_samples: int) -> np.array:
    return np.array(
        [x ** settings.get("exponent") for x in np.linspace(0, 1, length_in_samples)]
    )


def sinusoidal_crossfade(
    settings: CrossfadeSettings, length_in_samples: int
) -> np.array:
    return np.sin(np.linspace(0, np.pi / 2, length_in_samples))


def hann_crossfade(settings: CrossfadeSettings, length_in_samples: int) -> np.array:
    return np.hanning(length_in_samples * 2)[settings.length_in_samples]


class Crossfade(object):
    def __init__(self, settings: Settings, crossfade_settings: Settings) -> None:
        self.settings = settings
        self.crossfade_settings = crossfade_settings
        self.fs = settings.get("fs")
        self.length = self.crossfade_settings.get("length")
        self.length_in_samples = round(self.length * self.fs * 0.001)
        self._ongoing = False
        self.idx = 0

        self.function = self.crossfade_settings.get("function")
        if self.function == CrossfadeFunction.power:
            self.crossfade_buffer_a = power_crossfade(
                self.crossfade_settings, self.length_in_samples
            )
        elif self.function == CrossfadeFunction.sinusoidal:
            self.crossfade_buffer_a = sinusoidal_crossfade(
                self.crossfade_settings, self.length_in_samples
            )

        self.type = self.crossfade_settings.get("type")
        if self.type == CrossfadeType.power:
            self.crossfade_buffer_b = (1 - self.crossfade_buffer_a**2) ** (1 / 2)
        elif self.type == CrossfadeType.amplitude:
            self.crossfade_buffer_b = 1 - self.crossfade_buffer_a

    def __call__(self, prediction: np.ndarray, buffer: np.ndarray = None) -> np.ndarray:
        """ """
        # One-pad the crossfade_buffer_a to match the length of the buffer in case it is shorter
        if self._ongoing:
            if (
                np.shape(self.crossfade_buffer_a)[0] - self.idx
                < np.shape(prediction)[0]
            ):
                self.crossfade_buffer_a = np.pad(
                    self.crossfade_buffer_a,
                    (0, len(prediction) - (len(self.crossfade_buffer_a) - self.idx)),
                    "constant",
                    constant_values=(1),
                )
                self.crossfade_buffer_b = np.pad(
                    self.crossfade_buffer_b,
                    (0, len(prediction) - (len(self.crossfade_buffer_b) - self.idx)),
                    "constant",
                    constant_values=(0),
                )
            if buffer is None:
                buffer = np.zeros_like(prediction)
            output_buffer = (
                prediction
                * self.crossfade_buffer_b[
                    self.idx : self.idx + len(prediction), np.newaxis
                ]
                + buffer
                * self.crossfade_buffer_a[
                    self.idx : self.idx + len(prediction), np.newaxis
                ]
            )
            self.idx += len(prediction)
        else:
            output_buffer = buffer
        return output_buffer

    def start(self) -> None:
        self._ongoing = True
        self.idx = 0

    def ongoing(self) -> bool:
        if self.idx >= self.length_in_samples:
            self._ongoing = False
        return self._ongoing


class MultibandCrossfade(object):
    def __init__(self, settings: Settings, crossfade_settings: list) -> None:
        self.settings = settings
        self.crossfade_settings = crossfade_settings
        self.frequencies = self.settings.get("crossfade_frequencies")
        assert len(self.frequencies) + 1 == len(
            self.crossfade_settings
        ), "Number of bands and number of crossfade settings do not match"
        self.crossover_order = self.settings.get("crossover_order")
        self.fs = self.settings.get("fs")
        self.crossovers = [
            LinkwitzRileyCrossover(self.crossover_order, freq, self.settings.get("fs"))
            for freq in self.frequencies
        ]
        self.crossfades = [
            Crossfade(self.settings, xfade_settings)
            for xfade_settings in self.crossfade_settings
        ]

    def __call__(self, prediction: np.ndarray, buffer: np.ndarray = None) -> np.ndarray:
        """ """
        if buffer is None:
            buffer = np.zeros_like(prediction)
        prediction_bands = recursive_split_audio(prediction, self.crossovers)
        buffer_bands = recursive_split_audio(buffer, self.crossovers)
        output_bands = []
        for pred, buff, xfade in zip(prediction_bands, buffer_bands, self.crossfades):
            output_bands.append(xfade(pred, buff))
        output = np.sum(output_bands, axis=0)
        return output

    def start(self) -> None:
        for xfade in self.crossfades:
            xfade.start()

    def ongoing(self) -> bool:
        return any([xfade.ongoing() for xfade in self.crossfades])
