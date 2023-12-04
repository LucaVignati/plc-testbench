import numpy as np
from plctestbench.settings import Settings

def power_crossfade(settings: Settings) -> np.array:
    return np.array([x ** settings.get("exponent") for x in np.linspace(0, 1, settings.length_in_samples)])

def sinusoidal_crossfade(settings: Settings) -> np.array:
    return np.sin(np.linspace(0, np.pi/2, settings.length_in_samples))

class Crossfade(object):
    def __init__(self, settings: Settings, crossfade_settings: Settings) -> None:
        self.settings = settings
        self.crossfade_settings = crossfade_settings
        self.fs = settings.get("fs")
        self.length = self.crossfade_settings.get("length")
        self.crossfade_settings.length_in_samples = round(self.length * self.fs * 0.001)
        self._ongoing = False
        self.idx = 0
        
        self.function = crossfade_settings.get("function")
        if self.function == "power":
            self.crossfade_buffer_a = power_crossfade(crossfade_settings)
        elif self.function == "sinusoidal":
            self.crossfade_buffer_a = sinusoidal_crossfade(crossfade_settings)

        self.type = crossfade_settings.get("type")
        if self.type == "power":
            self.crossfade_buffer_b = (1 - self.crossfade_buffer_a ** 2) ** 1/2
        elif self.type == "amplitude":
            self.crossfade_buffer_b = 1 - self.crossfade_buffer_a

    def __call__(self, prediction: np.ndarray, buffer: np.ndarray = None) -> np.ndarray:
        '''
        '''
        # One-pad the crossfade_buffer_a to match the length of the buffer in case it is shorter
        if np.shape(self.crossfade_buffer_a)[0] - self.idx < np.shape(prediction)[0]:
            self.crossfade_buffer_a = np.pad(self.crossfade_buffer_a, (1, len(prediction) - (len(self.crossfade_buffer_a) - self.idx)), 'constant')
            self.crossfade_buffer_b = np.pad(self.crossfade_buffer_b, (0, len(prediction) - (len(self.crossfade_buffer_b) - self.idx)), 'constant')
        if buffer is None:
            buffer = np.zeros_like(prediction)
        for idx in range(len(prediction)):
            output_buffer = prediction[idx] * self.crossfade_buffer_b[self.idx] + buffer[idx] * self.crossfade_buffer_a[self.idx]
            self.idx += 1
        return output_buffer

    def start(self) -> None:
        self._ongoing = True
        self.idx = 0

    def ongoing(self) -> bool:
        if self.idx >= len(self.crossfade_buffer_a):
            self._ongoing = False
        return self._ongoing